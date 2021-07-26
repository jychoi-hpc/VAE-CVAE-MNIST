import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# %%
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        #import pdb; pdb.set_trace()
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        #import pdb; pdb.set_trace()
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(self._device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

# %%
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(self._device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        loss = self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

# %%
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.leaky_relu(x)

# %%
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

    def forward(self, x):
        return self._block(x)

class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvTBlock, self).__init__()
        self._block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        return self._block(x)

# %%
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        print ("in_channels, num_hiddens:", in_channels, num_hiddens)

        self._conv_0 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=4,
                                 stride=2, padding=1)
        # (2020/11) possible kernel size
        # kernel_size=4, stride=2, padding=1
        # kernel_size=3, stride=2, padding=1
        # kernel_size=2, stride=2, padding=0
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=3, stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=3, stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3, stride=2, padding=1)

        # self._block = nn.Sequential(
        #     ConvBlock(in_channels, num_hiddens//2),
        #     ConvBlock(num_hiddens//2, num_hiddens),
        #     ConvBlock(num_hiddens, num_hiddens),
        # )

        # kernel_size=3, stride=1, padding=1
        # kernel_size=2, stride=2, padding=0
        self._conv_4 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs, da=None):
        # (2020/11) Testing with resize
        x = inputs
        if da is not None:
            _d = torch.zeros_like(x)
            _a = torch.zeros_like(x)
            _d[:,:,:,:] = da[:,0,np.newaxis,np.newaxis,np.newaxis]
            _a[:,:,:,:] = da[:,1,np.newaxis,np.newaxis,np.newaxis]
            _da = torch.cat((_d, _a), dim=1)
            x = torch.cat((x, _d, _a), dim=1)
        # print ('ENC #1:', x.shape)
        # if self._rescale is not None:
        #     x = F.interpolate(inputs, size=x.shape[-1]*self._rescale)
        #     x = self._conv_0(x)
        #     x = F.leaky_relu(x)

        x = self._conv_1(x)
        x = F.leaky_relu(x)
        # print ('ENC #2:', x.shape)
        
        x = self._conv_2(x)
        x = F.leaky_relu(x)
        # print ('ENC #3:', x.shape)
        
        x = self._conv_3(x)
        x = F.leaky_relu(x)
        # x = self._block(x)
        # print ('ENC #4:', x.shape)

        x = self._conv_4(x)
        x = self._residual_stack(x)
        # print ('ENC #5:', x.shape)
        return x

# %%
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_channels, padding=[1,1,1], layer_sizes=[]):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        # (2020/11) possible kernel size
        # kernel_size=4, stride=2, padding=1, output_padding=0
        # kernel_size=3, stride=2, padding=1, output_padding=1
        # kernel_size=2, stride=2, padding=0, output_padding=0
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=3, stride=2, padding=1, output_padding=padding[0])
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=3, stride=2, padding=1, output_padding=padding[1])
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_channels,
                                                kernel_size=3, stride=2, padding=1, output_padding=padding[2])

        self.MLP = None
        if len(layer_sizes)>1:
            self.MLP = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if (in_size < 0) and (out_size < 0):
                    in_size, out_size = abs(in_size), abs(out_size)
                    assert in_size == out_size

                    m = ResNet_block(torch.nn.Sequential(
                            torch.nn.Linear(in_size, in_size),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(in_size, in_size)),
                            torch.nn.LeakyReLU(),
                            )
                    self.MLP.add_module(name="R{:d}".format(i), module=m)
                else:
                    in_size, out_size = abs(in_size), abs(out_size)
                    self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                    self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU())

        # # (2021/03)
        # self._block = nn.Sequential(
        #     ConvTBlock(num_hiddens, num_hiddens//2, 4, 2, 1),
        #     ConvTBlock(num_hiddens//2, num_hiddens//4, 4, 2, 1),
        #     ConvTBlock(num_hiddens//4, num_channels, 4, 2, 1)
        # )

        # # (2021/03)
        # self._block = nn.Sequential(
        #     ## 128->64, 5->10
        #     ConvTBlock(128, 64, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(64, 64, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(64, 64, kernel_size=4, stride=1, padding=1),
        #     ## 64->32, 10->20
        #     ConvTBlock(64, 32, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(32, 32, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(32, 32, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(32, 32, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(32, 32, kernel_size=3, stride=1, padding=0),
        #     ## 32->1, 20->40
        #     ConvTBlock(32, 16, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(16, 16, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(16, 16, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(16, 16, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(16, 4, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(4, 4, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(4, 4, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(4, 4, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(4, 1, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(1, 1, kernel_size=3, stride=1, padding=0),
        # )

    #0: torch.Size([1, 16, 5, 5])
    #1: torch.Size([1, 128, 5, 5])
    #2: torch.Size([1, 128, 5, 5])
    #3: torch.Size([1, 64, 10, 10])
    #4: torch.Size([1, 64, 20, 20])
    #5: torch.Size([1, 1, 40, 40])
    def forward(self, inputs, da=None):
        x = inputs
        if da is not None:
            nb, nc, nx, ny = x.shape
            _d = torch.zeros_like(x)
            _a = torch.zeros_like(x)
            _d[:,:,:,:] = da[:,0,np.newaxis,np.newaxis,np.newaxis]
            _a[:,:,:,:] = da[:,1,np.newaxis,np.newaxis,np.newaxis]
            x = torch.cat((x, _d[:,-1:,:,:], _a[:,-1:,:,:]), dim=1)
        # print ('DEC #1:', x.shape)

        x = self._conv_1(x)
        x = self._residual_stack(x)
        # print ('DEC #2:', x.shape)
        
        x = self._conv_trans_1(x)
        x = F.leaky_relu(x)
        # print ('DEC #3:', x.shape)
        
        x = self._conv_trans_2(x)
        x = F.leaky_relu(x)
        # print ('DEC #4:', x.shape)
        
        x = self._conv_trans_3(x)
        # x = torch.sigmoid(x)
        # x = self._block(x)
        # print ('DEC #5:', x.shape)

        if self.MLP is not None:
            nb, nc, nx, ny = x.shape
            x = self.MLP(x.view(-1, nc*nx*ny))
            x = x.view(-1, nc, nx, ny)

        return x

# %%
class VQVAE(nn.Module):
    def __init__(self, num_channels, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0, rescale=None, learndiff=False, 
                 input_shape=None, shaconv=False, grid=None, conditional=False, decoder_padding=[1,1,1], 
                 da_conditional=False, decoder_layer_sizes=[]):
        super(VQVAE, self).__init__()
        
        self._grid = grid
        self.width = num_channels
        self.rescale = rescale
        self.conditional = conditional
        print ("Model rescale:", self.rescale)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if grid is not None:
            self.width = 32
            self.fc0 = nn.Linear(3, self.width)
            self.fc1 = nn.Linear(self.width, 128)
            self.fc2 = nn.Linear(128, num_channels)

        self.ncond = 0
        if da_conditional:
            self.ncond = 2

        self._encoder = Encoder(self.width+self.ncond, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        _embedding_dim = embedding_dim
        if self.conditional:
            _embedding_dim = embedding_dim + 256
        self._decoder = Decoder(_embedding_dim+self.ncond,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens, self.width, padding=decoder_padding, 
                                layer_sizes=decoder_layer_sizes)

        """
        Learn diff
        """
        self._learndiff = learndiff
        print ("Model learndiff: %s"%self._learndiff)
        if self._learndiff:
            """
            self._encoder2 = Encoder(num_channels, num_hiddens,
                                    num_residual_layers, 
                                    num_residual_hiddens, rescale=rescale)
            self._pre_vq_conv2 = nn.Conv2d(in_channels=num_hiddens, 
                                        out_channels=embedding_dim,
                                        kernel_size=1, 
                                        stride=1)
            if decay > 0.0:
                self._vq_vae2 = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                                commitment_cost, decay)
            else:
                self._vq_vae2 = VectorQuantizer(num_embeddings, embedding_dim,
                                            commitment_cost)
            self._decoder2 = Decoder(embedding_dim,
                                    num_hiddens, 
                                    num_residual_layers, 
                                    num_residual_hiddens, num_channels)
            """

            self._input_shape = input_shape
            nbatch, nchannel, dim1, dim2 = self._input_shape
            self._dmodel = AE(input_shape=nchannel*dim1*dim2)
            #self._doptimizer = optim.Adam(self._dmodel.parameters(), lr=1e-3)
            self._dcriterion = nn.MSELoss()
        
        self._shaconv = shaconv

        if self.conditional:
            # XGC-VGG19
            modelfile = 'xgc-vgg19-ch3-N1000.torch'
            self.feature_extractor = XGCFeatureExtractor(modelfile)
            # VGG
            # self.feature_extractor = FeatureExtractor()
            self.feature_extractor = self.feature_extractor.to(self._device)
            self.feature_extractor.eval()

    def forward(self, x, da=None):
        nbatch, nchannel, dim1, dim2 = x.shape
        if self._grid is not None:
            x = torch.cat([x, self._grid.repeat(nbatch,1,1,1)], dim=1)
            x = x.permute(0, 2, 3, 1)
            x = self.fc0(x)
            x = x.permute(0, 3, 1, 2)

        ## sha conv
        if self._shaconv:
            x = conv_hash_torch(x)
        
        if self.rescale is not None:
            b, c, nx, ny = x.shape
            x = F.interpolate(x, size=(nx*self.rescale, ny*self.rescale))
            print ('scale-up', x.shape)

        # print ('#0:', x.shape)
        z = self._encoder(x, da)
        # print ('#1:', z.shape)
        z = self._pre_vq_conv(z)
        # print ('#2:', z.shape)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        # print ('#3:', quantized.shape)
        if self.conditional:
            _x = torch.cat((x,x,x), axis=1)
            feature = self.feature_extractor(_x)
            cond = F.avg_pool2d(feature, kernel_size=3, stride=2, padding=1)
            # cond = F.avg_pool1d(feature.view(nbatch,nchannel,1090), kernel_size=11)
            # p1d = (0, 1)
            # cond = F.pad(cond, p1d, "constant", 0)
            # cond = torch.reshape(cond, (nbatch, 4, 5, 5))
            quantized = torch.cat((quantized, cond), dim=1)

        x_recon = self._decoder(quantized, da)
        # print ('#4:', x_recon.shape)

        if self.rescale is not None:
            x_recon = F.interpolate(x_recon, size=(nx, ny))
            print ('scale-down', x_recon.shape)

        if self._grid is not None:
            x = x_recon
            x = x.permute(0, 2, 3, 1)
            x = self.fc1(x)
            x = F.leaky_relu(x)
            x = self.fc2(x)
            x = x.permute(0, 3, 1, 2)
            x_recon = x.contiguous()

        drecon = 0
        dloss = 0

        if self._learndiff:
            # z2 = self._encoder2(x-x_recon)
            # z2 = self._pre_vq_conv2(z2)
            # loss2, quantized2, perplexity2, _ = self._vq_vae2(z2)
            # x_recon2 = self._decoder2(quantized2)
            # return loss+loss2, x_recon+x_recon2, perplexity+perplexity2

            nbatch, nchannel, dim1, dim2 = x.shape
            dx = (x-x_recon).view(-1, nchannel*dim1*dim2)
            outputs = self._dmodel(dx)
            dloss = self._dcriterion(outputs, dx)
            drecon = outputs.view(-1, nchannel, dim1, dim2)

        return loss, x_recon+drecon, perplexity, dloss

