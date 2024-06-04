
import torch
import torch.nn as nn
from .snn_layers import *
from .fsvae_prior import *
from .fsvae_posterior import *
import torch.nn.functional as F

import global_v as glv


class FSVAE(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = glv.network_config['in_channels']
        if glv.network_config['out_channels'] is not None:
            out_channels = glv.network_config['out_channels']
        else:
            out_channels = in_channels
        input_size = glv.network_config['input_size']
        lowered_size = int(input_size/16)
        self.lowered_size = lowered_size
            
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike(),
                        is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*lowered_size**2,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)
        
        self.posterior = PosteriorBernoulliSTBP(self.k)

        # Build Decoder
        modules = []
        
        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * lowered_size**2, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * lowered_size**2),
                                        spike=LIFSpike())
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike()),
                            tdConvTranspose(hidden_dims[-1], 
                                            out_channels=glv.network_config['out_channels'],
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()

    def forward(self, x, scheduled=False):
        sampled_z, q_z, p_z = self.encode(x, scheduled)
        x_recon = self.decode(sampled_z)
        return x_recon, q_z, p_z, sampled_z
    
    def simulate(self, x, scheduled=False):
        sampled_z, q_z, p_z = self.encode(x, scheduled)
        result = self.decoder_input(sampled_z) # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], self.lowered_size, self.lowered_size, self.n_steps) # (N,C,H,W,T)
        result = self.decoder(result)# (N,C,H,W,T)
        result = self.final_layer(result)# (N,C,H,W,T)
        out = torch.tanh(result) 
        return out
    
    def encode(self, x, scheduled=False):
        x = self.encoder(x) # (N,C,H,W,T)
        x = torch.flatten(x, start_dim=1, end_dim=3) # (N,C*H*W,T)
        latent_x = self.before_latent_layer(x) # (N,latent_dim,T)
        sampled_z, q_z = self.posterior(latent_x) # sampled_z:(B,C,1,1,T), q_z:(B,C,k,T)

        p_z = self.prior(sampled_z, scheduled, self.p)
        return sampled_z, q_z, p_z

    def decode(self, z):
        result = self.decoder_input(z) # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], self.lowered_size, self.lowered_size, self.n_steps) # (N,C,H,W,T)
        result = self.decoder(result)# (N,C,H,W,T)
        result = self.final_layer(result)# (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))        
        return out

    def sample(self, batch_size=64):
        sampled_z = self.prior.sample(batch_size)
        sampled_x = self.decode(sampled_z)
        return sampled_x, sampled_z
        
    def loss_function_mmd(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        #recons_loss = F.mse_loss(recons_img, input_img)
        #recons_loss = torch.amax(torch.abs(input_img-recons_img),dim=1).mean()
        recons_loss = F.mse_loss(recons_img, input_img)
        q_z_ber = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        p_z_ber = torch.mean(p_z, dim=2) # (N, latent_dim, T)

        #kld_loss = torch.mean((q_z_ber - p_z_ber)**2)
        mmd_loss = torch.mean((self.psp(q_z_ber)-self.psp(p_z_ber))**2)
        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': mmd_loss}

    def loss_function_kld(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        prob_q = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        prob_p = torch.mean(p_z, dim=2) # (N, latent_dim, T)
        
        kld_loss = prob_q * torch.log((prob_q+1e-2)/(prob_p+1e-2)) + (1-prob_q)*torch.log((1-prob_q+1e-2)/(1-prob_p+1e-2))
        kld_loss = torch.mean(torch.sum(kld_loss, dim=(1,2)))

        loss = recons_loss + 1e-4 * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': kld_loss}
    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4,4)

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p-init_p) * epoch / max_epoch + init_p
        
class FSVAELarge(FSVAE):
    def __init__(self):
        super(FSVAE, self).__init__()
        in_channels = glv.network_config['in_channels']
        if glv.network_config['out_channels'] is not None:
            out_channels = glv.network_config['out_channels']
        else:
            out_channels = in_channels
        input_size = glv.network_config['input_size']
        lowered_size = int(input_size/32)
        self.lowered_size = lowered_size
        
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*lowered_size**2,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)
        
        self.posterior = PosteriorBernoulliSTBP(self.k)
        
        # Build Decoder
        modules = []
        
        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * lowered_size**2, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * lowered_size**2),
                                        spike=LIFSpike())
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike()),
                            tdConvTranspose(hidden_dims[-1], 
                                            out_channels=glv.network_config['out_channels'],
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()
        
class FSVAESmall(FSVAE):
    def __init__(self):
        super(FSVAE, self).__init__()
        in_channels = glv.network_config['in_channels']
        if glv.network_config['out_channels'] is not None:
            out_channels = glv.network_config['out_channels']
        else:
            out_channels = in_channels
        input_size = glv.network_config['input_size']
        lowered_size = int(input_size/16)
        self.lowered_size = lowered_size
        
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*lowered_size**2,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)
        
        self.posterior = PosteriorBernoulliSTBP(self.k)
        
        # Build Decoder
        modules = []
        
        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * lowered_size**2, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * lowered_size**2),
                                        spike=LIFSpike())
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    tdUpsample(size=lowered_size*2**(i+1),
                               mode='trilinear',
                               align_corners=True)
            )
            modules.append(
                    tdConv(in_channels=hidden_dims[i],
                            out_channels=hidden_dims[i + 1],
                            kernel_size=3,
                            stride = 1,
                            padding=1,
                            bias=True,
                            bn=tdBatchNorm(hidden_dims[i+1]),
                            spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            tdConv(in_channels=hidden_dims[-1],
                                        out_channels=hidden_dims[-1],
                                        kernel_size=3,
                                        stride = 1,
                                        padding=1,
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1]),
                                        spike=LIFSpike()),
                            tdUpsample(size=input_size,
                                       mode='trilinear',
                                       align_corners=True),
                            tdConv(in_channels=hidden_dims[-1],
                                        out_channels=out_channels,
                                        kernel_size=3,
                                        stride = 1,
                                        padding=1,
                                        bias=True,
                                        bn=None,
                                        spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()
class FSAESmall(FSVAE):
    def __init__(self):
        super(FSVAE, self).__init__()
        in_channels = glv.network_config['in_channels']
        if glv.network_config['out_channels'] is not None:
            out_channels = glv.network_config['out_channels']
        else:
            out_channels = in_channels
        input_size = glv.network_config['input_size']
        
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256, 512, 1024]
        
        lowered_size = int(input_size/(2**len(hidden_dims)))
        self.lowered_size = lowered_size
        
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        modules.append(
            tdConv(in_channels,
                    out_channels=hidden_dims[0],
                    kernel_size=3, 
                    stride=1, 
                    padding=1,
                    bias=True,
                    bn=tdBatchNorm(hidden_dims[0]),
                    spike=LIFSpike())
        )
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels=in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike())
            )
            modules.append(
                tdConv(in_channels=h_dim,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=1, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*lowered_size**2,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())
        # Build Decoder
        modules = []
        
        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * lowered_size**2, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * lowered_size**2),
                                        spike=LIFSpike())
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    tdUpsample(size=lowered_size*2**(i+1),
                               mode='trilinear',
                               align_corners=True)
            )
            modules.append(
                    tdConv(in_channels=hidden_dims[i],
                            out_channels=hidden_dims[i + 1],
                            kernel_size=3,
                            stride = 1,
                            padding=1,
                            bias=True,
                            bn=tdBatchNorm(hidden_dims[i+1]),
                            spike=LIFSpike())
            )
            modules.append(
                    tdConv(in_channels=hidden_dims[i+1],
                            out_channels=hidden_dims[i+1],
                            kernel_size=3,
                            stride = 1,
                            padding=1,
                            bias=True,
                            bn=tdBatchNorm(hidden_dims[i+1]),
                            spike=LIFSpike())
            )
                
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            tdConv(in_channels=hidden_dims[-1],
                                        out_channels=hidden_dims[-1],
                                        kernel_size=3,
                                        stride = 1,
                                        padding=1,
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1]),
                                        spike=LIFSpike()),
                            tdUpsample(size=input_size,
                                       mode='trilinear',
                                       align_corners=True),
                            tdConv(in_channels=hidden_dims[-1],
                                        out_channels=out_channels,
                                        kernel_size=3,
                                        stride = 1,
                                        padding=1,
                                        bias=True,
                                        bn=None,
                                        spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()
        
    def forward(self, x, scheduled=False):
        latent_x = self.encode(x, scheduled)
        x_recon = self.decode(latent_x)
        return x_recon, latent_x
    
    def encode(self, x, scheduled=False):
        x = self.encoder(x) # (N,C,H,W,T)
        x = torch.flatten(x, start_dim=1, end_dim=3) # (N,C*H*W,T)
        latent_x = self.before_latent_layer(x) # (N,latent_dim,T)
        return latent_x
    
    def decode(self, z):
        result = self.decoder_input(z) # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], self.lowered_size, self.lowered_size, self.n_steps) # (N,C,H,W,T)
        result = self.decoder(result)# (N,C,H,W,T)
        result = self.final_layer(result)# (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result)).squeeze()        
        return out
        
    def loss_function_mse(self, input_img, recons_img):

        recons_loss = F.mse_loss(recons_img, input_img)
        
        loss = recons_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': torch.zeros_like(recons_loss)}
    
    def loss_function_l1(self, input_img, recons_img):

        recons_loss = F.l1_loss(recons_img, input_img)
        
        loss = recons_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': torch.zeros_like(recons_loss)}
    
