import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import global_v as glv

import lpips

class AE(nn.Module):
    def __init__(self) -> None:
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

        modules = []
        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.latent_layer = nn.Linear(hidden_dims[-1]*lowered_size**2, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] *lowered_size**2)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.ConvTranspose2d(hidden_dims[-1], out_channels = out_channels, # deconvolution
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    
    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        latent = self.latent_layer(result)

        return latent

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.lowered_size, self.lowered_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    
    
    def forward(self, input):
        latent = self.encode(input)
        decoded = self.decode(latent)
        return  decoded, latent

    def loss_function_mse(self, input_img, recons_img):

        recons_loss = F.mse_loss(recons_img, input_img)
        
        loss = recons_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': torch.zeros_like(recons_loss)}
    
    def loss_function_l1(self, input_img, recons_img):

        #recons_loss = F.l1_loss(recons_img, input_img)
        
        #loss = recons_loss
        
        loss = torch.amax(torch.abs(input_img-recons_img),dim=1).mean()

        return {'loss': loss, 'Reconstruction_Loss':loss, 'Distance_Loss': torch.zeros_like(loss)}
    
    def loss_function_lpips(self, input_img, recons_img):
        """
        Computes the LPIPS loss between two batches of images.
    
        Args:
            network (nn.Module): The LPIPS network.
            img1 (tensor): Batch of images.
            img2 (tensor): Batch of images.
        
        Returns:
            tensor: LPIPS loss.
        """
        # Ensure images are in the range [0, 1]
        input_img = torch.clamp(input_img, 0, 1)
        recons_img = torch.clamp(recons_img, 0, 1)

        # Normalize images to [-1, 1] range
        input_img = 2 * input_img - 1
        recons_img = 2 * recons_img - 1
        
        loss_fn_alex = lpips.LPIPS(net='alex').to('cuda') # best forward scores
        
        loss = loss_fn_alex(input_img, recons_img).mean()
                
        return {'loss': loss, 'Reconstruction_Loss':loss, 'Distance_Loss': torch.zeros_like(loss)}

    
    def loss_function_ssim(self, img1, img2, window_size=11, sigma=4, k1=0.001, k2=0.003):
        # Ensure inputs are 4D tensors (batch_size, channels, height, width)
        # Compute constants
        C1 = (k1 * 255) ** 2
        C2 = (k2 * 255) ** 2
    
        # Gaussian window
        x = torch.arange(window_size, dtype=torch.float32)
        gauss = torch.exp(-(x - window_size // 2)**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()

        # Expand dimensions of gauss to match img1 and img2
        gauss = gauss.unsqueeze(0).unsqueeze(0)
        gauss = gauss.expand(img1.shape[1], -1, -1).unsqueeze(0).to('cuda')

        # Compute means
        mu1 = F.conv2d(img1, gauss, padding=window_size//2)
        mu2 = F.conv2d(img2, gauss, padding=window_size//2)
    
        # Compute variances and covariances
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 ** 2, gauss, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, gauss, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, gauss, padding=window_size//2) - mu1_mu2
    
        # Compute SSIM
        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / denom
        
        ssim_val = torch.mean(ssim_map)

        return {'loss': ssim_val, 'Reconstruction_Loss':ssim_val, 'Distance_Loss': torch.zeros_like(ssim_val)}

    
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        raise NotImplementedError()

class AELarge(AE):
    def __init__(self) -> None:
        super(AE, self).__init__()
        
        in_channels = glv.network_config['in_channels']
        if glv.network_config['out_channels'] is not None:
            out_channels = glv.network_config['out_channels']
        else:
            out_channels = in_channels
        input_size = glv.network_config['input_size']
        lowered_size = int(input_size/32)
        self.lowered_size = lowered_size
        
        latent_dim = glv.network_config['latent_dim']

        modules = []
        hidden_dims = [32, 64, 128, 256, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.latent_layer = nn.Linear(hidden_dims[-1]*latent_dim**2, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * latent_dim**2)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.ConvTranspose2d(hidden_dims[-1], out_channels= out_channels, # deconvolution
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())