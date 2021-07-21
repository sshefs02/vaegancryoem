"""
This file defines the VAE-GAN System of the network architecture of the proposed VAE-GAN model in the paper:
Estimation of Orientation and Camera Parameters from Cryo-Electron Microscopy Images with Variational Autoencoders and Generative Adversarial Networks
"""
import torch.nn as nn
from torch.autograd import Variable

from vaegancryoem.script.encoder import Encoder
from vaegancryoem.script.decoder import Decoder
from vaegancryoem.script.discriminator import Discriminator


class VAEGAN(nn.Module):
    __doc__ = r"""
    Entire VAE-GAN System for the Variational Auto-Encoder cum Generative Adversarial Network.

    Attributes:
        input_channels (int): The number of channels in the input Images
        input_image_width (int): Width (n) of the input (n*n) image.
        latent_space_dimensions (int): The dimensions of the z variable of the latent space
    """

    def __init__(self, input_channels, input_image_width, latent_space_dimensions):
        """
        Entire VAE-GAN System for the Variational Auto-Encoder cum Generative Adversarial Network.

        Attributes:
            input_channels (int): The number of channels in the input Images
            input_image_width (int): Width (n) of the input (n*n) image.
            latent_space_dimensions (int): The dimensions of the z variable of the latent space
        """
        super(VAEGAN, self).__init__()

        # Define the Encoder.
        num_encoder_conv_layers = 5
        self.encoder = Encoder(
            input_channels=input_channels,
            input_image_width=input_image_width,
            latent_space_dimensions=latent_space_dimensions,
            num_conv_layers=num_encoder_conv_layers
        )

        # Define the Decoder.
        num_decoder_conv_layers = 5
        decoder_first_conv_layer_channels = 64
        self.decoder = Decoder(
            latent_space_dimensions=latent_space_dimensions,
            num_conv_layers=num_decoder_conv_layers,
            input_channels=decoder_first_conv_layer_channels,
            output_channels=input_channels,
            input_image_width=input_image_width
        )

        # Define the Discriminator
        self.Discriminator = Discriminator(
            input_image_width=input_image_width,
            input_channels=input_channels
        )

    def forward(self, x):
        """
        Forward Pass for the Entire VAE-GAN System of Variational Auto-Encoder cum Generative Adversarial Network.

        Parameters:
            x (tensor): Training Images

        Returns:
            reconstructed_x (tensor): Reconstructed Image from the Decoder
            real_discriminator_preds (tensor): Predictions of the Input Images
            fake_discriminator_preds (tensor): Predictions of the Generated images
            mus (tensor): Mean of the Assumed Gaussian Distribution of the latent space
            log_variances (tensor): Sigma of the Assumed Gaussian Distribution of the latent space
        """

        # VAE: Encoder - Decoder Pass
        mus, log_variances = self.encoder(x)
        z = self.reparameterize(mus, log_variances)
        reconstructed_x = self.decoder(z)

        # GAN: Discriminator Pass
        real_discriminator_preds = self.Discriminator(x)
        fake_discriminator_preds = self.Discriminator(reconstructed_x)

        return reconstructed_x, real_discriminator_preds, fake_discriminator_preds, mus, log_variances

    @staticmethod
    def reparameterize(mus, log_variances):
        """
        Reparameterization Trick to compute gradients easily for the latent space variable z.

        Parameters:
            mus (tensor): Mean of the Assumed Gaussian Distribution of the latent space
            log_variances (tensor): Sigma of the Assumed Gaussian Distribution of the latent space

        Returns:
            Sampled z from the mu and sigma of the Assumed Gaussian Distribution of the latent space
        """

        logvar = log_variances.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mus)
