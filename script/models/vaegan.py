"""
This file defines the VAE-GAN System of the network architecture of the proposed VAE-GAN model in the paper:
Estimation of Orientation and Camera Parameters from Cryo-Electron Microscopy Images with Variational Autoencoders and Generative Adversarial Networks
"""
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from vaegancryoem.script.models.encoder import Encoder
from vaegancryoem.script.models.decoder import Decoder
from vaegancryoem.script.models.discriminator import Discriminator


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
        self.discriminator = Discriminator(
            input_image_width=input_image_width,
            input_channels=input_channels
        )

        self.init_parameters()

    def init_parameters(self):
        """
        Parameter Initialisation for the Entire VAE-GAN System of Variational Auto-Encoder cum Generative Adversarial Network.
        """

        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    scale = 1.0/np.sqrt(np.prod(m.weight.shape[1:]))
                    scale /= np.sqrt(3)
                    nn.init.uniform(m.weight,-scale,scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

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
        reconstructed_x = self.decoder(z.detach())

        # GAN: Discriminator Pass
        real_discriminator_preds = self.discriminator(x)
        fake_discriminator_preds = self.discriminator(reconstructed_x.detach())

        return reconstructed_x, real_discriminator_preds, fake_discriminator_preds, mus, log_variances, z

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

    @staticmethod
    def get_loss(x, reconstructed_x, mus, log_variances, z, real_discriminator_preds, fake_discriminator_preds):

        # Binary Cross Entropy between input image and reconstructed image.
        reconstruction_loss = torch.mean(
            0.5 * (x.view(len(x), -1) - reconstructed_x.view(len(reconstructed_x), -1)) ** 2
        )  # Params: Encoder+Decoder, Propagate: Encoder+Decoder

        # KL Divergence.
        regularisation_loss = torch.mean(
            -0.5 * torch.sum(-log_variances.exp() - torch.pow(mus, 2) + log_variances + 1, 1), dim=0
        )  # Params: Encoder, Propagate: Encoder

        # Cone Loss
        z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
        cone_loss = torch.sum(z1 ** 2 + z2 ** 2 - z3 ** 2)**2

        gan_loss_original = torch.sum(
            -torch.log(real_discriminator_preds + 1e-3)
        )

        gan_loss_predicted = torch.sum(
            -torch.log(1 - fake_discriminator_preds + 1e-3)
        )

        gan_loss = gan_loss_original + gan_loss_predicted  # Params: Encoder+Decoder+Discriminator, Propagate: Decoder+Discriminator

        return reconstruction_loss, regularisation_loss, cone_loss, gan_loss_original, gan_loss_predicted, gan_loss