"""
This file defines the Decoder System of the network architecture of the proposed VAE-GAN model in the paper:
Estimation of Orientation and Camera Parameters from Cryo-Electron Microscopy Images with Variational Autoencoders and Generative Adversarial Networks
"""

import torch.nn as nn
import math


class Decoder(nn.Module):
    """
    Decoder System for the Variational Auto-Encoder.

    Attributes:
        latent_space_dimensions (int): The dimensions of the z variable of the latent space
        num_conv_layers (int): Desired number of convolution layers in the encoder
        input_channels (int): Desired number of input channels in the conv layers for decoder
        output_channels (int): Desired number of output channels in the final reconstructed output of the decoder
        input_image_width (int): Desired number of image width (n) in the output of the decoder for (n*n) image
    """

    def __init__(self, latent_space_dimensions=4, num_conv_layers=5, input_channels=64,
                 output_channels=1, input_image_width=40):
        """
        Constructor for the Decoder System of Variational Auto-Encoder.

        Parameters:
            latent_space_dimensions (int): The dimensions of the z variable of the latent space
            num_conv_layers (int): Desired number of convolution layers in the encoder
            input_channels (int): Desired number of input channels in the conv layers for decoder
            output_channels (int): Desired number of output channels in the final reconstructed output of the decoder
            input_image_width (int): Desired number of image width (n) in the output of the decoder for (n*n) image
        """
        super(Decoder, self).__init__()
        self.input_channels = input_channels

        # Define Fully Connected Layer
        self.image_width = 2  # Modify image_size as per requirement.
        output_feature_dims = self.image_width * self.image_width * input_channels
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=latent_space_dimensions,
                out_features=output_feature_dims,
                bias=False
            ),
            nn.BatchNorm1d(
                num_features=output_feature_dims,
                momentum=0.9
            ),
            nn.ReLU(True)
        )

        # Define Convolution Layers
        layers = self.get_conv_layers(
            num_conv_layers,
            output_channels,
            input_image_width
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, z):
        """
        Forward Pass for the Decoder System of Variational Auto-Encoder.

        Parameters:
            x (tensor): Training Images

        Returns:
            Generated/Reconstructed output from the forward pass of the decoder
        """

        output = self.fc(z)
        num_samples = len(output)
        output = output.view(num_samples, self.input_channels, self.image_width, self.image_width)
        output = self.conv(output)
        return output

    def get_conv_layers(self, num_conv_layers=5, output_channels=1, required_width=40):
        """
        This function returns the Conv layers of the Decoder of the Variational Auto-Encoder.

        Parameters:
            num_conv_layers (int): Desired number of convolution layers in the encoder
            output_channels (int): Width/Height (n) of the input image of shape (n*n)
            required_width (int): Width/Height (n) of the input image of shape (n*n)

        Returns:
            The sequential set of convolutional layers and the final dimensions of the
            output to feed to the fully connected layer.
        """

        layers = []
        channels = self.input_channels
        kernel_size = 5
        padding = 2
        stride = 2
        momentum = 0.9
        output_padding = 1
        image_width = self.image_width

        for i in range(num_conv_layers):
            conv = nn.ConvTranspose2d(in_channels=channels,
                                      out_channels=int(channels/2),  # Half the channels.
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      stride=stride,
                                      output_padding=output_padding,
                                      bias=False)
            bn = nn.BatchNorm2d(int(channels/2),
                                momentum=momentum)
            r = nn.ReLU(True)

            image_width = int((image_width - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1)
            channels = int(channels/2)
            layers.append(conv)
            layers.append(bn)
            layers.append(r)

        # MAKE SURE: Be careful to choose positive padding.
        # This will depend on the number of conv layers
        # and the image width at the end of the loop above.
        padding = math.floor((stride * required_width - 1 - image_width + kernel_size) / 2)

        # Final conv layers to get the reconstructed image of original shape.
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=output_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.Tanh()
        ))

        return layers

