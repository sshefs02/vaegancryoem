"""
This file defines the Encoder System of the network architecture of the proposed VAE-GAN model in the paper:
Estimation of Orientation and Camera Parameters from Cryo-Electron Microscopy Images with Variational Autoencoders and Generative Adversarial Networks
"""

import torch.nn as nn


class Encoder(nn.Module):
    __doc__ = r"""
    Encoder System for the Variational Auto-Encoder.

    Attributes:
        input_channels (int): The number of channels in the input Images
        latent_space_dimensions (int): The dimensions of the z variable of the latent space
        num_conv_layers (int): Desired number of convolution layers in the encoder
    """

    def __init__(self, input_channels, input_image_width, latent_space_dimensions=4, num_conv_layers=5):
        """
        Constructor for the Encoder System of Variational Auto-Encoder.

        Parameters:
            input_channels (int): The number of channels in the input Images
            latent_space_dimensions (int): The dimensions of the z variable of the latent space
            num_conv_layers (int): Desired number of convolution layers in the encoder
        """

        super(Encoder, self).__init__()

        # Set Parameters
        self.input_channels = input_channels
        self.latent_space_dimensions = latent_space_dimensions
        self.num_conv_layers = num_conv_layers

        # Get & Set Convolution Layers
        layers, final_output_feature_dims = self.get_layers(
            input_channels,
            num_conv_layers,
            input_image_width
        )
        self.conv = nn.Sequential(*layers)

        input_feature_dims = final_output_feature_dims
        output_feature_dims = 1024

        # Define Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=input_feature_dims,
                out_features=output_feature_dims,
                bias=False
            ),
            nn.BatchNorm1d(
                num_features=output_feature_dims,
                momentum=0.9
            ),
            nn.ReLU(True)
        )

        # Two Linear Layers to get the (mu) vector and the diagonal of the (log_variance)
        self.l_mu = nn.Linear(
            in_features=output_feature_dims,
            out_features=self.latent_space_dimensions
        )
        self.l_var = nn.Linear(
            in_features=output_feature_dims,
            out_features=self.latent_space_dimensions
        )

    def forward(self, x):
        """
        Forward Pass for the Encoder System of Variational Auto-Encoder.

        Parameters:
            x (tensor): Training Images

        Returns:
            Generated mu and sigma vectors from the forward pass
        """

        output = self.conv(x)
        num_samples = len(output)
        output = output.view(num_samples, -1)  # Reshape before feeding to the fully connected layer.
        output = self.fc(output)
        mus = self.l_mu(output)
        log_variances = self.l_var(output)
        return mus, log_variances

    @staticmethod
    def get_layers(input_channels, num_conv_layers, input_image_width):
        """
        This function returns the layers of the Encoder of the Variational Auto-Encoder.

        Parameters:
            input_channels (int): The number of channels in the input Images
            num_conv_layers (int): Desired number of convolution layers in the encoder
            input_image_width (int): Width/Height (n) of the input image of shape (n*n)

        Returns:
            The sequential set of convolutional layers and the final dimensions of the
            output to feed to the fully connected layer.
        """

        output_channels = 16  # For the first convolution layer
        image_width = input_image_width
        kernel_size = 5
        padding = 2
        stride = 2
        momentum = 0.9
        layers = []

        # Define the first convolution layer
        conv = nn.Conv2d(in_channels=input_channels,
                         out_channels=output_channels,
                         kernel_size=kernel_size,
                         padding=padding,
                         stride=stride,
                         bias=False)
        bn = nn.BatchNorm2d(num_features=output_channels,
                            momentum=momentum)

        image_width = int((image_width - kernel_size + 2 * padding) / stride + 1)
        layers.append(conv)
        layers.append(bn)

        # Add the next (num_conv_layers-1) convolution layers
        for i in range(1, num_conv_layers):
            conv = nn.Conv2d(in_channels=output_channels,
                             out_channels=output_channels * 2,  # Double the output channel with every conv layer
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=stride,
                             bias=False)
            bn = nn.BatchNorm2d(num_features=output_channels * 2,
                                momentum=momentum)
            output_channels *= 2
            image_width = int((image_width - kernel_size + 2 * padding) / stride + 1)
            layers.append(conv)
            layers.append(bn)

        final_output_feature_dims = int(image_width * image_width * output_channels)  # Final number of features.
        return layers, final_output_feature_dims
