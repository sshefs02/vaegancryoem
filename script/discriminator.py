"""
This file defines the Discriminator System of the network architecture of the proposed VAE-GAN model in the paper:
Estimation of Orientation and Camera Parameters from Cryo-Electron Microscopy Images with Variational Autoencoders and Generative Adversarial Networks
"""
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Discriminator System for the Generative Adversarial Network.

    Attributes:

    """

    def __init__(self, input_image_width=40, input_channels=1):
        """
        Discriminator System for the Generative Adversarial Network.

        Attributes:
            input_image_width (int): Width (n) of the input (n*n) image.
            input_channels (int): Number of channels in the input image.
        """

        super(Discriminator, self).__init__()

        output_channels = 64
        num_conv_layers = 4
        layers, final_output_feature_dims = self.get_layers(
            input_image_width,
            input_channels,
            output_channels,
            num_conv_layers
        )
        self.conv = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(in_features=final_output_feature_dims, out_features=512, bias=False),
            nn.BatchNorm1d(num_features=512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1),
        )

    def forward(self, x):
        """
        Forward Pass of the Discriminator System for the Generative Adversarial Network.

        Attributes:
            x (tensor): Training Images

        Returns:
            Sigmoid/Probability of the input image x being real.
        """

        output = self.conv(x)
        output = output.view(len(output), -1)
        output = self.fc(output)
        sigmoid_output = F.sigmoid(output)
        return sigmoid_output

    @staticmethod
    def get_layers(input_image_width=40, input_channels=1, output_channels=64, num_conv_layers=4):
        """
        This function returns the Conv layers of the Discriminator of the Generative Adversarial Network.

        Parameters:
            input_image_width (int): Width (n) of the input (n*n) image.
            input_channels (int): Number of channels in the input image.
            output_channels (int): Output channels for the first conv layer.
            num_conv_layers (int): Number of conv layers in the system.

        Returns:
            The sequential set of convolutional layers and the final dimensions of the
            output to feed to the fully connected layer.
        """

        layers = []
        kernel_size = 5
        padding = 2
        stride = 2
        momentum = 0.9

        for i in range(num_conv_layers):
            conv = nn.Conv2d(in_channels=input_channels,
                             out_channels=output_channels,
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=stride,
                             bias=False)
            bn = nn.BatchNorm2d(num_features=output_channels,
                                momentum=momentum)
            input_channels = output_channels
            output_channels *= 2
            input_image_width = int((input_image_width - kernel_size + 2 * padding) / stride + 1)
            layers.append(conv)
            layers.append(bn)

        final_output_feature_dims = int(input_image_width * input_image_width * int(output_channels/2))  # Final number of features.
        return layers, final_output_feature_dims
