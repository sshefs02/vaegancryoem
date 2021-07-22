"""
Unit Testing for Encoder System of a VAE-GAN.
"""

import unittest
import torch
import torch.nn as nn
from vaegancryoem.script.models.encoder import Encoder


class EncoderTest(unittest.TestCase):

    def test_encoder_creation(self):
        """
        Test Encoder Creation to run without error.

        """
        input_channels = 1
        image_width = 40

        self.encoder = Encoder(
            input_channels,
            image_width,
            latent_space_dimensions=4,
            num_conv_layers=5
        )

    def test_encoder_forward_pass(self):
        """
        Test Encoder Forward Pass to run without error.

        """

        input_channels = 1
        image_width = 40

        self.encoder = Encoder(
            input_channels,
            image_width,
            latent_space_dimensions=4,
            num_conv_layers=5
        )
        x = torch.rand((100, 1, 40, 40))
        mus, variances = self.encoder(x)

    @staticmethod
    def test_encoder_setup():
        x = torch.rand((100, 1, 40, 40))

        num_conv_layers = 5
        input_channels = 1
        output_channels = 16
        image_width = 40
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
        x = conv(x)
        print("After first conv layer: ", x.shape)

        bn = nn.BatchNorm2d(num_features=output_channels,
                            momentum=momentum)

        x = bn(x)
        print("After first batch-norm layer: ", x.shape)

        image_width = int((image_width - kernel_size + 2 * padding) / stride + 1)
        print(f"Current image width={image_width}")

        print()

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
            x = conv(x)
            print(f"After {i+1}th conv layer: ", x.shape)

            x = bn(x)
            print(x.shape)
            print(f"After {i + 1}th batch-norm layer: ", x.shape)

            output_channels *= 2
            image_width = int((image_width - kernel_size + 2 * padding) / stride + 1)
            print(f"Current image width={image_width}")

            layers.append(conv)
            layers.append(bn)

            print()

        print(f"Current shape of output={x.shape}")
        print()

        print(f"Current output channels={output_channels}")

        final_output_feature_dims = int(image_width * image_width * output_channels)  # Final number of features.

        print(f"final_output_feature_dims={final_output_feature_dims}")
        print(f"image_width={image_width}")
        print(f"output_channels={output_channels}")

        return layers, final_output_feature_dims


if __name__ == '__main__':
    unittest.main()
