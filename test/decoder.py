"""
Unit Testing for Decoder System of a VAE-GAN.
"""
import unittest
import torch
from vaegancryoem.script.models.decoder import Decoder


class DecoderTest(unittest.TestCase):

    def test_decoder_creation(self):
        """
        Test Decoder Creation to run without error.

        """

        latent_space_dimensions = 4
        num_conv_layers = 5
        input_channels = 64
        output_channels = 1
        input_image_width = 40

        self.decoder = Decoder(
            latent_space_dimensions=latent_space_dimensions,
            num_conv_layers=num_conv_layers,
            input_channels=input_channels,
            output_channels=output_channels,
            input_image_width=input_image_width
        )

    def test_decoder_forward_pass(self):
        """
        Test Encoder Forward Pass to run without error.

        """

        latent_space_dimensions = 4
        num_conv_layers = 5
        input_channels = 64
        output_channels = 1
        input_image_width = 40
        batch_size = 100

        self.decoder = Decoder(
            latent_space_dimensions=latent_space_dimensions,
            num_conv_layers=num_conv_layers,
            input_channels=input_channels,
            output_channels=output_channels,
            input_image_width=input_image_width
        )
        
        z = torch.rand((batch_size, latent_space_dimensions))
        output = self.decoder(z)
        self.assertEqual(output.shape[0], 100)
        self.assertEqual(output.shape[1], output_channels)
        self.assertEqual(output.shape[2], input_image_width)
        self.assertEqual(output.shape[3], input_image_width)


if __name__ == '__main__':
    unittest.main()


# Test
# import torch
# x = torch.rand((100, 16, 8, 8))
# channel_in = 16
# channel_out = 8
# conv = nn.ConvTranspose2d(channel_in,
#                           channel_out,
#                           kernel_size=5,
#                           padding=2,
#                           stride=2,
#                           output_padding=1,
#                           bias=False)
# output = conv(x)
# print(output.shape)
#
# channel_in = 8
# conv = nn.ConvTranspose2d(channel_in,
#                           channel_out,
#                           kernel_size=5,
#                           padding=2,
#                           stride=2,
#                           output_padding=1,
#                           bias=False)
#
# output = conv(output)
# print(output.shape)
#

# import torch
# import math
#
# x = torch.rand((100, 64, 2, 2))
#
# input_channels = x.shape[1]
# image_width = 2
# num_conv_layers = 5
# layers = []
# channels = input_channels
# kernel_size = 5
# padding = 2
# stride = 2
# momentum = 0.9
# output_padding = 1
# output_channels = 1
#
# for i in range(num_conv_layers):
#     conv = nn.ConvTranspose2d(in_channels=channels,
#                               out_channels=int(channels / 2),  # Half the channels.
#                               kernel_size=kernel_size,
#                               padding=padding,
#                               stride=stride,
#                               output_padding=output_padding,
#                               bias=False)
#
#     bn = nn.BatchNorm2d(int(channels / 2),
#                         momentum=momentum)
#
#     x = conv(x)
#     x = bn(x)
#     print(f"x.shape = {x.shape}")
#
#     image_width = int((image_width - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1)
#     print(f"image width = {image_width}")
#
#     channels = int(channels / 2)
#     layers.append(conv)
#     layers.append(bn)
#
# # Final conv layers to get the reconstructed image.
# required_width = 40
# padding = math.floor((stride * required_width - 1 - image_width + kernel_size)/2)
# print(f"Padding = {padding}")
# conv = nn.Conv2d(in_channels=channels,
#                  out_channels=output_channels,
#                  kernel_size=kernel_size,
#                  stride=stride,
#                  padding=padding)
# tanh = nn.Tanh()
#
# x = conv(x)
# x = tanh(x)
# print(f"x.shape = {x.shape}")