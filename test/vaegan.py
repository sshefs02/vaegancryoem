"""
Unit Testing for Entire System of a VAE-GAN.
"""
import unittest
import torch
from vaegancryoem.script.vaegan import VAEGAN


class VAEGANTest(unittest.TestCase):

    def test_vaegan_creation(self):
        """
        Test VAE-GAN Creation to run without error.

        """
        input_channels = 1
        input_image_width = 40
        latent_space_dimensions = 3

        self.vaegan = VAEGAN(
            input_channels,
            input_image_width,
            latent_space_dimensions=latent_space_dimensions
        )

    def test_vaegan_forward_pass(self):
        """
        Test VAE-GAN Forward Pass to run without error.

        """

        input_channels = 1
        input_image_width = 40
        latent_space_dimensions = 3

        self.vaegan = VAEGAN(
            input_channels,
            input_image_width,
            latent_space_dimensions=latent_space_dimensions
        )

        x = torch.rand((100, 1, 40, 40))
        self.vaegan(x)


if __name__ == '__main__':
    unittest.main()
