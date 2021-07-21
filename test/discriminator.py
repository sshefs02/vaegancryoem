"""
Unit Testing for Discriminator System of a VAE-GAN.
"""
import unittest
import torch
from vaegancryoem.script.discriminator import Discriminator


class DiscriminatorTest(unittest.TestCase):

    def test_discriminator_creation(self):
        """
        Test Discriminator Creation to run without error.

        """
        input_image_width = 40
        input_channels = 1

        self.discriminator = Discriminator(
            input_image_width=input_image_width,
            input_channels=input_channels
        )

    def test_discriminator_forward_pass(self):
        """
        Test Encoder Forward Pass to run without error.

        """
        input_image_width = 40
        input_channels = 1

        self.discriminator = Discriminator(
            input_image_width=input_image_width,
            input_channels=input_channels
        )
        x = torch.rand((100, 1, 40, 40))
        self.discriminator(x)


if __name__ == '__main__':
    unittest.main()
