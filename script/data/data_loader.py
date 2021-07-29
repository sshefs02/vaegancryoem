"""
This file defines the Data Loader to run the VAE-GAN proposed in the paper:
Estimation of Orientation and Camera Parameters from Cryo-Electron Microscopy Images with Variational Autoencoders and Generative Adversarial Networks
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class CryoEMDataset(Dataset):
    """
    Data Loader for the Cryo-EM Dataset.

    Attributes:
        file_path (str): The file path of the npy data file to load
        input_image_width (int): Width (n) of the input (n*n) image
        transform (Transform): Desired transformations of the input image if required
    """

    def __init__(self, file_path, input_image_width, input_channels, transform=None):
        """
        Data Loader for the Cryo-EM Dataset.

        Attributes:
            file_path (str): The file path of the npy data file to load
            input_image_width (int): Width (n) of the input (n*n) image
            transform (Transform): Desired transformations of the input image if required
        """

        # Loading Data
        total_data = np.load(file_path)
        total_data = total_data.reshape(-1, input_channels, input_image_width, input_image_width)

        self.data = torch.from_numpy(total_data)
        self.num_images = self.data.shape[0]
        self.input_channels = input_channels
        self.transform = transform

        # TODO: Normalise data.

    def __getitem__(self, index):
        """
        Return one image after the specified transformations.
        """

        sample = self.data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        """
        Return the number of images.
        """

        return self.num_images


def get_data_loader(file_path,
                    input_image_width=40,
                    input_channels=1,
                    batch_size=4,
                    shuffle=False,
                    num_workers=None):
    """
    Get the Data Loader for the Cryo EM Dataset.
    """

    dataset = CryoEMDataset(
        file_path,
        input_image_width,
        input_channels
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return data_loader
