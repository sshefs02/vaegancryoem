"""
This file contains the test script for the proposed VAE-GAN model in the paper:
Estimation of Orientation and Camera Parameters from Cryo-Electron Microscopy Images with Variational Autoencoders and Generative Adversarial Networks
"""
import matplotlib.pyplot as plt
import torch

from vaegancryoem.script.data.data_loader import get_data_loader
from vaegancryoem.script.models.vaegan import VAEGAN

# Load Model
# ---------------------------------------------------------------------

input_channels = 1
input_image_width = 40
latent_space_dimensions = 3

model = VAEGAN(
    input_channels,
    input_image_width,
    latent_space_dimensions
)

PATH = 'model_state_dictionary'
model.load_state_dict(torch.load(PATH))

# Load Data
# ---------------------------------------------------------------------

file_path = "/Users/shesriva/Desktop/RA/vaegancryoem/cryoem/total_processed_data/test.npy"
input_image_width = 40
input_channels = 1
batch_size = 4
shuffle = False
num_workers = 0

data_loader = get_data_loader(
    file_path,
    input_image_width,
    input_channels,
    batch_size,
    shuffle,
    num_workers
)

# Plot and Test Data
# ---------------------------------------------------------------------


def test(data_loader, num_epochs=1, device='cpu'):
    model.eval()
    i = 1
    for index, current_batch in enumerate(data_loader):
        # TODO: Store data in computing device
        if i in range(5):
            x = current_batch.to(device)
            reconstructed_x, real_discriminator_preds, fake_discriminator_preds, mus, log_variances, z = model(x)
            plot_original_and_reconstructed_image(x, reconstructed_x)
        i += 1


def plot_original_and_reconstructed_image(x, reconstructed_x):
    num_images = min(4, x.shape[0])
    fig, axs = plt.subplots(num_images, 2)
    with torch.no_grad():
        for i in range(num_images):
            current_image = x[i][0]
            current_reconstruction = reconstructed_x[i][0]
            axs[i][0].imshow(current_image, cmap='gray')
            axs[i][1].imshow(current_reconstruction, cmap='gray')
        plt.show()


test(data_loader)
