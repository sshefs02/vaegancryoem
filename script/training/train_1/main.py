"""
This file contains the main run for the proposed VAE-GAN model in the paper:
Estimation of Orientation and Camera Parameters from Cryo-Electron Microscopy Images with Variational Autoencoders and Generative Adversarial Networks
"""
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from vaegancryoem.script.training.train_1.train_definition import training_loop
from vaegancryoem.script.data.data_loader import get_data_loader
from vaegancryoem.script.models.vaegan import VAEGAN
import torch
import numpy

# Set Anomaly Detection to True for Reproducibility
# --------------------------------------------------------------------

torch.autograd.set_detect_anomaly(True)

# Set Seed for Reproducibility
# ---------------------------------------------------------------------

# seed = random.randint(1, 10000) # use if you want new results
seed = 999
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print(f"Random Seed: {seed}")

# Load Data
# ---------------------------------------------------------------------

# Set Device
device = 'cpu'
if torch.cuda.is_available():
    device = 'gpu'

# Load Data
# ---------------------------------------------------------------------

# TODO: Pick from Arguments.
file_path = "/vaegancryoem/cryoem/total_processed_data/train.npy"
input_image_width = 40
input_channels = 1
batch_size = 64
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

# Create Model
# ---------------------------------------------------------------------

# TODO: Pick from Arguments.
latent_space_dimensions = 12

model = VAEGAN(
    input_channels,
    input_image_width,
    latent_space_dimensions
)

# Define Optimisers and Loss
# ---------------------------------------------------------------------

# TODO: Pick from Arguments.
encoder_optimiser_learning_rate = 0.00000000001
decoder_optimiser_learning_rate = 0.001
discriminator_optimiser_learning_rate = 0.0001


def get_optimiser(parameters, learning_rate, beta1=0.9, beta2=0.999):
    # Define ADAM Optimiser
    optimizer = Adam(
        params=parameters,
        lr=learning_rate,
        betas=(beta1, beta2)
    )

    # Define Scheduler
    lr = MultiStepLR(
        optimizer,
        milestones=[2],
        gamma=1
    )
    return optimizer, lr


optimizer_encoder, scheduler_encoder = get_optimiser(
    model.encoder.parameters(),
    learning_rate=encoder_optimiser_learning_rate
)

optimizer_decoder, scheduler_decoder = get_optimiser(
    model.decoder.parameters(),
    learning_rate=decoder_optimiser_learning_rate
)

optimizer_discriminator, scheduler_decoder = get_optimiser(
    model.discriminator.parameters(),
    learning_rate=discriminator_optimiser_learning_rate
)

# Training Loop
# ---------------------------------------------------------------------

# TODO: Pick from Arguments.
num_epochs = 4
lambda_regularisation_loss = 0.1
lambda_cone_loss = 1
lambda_gan_loss = 0.1
optimizer_list = [optimizer_encoder, optimizer_decoder, optimizer_discriminator]

training_loop(
    data_loader,
    model,
    optimizer_list,
    device,
    num_epochs=num_epochs,
    lambda_regularisation_loss=lambda_regularisation_loss,
    lambda_cone_loss=lambda_cone_loss,
    lambda_gan_loss=lambda_gan_loss
)

# TODO: Save the trained model.
model_save_path = 'model_state_dictionary'
torch.save(model.state_dict(), model_save_path)
