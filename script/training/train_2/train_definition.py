"""
This file contains the training loop for the proposed VAE-GAN model in the paper:
Estimation of Orientation and Camera Parameters from Cryo-Electron Microscopy Images with Variational Autoencoders and Generative Adversarial Networks
"""
# import copy
# import matplotlib.pyplot as plt
# from torchsummary import summary
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable


def training_loop(data_loader,
                  encoder,
                  decoder,
                  discriminator,
                  optimizer_list,
                  device,
                  num_epochs=100,
                  lambda_regularisation_loss=0.01,
                  lambda_cone_loss=1,
                  lambda_gan_loss=0.01):
    optimizer_encoder = optimizer_list[0]
    optimizer_decoder = optimizer_list[1]
    optimizer_discriminator = optimizer_list[2]

    # Print All models
    # print(summary(encoder, (1, 40, 40)))
    # print(summary(decoder, (12,)))
    # print(summary(discriminator, (1, 40, 40)))

    # Loss function
    adversarial_loss = nn.BCELoss()

    for epoch in range(num_epochs):
        for index, current_batch in enumerate(data_loader):

            # Encoder Forward Pass.
            x = current_batch.to(device)
            mus, log_variances = encoder(x)

            logvar = log_variances.mul(0.5).exp_()
            eps = Variable(logvar.data.new(logvar.size()).normal_())
            z = eps.mul(logvar).add_(mus)

            # Decoder/Generator Forward Pass.
            reconstructed_x = decoder(z.detach())

            # ------------------------------------------------------------
            # Training the Discriminator
            # ------------------------------------------------------------

            # Discriminator Forward Pass.
            disc_preds_for_real_images = discriminator(x)
            disc_preds_for_generated_images = discriminator(reconstructed_x.detach())

            # Adversarial ground truths
            valid = Variable(Tensor(x.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(reconstructed_x.size(0), 1).fill_(0.0), requires_grad=False)

            # Clear Gradients
            optimizer_discriminator.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(disc_preds_for_real_images, valid)
            fake_loss = adversarial_loss(disc_preds_for_generated_images, fake)
            discriminator_loss = (real_loss + fake_loss) / 2

            discriminator_loss.backward(retain_graph=True)
            optimizer_discriminator.step()

            # ------------------------------------------------------------
            # Training the Generator/Decoder
            # ------------------------------------------------------------

            # Discriminator Forward Pass.
            disc_preds_for_generated_images = discriminator(reconstructed_x)

            # Clear Gradients
            optimizer_decoder.zero_grad()

            # Measure decoder's ability to generated samples that look real
            reconstruction_loss = torch.mean(0.5 * (x.view(len(x), -1) - reconstructed_x.view(len(reconstructed_x), -1)) ** 2)
            decoder_loss = lambda_gan_loss * adversarial_loss(disc_preds_for_generated_images, valid) + reconstruction_loss

            decoder_loss.backward(retain_graph=True)
            optimizer_decoder.step()

            # ------------------------------------------------------------
            # Training the Encoder
            # ------------------------------------------------------------

            # Encoder Pass
            reconstructed_x = decoder(z)
            reconstruction_loss = torch.mean(0.5 * (x.view(len(x), -1) - reconstructed_x.view(len(reconstructed_x), -1)) ** 2)

            # Clear Gradients
            optimizer_encoder.zero_grad()

            # Compute Loss
            regularisation_loss = torch.mean(-0.5 * torch.sum(-log_variances.exp() - torch.pow(mus, 2) + log_variances + 1, 1), dim=0)
            z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
            cone_loss = torch.sum(z1 ** 2 + z2 ** 2 - z3 ** 2) ** 2
            encoder_loss = reconstruction_loss + lambda_regularisation_loss * regularisation_loss + lambda_cone_loss * cone_loss

            encoder_loss.backward()
            optimizer_encoder.step()

            # ------------------------------------------------------------
            # Print Losses
            # ------------------------------------------------------------

            # Print every 2nd epoch
            if (epoch + 1) % 1 == 0:
                print(
                    f"epoch={epoch + 1}/{num_epochs}, "
                    f"encoder_loss={encoder_loss.item():.4f}, "
                    f"decoder_loss={decoder_loss.item():.4f}, "
                    f"discriminator_loss={discriminator_loss.item():.4f}"
                )
                print()


