"""
This file contains the training loop for the proposed VAE-GAN model in the paper:
Estimation of Orientation and Camera Parameters from Cryo-Electron Microscopy Images with Variational Autoencoders and Generative Adversarial Networks
"""
import torch


def training_loop(data_loader,
                  model,
                  optimizer_list,
                  device,
                  num_epochs=100,
                  lambda_regularisation_loss=0.01,
                  lambda_cone_loss=1,
                  lambda_gan_loss=0.01):
    optimizer_encoder = optimizer_list[0]
    optimizer_decoder = optimizer_list[1]
    optimizer_discriminator = optimizer_list[2]

    for epoch in range(num_epochs):
        for index, current_batch in enumerate(data_loader):
            # Store data in computing device
            x = current_batch.to(device)

            # Forward Pass
            reconstructed_x, real_discriminator_preds, fake_discriminator_preds, mus, log_variances, z = model(x)

            # Loss Computation
            reconstruction_loss = torch.sum(
                0.5 * (x.view(len(x), -1) - reconstructed_x.view(len(reconstructed_x), -1)) ** 2
            )  # Params: Encoder+Decoder, Propagate: Encoder+Decoder

            regularisation_loss = torch.sum(
                -0.5 * torch.sum(-log_variances.detach().exp() - torch.pow(mus.detach(), 2) + log_variances.detach() + 1, 1)
            )  # KL Divergence. # Params: Encoder, Propagate: Encoder

            z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
            cone_loss = torch.sum(z1**2 + z2**2 - z3**2)

            gan_loss_original = torch.sum(
                -torch.log(real_discriminator_preds + 1e-3)
            )

            gan_loss_predicted = torch.sum(
                -torch.log(1 - fake_discriminator_preds + 1e-3)
            )

            gan_loss = gan_loss_original + gan_loss_predicted  # Params: Encoder+Decoder+Discriminator, Propagate: Decoder+Discriminator

            encoder_loss = reconstruction_loss + lambda_regularisation_loss * regularisation_loss + lambda_cone_loss * cone_loss
            decoder_loss = reconstruction_loss + lambda_gan_loss * gan_loss_predicted
            discriminator_loss = gan_loss

            # Clear Gradients
            model.zero_grad()

            # Encoder Backward Pass
            encoder_loss.backward(retain_graph=True)  # TODO: Want to compute only gradients of encoder params.
            optimizer_encoder.step()

            # Decoder Backward Pass
            decoder_loss.backward(retain_graph=True)  # TODO: Want to compute gradients of only decoder params.
            optimizer_decoder.step()

            # # Discriminator Backward Pass
            discriminator_loss.backward(retain_graph=True)  # TODO: Want to compute gradients of only decoder params.
            optimizer_discriminator.step()

            # Print every 10th epoch
            if (epoch + 1) % 2 == 0:
                print(
                    f"epoch={epoch + 1}/{num_epochs}, "
                    f"encoder_loss={encoder_loss.item():.4f}, "
                    f"decoder_loss={decoder_loss.item():.4f}, "
                    f"discriminator_loss={discriminator_loss.item():.4f}"
                )
