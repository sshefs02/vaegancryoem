"""
This file contains the training loop for the proposed VAE-GAN model in the paper:
Estimation of Orientation and Camera Parameters from Cryo-Electron Microscopy Images with Variational Autoencoders and Generative Adversarial Networks
"""
# import copy
# import matplotlib.pyplot as plt
# import torch


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
            reconstructed_x, real_discriminator_preds, fake_discriminator_preds, mus, log_variances, z = model(x, "GENERATOR")

            # Loss
            reconstruction_loss, regularisation_loss, cone_loss, gan_loss_original, gan_loss_predicted, gan_loss_generator, gan_loss = model.get_loss(
                x, reconstructed_x, mus, log_variances, z, real_discriminator_preds, fake_discriminator_preds
            )

            # print(
            #     f"reconstruction_loss={reconstruction_loss.item()}\n"
            #     f"regularisation_loss={regularisation_loss.item():.4f}\n"
            #     f"cone_loss={cone_loss.item():.4f}\n"
            #     f"gan_loss_original={gan_loss_original.item():.4f}\n"
            #     f"gan_loss_predicted={gan_loss_predicted.item():.4f}\n"
            #     f"gan_loss={gan_loss.item():.4f}"
            # )
            # print()

            encoder_loss = reconstruction_loss + lambda_regularisation_loss * regularisation_loss + lambda_cone_loss * cone_loss
            decoder_loss = reconstruction_loss + lambda_gan_loss * gan_loss_generator
            discriminator_loss = gan_loss

            # Clear Gradients
            model.zero_grad()

            # ------- Encoder Backward Pass --------

            # old_param_list_encoder = copy.deepcopy(list(model.encoder.parameters()))
            # old_param_list_decoder = copy.deepcopy(list(model.decoder.parameters()))
            # old_param_list_discriminator = copy.deepcopy(list(model.discriminator.parameters()))

            encoder_loss.backward(retain_graph=True)
            optimizer_encoder.step()

            # new_param_list_encoder = copy.deepcopy(list(model.encoder.parameters()))
            # new_param_list_decoder = copy.deepcopy(list(model.decoder.parameters()))
            # new_param_list_discriminator = copy.deepcopy(list(model.discriminator.parameters()))

            # ------- Decoder Backward Pass --------

            # print()

            # old_param_list_encoder = copy.deepcopy(list(model.encoder.parameters()))
            # old_param_list_decoder = copy.deepcopy(list(model.decoder.parameters()))
            # old_param_list_discriminator = copy.deepcopy(list(model.discriminator.parameters()))

            decoder_loss.backward(retain_graph=True)
            optimizer_decoder.step()

            # new_param_list_encoder = copy.deepcopy(list(model.encoder.parameters()))
            # new_param_list_decoder = copy.deepcopy(list(model.decoder.parameters()))
            # new_param_list_discriminator = copy.deepcopy(list(model.discriminator.parameters()))

            # ------- Discriminator Backward Pass --------

            # print()

            # old_param_list_encoder = copy.deepcopy(list(model.encoder.parameters()))
            # old_param_list_decoder = copy.deepcopy(list(model.decoder.parameters()))
            # old_param_list_discriminator = copy.deepcopy(list(model.discriminator.parameters()))

            # TODO: Train discriminator in separate step. 
            discriminator_loss.backward()
            optimizer_discriminator.step()

            # new_param_list_encoder = copy.deepcopy(list(model.encoder.parameters()))
            # new_param_list_decoder = copy.deepcopy(list(model.decoder.parameters()))
            # new_param_list_discriminator = copy.deepcopy(list(model.discriminator.parameters()))

            # Show Images
            # plot_original_and_reconstructed_image(x, reconstructed_x)

            # Print every 2nd epoch
            if (epoch + 1) % 1 == 0:
                print(
                    f"epoch={epoch + 1}/{num_epochs}, "
                    f"encoder_loss={encoder_loss.item():.4f}, "
                    f"decoder_loss={decoder_loss.item():.4f}, "
                    f"discriminator_loss={discriminator_loss.item():.4f}"
                )
                print()


