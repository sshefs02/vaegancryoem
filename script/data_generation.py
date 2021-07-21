"""
This file studies the data of cryo EM images to run the VAE-GAN proposed in the paper:
Estimation of Orientation and Camera Parameters from Cryo-Electron Microscopy Images with Variational Autoencoders and Generative Adversarial Networks
"""

import numpy as np
import os
import matplotlib.pyplot as plt

current_working_directory = str(os.getcwd())
cryoemdata_path = '/cryoem/'

hdb5_data_file_name = '5HDB_processed_train.npy'
hdb5_data_path = current_working_directory + cryoemdata_path + hdb5_data_file_name

antibody_data_file_name = 'antibody_processed_train.npy'
antibody_data_path = current_working_directory + cryoemdata_path + antibody_data_file_name

codhacs_data_file_name = 'codhacs_processed_train.npy'
codhacs_data_path = current_working_directory + cryoemdata_path + codhacs_data_file_name

antibody_data = np.load(antibody_data_path)
print(f"Antibody Data Shape = {antibody_data.shape}")

hdb5_data = np.load(hdb5_data_path)
print(f"HDB5 Data Shape = {hdb5_data.shape}")

codhacs_data = np.load(codhacs_data_path)
print(f"Codhacs Data Shape = {codhacs_data.shape}")


def plot_images(data, title):
    images = 20
    rows = 4
    columns = int(images/rows)
    plt.title(title)
    for i in range(20, images+20):
        current_image = data[i]
        plt.subplot(rows, columns, i-20 + 1)
        plt.imshow(current_image, cmap='gray')
    plt.show()


# plot_images(antibody_data, 'Antibody Data')  # Slightly Unclear Images
# plot_images(codhacs_data, 'Codhacs Data')  # Very Unclear Images
# plot_images(hdb5_data, 'HDB5 Data')  # Very Clear Images

total_data = np.concatenate([antibody_data, hdb5_data, codhacs_data], axis=0)
idx = np.arange(len(total_data))
np.random.shuffle(idx)

# plot_images(total_data, 'All Data')
# print(total_data.shape[0], codhacs_data.shape[0] + antibody_data.shape[0] + hdb5_data.shape[0])

test_ratio = 0.1
num_test_images = int(0.1 * len(idx))
test_data = total_data[:num_test_images]
train_data = total_data[num_test_images:]

train_data_path = current_working_directory + cryoemdata_path + 'total_processed_data/train.npy'
test_data_path = current_working_directory + cryoemdata_path + 'total_processed_data/test.npy'

np.save(train_data_path, train_data)
np.save(test_data_path, test_data)




