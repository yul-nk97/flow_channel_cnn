import os
import random
from typing import Tuple, List, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from flow_channel_cnn.models import ChannelVAE
from flow_channel_cnn.utils import EXPERIMENTS_PATH
from flow_channel_cnn.utils import render_latex, latex_table 

# == SOURCE PARAMETERS ==
# The path to the source dataset folder

# :param SOURCE_PATH:
#       The path to the source dataset folder. THis folder should contain specific .NPY files which 
#       contain the images and the labels of the dataset.
SOURCE_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'dataset')

# == TRANING PARAMETERS ==

EPOCHS: int = 10
BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-4

# == EVALUATION PARAMETERS ==

NUM_EXAMPLES: int = 15


__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

def load_dataset(e: Experiment
                 ) -> Tuple[list, list]:
    """
    
    """
    
    # ~ train set
    x_train_path = os.path.join(e.SOURCE_PATH, 'X_removed_island.npy')
    x_train = np.load(x_train_path)
    
    y_train_path = os.path.join(e.SOURCE_PATH, 'y_removed_island.npy')
    y_train = np.load(y_train_path)
    
    # ~ test set
    x_test_path = os.path.join(e.SOURCE_PATH, 'X_test_removed_island.npy')
    x_test = np.load(x_test_path)
    
    y_test_path = os.path.join(e.SOURCE_PATH, 'y_test_removed_island.npy')
    y_test = np.load(y_test_path)
    
    # ~ flat channels
    x_flat_path = os.path.join(e.SOURCE_PATH, 'X_flat.npy')
    x_flat = np.load(x_flat_path)
    
    y_flat_path = os.path.join(e.SOURCE_PATH, 'y_flat.npy')
    y_flat = np.load(y_flat_path)
    
    return (
        [(x, y) for x, y in zip(x_train, y_train)] + [(x, y) for x, y in zip(x_flat[5:], y_flat[5:])],
        [(x, y) for x, y in zip(x_test, y_test)] + [(x, y) for x, y in zip(x_flat[:5], y_flat[:5])],
    )


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
    # ~ data loading
    e.log('loading flow channel dataset...')
    train, test = load_dataset(e)
    x_train = np.array([x.transpose(2, 0, 1) for x, _ in train])[:, :, :128, :]
    
    # Scale the target values
    scaler = StandardScaler()
    y_train = np.array([y for _, y in train])
    y_train = scaler.fit_transform(y_train)
    
    x_test = np.array([x.transpose(2, 0, 1) for x, _ in test])[:, :, :128, :]
    y_test = np.array([y for _, y in test])
    #y_test = scaler.transform(y_test)
    
    e.log('example shape:')
    e.log(x_train[0].shape)
    input_shape = (x_train[0].shape[1], x_train[0].shape[2])
    height, width = input_shape
    
    # ~ model training
    e.log('constructing model...')
    model = ChannelVAE(
        input_channels=1,
        input_shape=input_shape,
        units=[128, 128, 128, 128, 128, 8],
        latent_dim=512,
        learning_rate=e.LEARNING_RATE,
        kl_factor=1e-10,
    )
    e.log('model summary:')
    e.log(str(model))
    
    # Convert data to PyTorch tensors
    e.log('converting data to tensors...')
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create datasets
    e.log('creating tensor datasets...')
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=e.BATCH_SIZE, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=e.BATCH_SIZE, 
        shuffle=False
    )

    # Train the model
    e.log('training model...')
    trainer = pl.Trainer(max_epochs=e.EPOCHS)
    trainer.fit(model, train_loader, test_loader)
    model.eval()
    
    # Choose NUM_EXAMPLES samples from the test set
    e.log(f'Evaluating reconstructions...')
    indices_example = random.sample(range(len(test)), e.NUM_EXAMPLES)
    for c, index in enumerate(indices_example):
        
        e.log(f' * example {c}')
        x_example = x_test[index:index+1]

        mu, var_log = model.encode(torch.tensor(x_example, dtype=torch.float32))
        z = model.reparameterize(mu, var_log)
        x_recon = model.decode(z).cpu().detach().numpy()

        # Plot the original and reconstructed images
        fig, (ax_orig, ax_recon) = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
        ax_orig.imshow(x_example[0, 0, :, :], cmap='Greys')
        ax_orig.set_title('Original')
        ax_orig.axis('off')
        
        ax_recon.imshow(x_recon[0, 0, :, :], cmap='Greys')
        ax_recon.set_title('Reconstruction')
        ax_recon.axis('off')

        e.commit_fig(f'reconstruction_{index}.png', fig)
        
    # Choose two random images from the test set
    e.log(f'Interpolating between two random test images...')
    for c in range(5):
        index1, index2 = random.sample(range(len(test)), 2)
        x1 = x_test[index1:index1+1]
        x2 = x_test[index2:index2+1]

        mu1, var_log1 = model.encode(torch.tensor(x1, dtype=torch.float32))
        z1 = model.reparameterize(mu1, var_log1)
        
        mu2, var_log2 = model.encode(torch.tensor(x2, dtype=torch.float32))
        z2 = model.reparameterize(mu2, var_log2)

        # Interpolate between z1 and z2
        num_interpolations = 10
        interpolations = np.linspace(0, 1, num_interpolations)
        fig, axes = plt.subplots(2, num_interpolations, figsize=(20, 4))
        
        for i, alpha in enumerate(interpolations):
            z_interp = (1 - alpha) * z1 + alpha * z2
            x_interp = model.decode(z_interp).cpu().detach().numpy()
            
            axes[1, i].imshow(x_interp[0, 0, :, :], cmap='Greys')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'{alpha:.2f}')
        
        # Plot the original images
        axes[0, 0].imshow(x1[0, 0, :, :], cmap='Greys')
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Original 1')
        
        axes[0, -1].imshow(x2[0, 0, :, :], cmap='Greys')
        axes[0, -1].axis('off')
        axes[0, -1].set_title('Original 2')
        
        for ax in axes[0, 1:-1]:
            ax.axis('off')
        
        e.commit_fig(f'interpolation_{c}.png', fig)

    # Randomly sample from the latent space and plot the reconstructions
    e.log(f'Generating random samples from the latent space...')
    num_samples = 10
    z_random = torch.randn(num_samples, model.latent_dim)
    x_random = model.decode(z_random).cpu().detach().numpy()

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))
    for i in range(num_samples):
        axes[i].imshow(x_random[i, 0, :, :], cmap='Greys')
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
    
    e.commit_fig(f'random_samples.png', fig)


experiment.run_if_main()