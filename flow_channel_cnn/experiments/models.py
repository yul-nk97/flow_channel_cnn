import os
import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from layers import AdaptivePolyphaseSampling, CustomSum


# == CONVOLUTIONAL NEURAL NETWORK MODELS ==

class AbstractCNN(pl.LightningModule):
    """
    Abstract base class for CNN based models.
    
    This base class provides the following generic functionality:
    - Saving and loading of the model as a checkpoint file on the disk.
    - Managing a StandardScaler instance that can be attached to the model.
    - Providing convenient wrapper methods such as ``forward_array`` which can be used from the outside of
      perform a forward pass directly on a numpy array instead of having convert to a torch Tensor first.
    """
    
    def __init__(self,
                 **kwargs,
                 ) -> None:
        super().__init__(**kwargs)
        
        self.scaler: Optional[StandardScaler] = None
        
    # -- convenience methods --
    
    def forward_array(self,
                      array: np.ndarray,
                      batch_size: int = 32,
                      use_scaler: bool = True,
                      ) -> np.ndarray:
        """
        This is a convenience method that allows to pass a numpy ``array`` to the model to performa a forward pass.
        This method also takes care of the batching of the input data so that the model can handle larger amounts of
        data at once.
        
        :param array: The numpy array to pass to the model.
        :param batch_size: The batch size to use for the forward pass.
        :param use_scaler: A flag that indicates whether to use the scaler of the model to scale the output back to
            to the original scale.
            
        :returns: The output of the model as a numpy array.
        """
        assert isinstance(array, np.ndarray), 'The input must be a numpy array!'
        assert array.ndim == 4, 'The input array must have 4 dimensions: (batch, channels, height, width)!'
        
        # We simply convert the numpy array to a torch tensor and then pass it to the normal forward 
        # function. But additionally we also want to take care of the batching here.
        tensor: Tensor = torch.tensor(array.copy(), dtype=torch.float32)
        loader = DataLoader(tensor, batch_size=batch_size, shuffle=False)
        out: list[Tensor] = []
        for x in loader:
            out.append(self.forward(x))
        
        out: Tensor = torch.cat(out, dim=0)
        
        # Then we also convert the output to a numpy array
        arr: np.ndarray = out.cpu().detach().numpy()
        # And finally we also want to scale the output back to the original scale if a scaler is set and 
        # the corresponding flag is set.
        if use_scaler and self.scaler is not None:
            arr = self.scaler.inverse_transform(arr)
        
        return arr
        
    # -- scaler management --

    def set_scaler(self, 
                scaler: StandardScaler
                ) -> None:
        """
        Add a standard ``scaler`` to the model which will then be used to scale the output prediction values 
        when performing predictions.
        
        :param scaler: The StandardScaler instance to use for scaling the output values.
        
        :returns: None
        """
        self.scaler = scaler
        
    def set_scaler_from_parameters(self, 
                                   parameters: dict
                                   ) -> None:
        """
        Set the scaler of the model from a given set of ``parameters``. This is useful to restore the
        scaler from a saved model checkpoint.
        
        :param parameters: The parameters of the scaler to set.
        
        :returns: None
        """
        if len(parameters) != 0:
            self.scaler = StandardScaler()
            self.scaler.mean_ = parameters['mean_']
            self.scaler.var_ = parameters['var_']
            self.scaler.scale_ = parameters['scale_']
        
    def get_scaler_parameters(self) -> dict:
        """
        Returns the parameters of the scaler of the model. This is useful to save the parameters of the
        scaler to a model checkpoint.
        
        :returns: The parameters of the scaler.
        """
        parameters = {}
        if self.scaler:
            parameters.update({
                'mean_': self.scaler.mean_,
                'var_': self.scaler.var_,
                'scale_': self.scaler.scale_,
            })
            
        return parameters
        
    # -- saving and loading --
        
    def save(self, 
             path: str
             ) -> None:
        """
        Given the absolute string ``path`` to save the model checkpoint, this method will save the model to that
        checkpoint.
        
        :param path: The absolute string path to save the model checkpoint.
        
        :returns: None
        """
        torch.save({
            'state_dict': self.state_dict(),
            'hyper_parameters': self.hparams,
            'pytorch-lightning_version': pl.__version__,
            'scaler_parameters': self.get_scaler_parameters(),   
        }, path)

    @classmethod
    def load(cls,
             path: str
             ) -> Any:
        """
        Given the absolute string ``path`` to the model checkpoint path, this method will load the model
        from that checkpoint and return it.
        
        :param path: The absolute string path to the model checkpoint.
        
        :returns: None
        """
        # The load_from_checkpoint method is already pre-implemented in the LightningModule class which will 
        # load the actual model.
        # We also want to put it into evaluation mode after loading.
        model = cls.load_from_checkpoint(path)
        model.eval()
        
        # Additionally we also want to load the scaler parameters from the checkpoint and set the scaler
        # of the model accordingly.
        data: dict = torch.load(path)
        parameters: dict = data['scaler_parameters']
        model.set_scaler_from_parameters(parameters)
        
        return model



class LogLearningRateScheduler(_LRScheduler):
    """
    Logarithmically reduce learning rate from lr_start to lr_stop over 'epochs' epochs,
    with an initial 'epomin' epoch phase in which LR is held constant at lr_start.

    :param optimizer: PyTorch optimizer whose LR we want to schedule.
    :param lr_start:  Initial (max) learning rate to keep for the first 'epomin' epochs.
    :param lr_stop:   Final (min) learning rate at the end of 'epochs' (after epomin).
    :param epochs:    Total number of epochs for the schedule to go from lr_start to lr_stop.
    :param epomin:    Number of epochs to keep LR constant at lr_start (warm-up).
    :param verbose:   If True, prints a message each time the LR is updated.
    :param last_epoch: By PyTorch convention, set -1 for a fresh start.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_start: float = 1e-3,
        lr_stop: float = 1e-5,
        epochs: int = 100,
        epomin: int = 10,
        verbose: bool = False,
        last_epoch: int = -1
    ):
        self.lr_start = lr_start
        self.lr_stop = lr_stop
        self.epochs = epochs
        self.epomin = epomin
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        Called internally by PyTorch each time you do `scheduler.step()`.
        Returns a list of LRs (one per param group) for the current epoch index (self.last_epoch).
        """
        epoch = self.last_epoch  # 0-based after the first `scheduler.step()`
        
        if epoch < self.epomin:
            # For the first `epomin` epochs, keep LR fixed at lr_start
            lr = self.lr_start
        else:
            # Replicate the formula from your TF code:
            #
            # out = exp(
            #   log(lr_start)
            #   - (log(lr_start) - log(lr_stop)) / (epochs - epomin) * (epoch - epomin)
            # )
            lr = math.exp(
                math.log(self.lr_start) -
                (math.log(self.lr_start) - math.log(self.lr_stop))
                / (self.epochs - self.epomin)
                * (epoch - self.epomin)
            )
        
        # Return one LR value per param_group
        return [lr for _ in self.optimizer.param_groups]
    
def periodic_padding_flexible(tensor, axis, padding=1):
        """
        Add periodic (wrap-around) padding to a PyTorch tensor along one or more axes.
        """
        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(padding, int):
            padding = (padding,)

        for ax, p in zip(axis, padding):
            if p > 0:
                # Slices for "right" side and "left" side
                ind_right = [slice(-p, None) if i == ax else slice(None) for i in range(tensor.dim())]
                ind_left  = [slice(0,  p)     if i == ax else slice(None) for i in range(tensor.dim())]
            
                right  = tensor[tuple(ind_right)]
                left   = tensor[tuple(ind_left)]
                tensor = torch.cat([right, tensor, left], dim=ax)

        return tensor
    

class InvariantCNN(AbstractCNN):
    """
    Invariant CNN model that is invariant to vertical flipping and the horizontal shifting of the input data.
    
    :param input_dim: The number of channels in the input images.
    :param input_shape: A tuple (height, width) that specifies the dimensions of the input images
    :param conv_units: A list of integers that specify the number of units (=channels) in the corresponding layers
        Each element in this list represents/adds another layer in the convolutional part of the network.
    :param dense_units: A list of integers that specify the number of units in the corresponding layers of the dense
        prediction part of the network. This part of the network is used to make the final prediction.
        Each element in this list represents/adds another layer in the dense part of the network.
    :param learning_rate: The learning rate to use for the optimizer.
    :param kernel_size: The kernel size to use for the convolutional layers. For symmetry reasons this needs to be 
        an even number to preserve the shift invariance.
    :param stride: The stride to use for the downsampling in the convolutional layers.
    :param use_aps: A flag that indicates whether to use the Adaptive Polyphase Sampling layer to eliminate the shift
        variance of the striding operation.
    :param kernel_size_customsum: Kernel size for the 1D convolution in CustomSum
    :param n_output_customsum:    Number of filters (out_channels) in the 1D convolution
    """
    
    def __init__(self, 
                 input_dim: int, 
                 input_shape: Tuple[int, int],
                 conv_units: List[int] = [93,93,93,93], 
                 pool_size: int =2,
                 epochs: int = 150,
                 dense_units: int = 1365,
                 learning_rate: float = 1e-3,
                 final_lr: float = 0.000004520,
                 kernel_size: int = 9,
                 kernel_size_customsum: int = 4,
                 stride: int = 2,
                 use_aps: bool = True,
                 n_output_customsum: int = 12,
                 dropout: int = 0.3608,
                 **kwargs,
                 ):
        
        super().__init__(**kwargs)
        
        # The kernel size needs to be an even number for the circular padding to interact correctly with 
        # the APS stride reduction.
        # assert kernel_size % 2 == 0, 'The kernel size must be an even number!'
        
        self.input_dim = input_dim
        self.input_shape = input_shape
        self.conv_units = conv_units
        self.pool_size = pool_size
        self.epochs = epochs
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.final_lr = final_lr
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_aps = use_aps
        self.kernel_size_customsum = kernel_size_customsum
        self.n_output_customsum = n_output_customsum
        self.dropout = dropout
        
        
        # Optionally we can add a standard scaler to the model which will be used to scale the output 
        # prediction values back to their original scale. This will have to be added manually after
        # initializing the model.
        self.scaler: Optional[StandardScaler] = None
        
        height, width = self.input_shape
        self.hparams.update({
            'input_dim': input_dim,
            'input_shape': input_shape,
            'conv_units': conv_units,
            'pool_size': pool_size,
            'dense_units': dense_units,
            'learning_rate': learning_rate,
            'final_lr' : final_lr,
            'kernel_size': kernel_size,
            'stride': stride,
            'use_aps': use_aps,
            'kernel_size_customsum' : kernel_size_customsum,
            'n_output_customsum' : n_output_customsum
        })
    
        # This is the size of the concatenated embedding vector that will be fed into the final prediction 
        # network. We will calculate the actual size of this as we are setting up the network conv layers in 
        # the subsequent lines of code.
        self.embedding_size: int = 0
        
        # ~ convolutional encoder
        
        self.layers_conv = nn.ModuleList()
        self.customsum_list = nn.ModuleList()        
        
        prev_units = input_dim
        _height = height
        for units in conv_units:
            # (1) Conv
            self.layers_conv.append(
                nn.Conv2d(in_channels=prev_units,
                          out_channels=units,
                          kernel_size=self.kernel_size,
                          stride=1,
                          padding=0)  # no built-in padding
            )
            # (2) Activation
            self.layers_conv.append(nn.ReLU())

            # (3) Pool
            self.layers_conv.append(
                nn.MaxPool2d(kernel_size=2,
                             stride=(1 if use_aps else self.stride),
                             padding=0)
            )

            # (4) APS if requested
            if use_aps:
                self.layers_conv.append(AdaptivePolyphaseSampling(stride=self.stride, p=2))

            # For the aggregator per block
            self.customsum_list.append(
                CustomSum(in_channels=units, out_channels=self.n_output_customsum, kernel_size=self.kernel_size_customsum)
            )

            # Prepare for next block
            prev_units = units
            _height = math.ceil(_height / 2)

        # self.embedding_size = 0
        # _height = height  # Start with input height

        # for block_idx, units in enumerate(conv_units):
        #     _height = math.ceil(_height / 2)  # Pooling halves the height
        #     self.embedding_size += n_output_customsum * _height  # Add per-block contribution
        #     print(f"Block {block_idx}: height = {_height}, contribution = {n_output_customsum * _height}")

        # print(f"Final self.embedding_size = {self.embedding_size}")



        # ~ Dense Prediction Network
        self.dense1 = nn.Linear(1344, self.dense_units)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.dense2 = nn.Linear(self.dense_units, 2)

    def embedd_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        A layer-by-layer forward pass that mirrors your 'call' code.
        Prints debug info at each step: Conv -> Act -> Pool -> APS, etc.
        """
        #print("\n   ###   Start NN")
        csum_index = 0

        # Each block = {Conv, Act, Pool, [APS]}, so 4 sub-layers if APS is used,
        # otherwise 3 if APS is off.
        sublayers_per_block = 4 if self.use_aps else 3
        intermediates = []
        for block_idx in range(len(self.customsum_list)):
            #print(f"\n--- BLOCK {block_idx+1} ---")

            # (1) Conv layer
            conv_layer = self.layers_conv[block_idx * sublayers_per_block + 0]
            # print("   ###   Conv layer")
            padding_required = (self.kernel_size - 1) // 2
            x = periodic_padding_flexible(x, axis=(2, 3), padding=(padding_required, padding_required))
            # print(f"cnn layer: input shape {x.shape}, applying periodic padding (height,width)")
            x = conv_layer(x)
            # print(f"After Conv: {x.shape}")

            # (2) Activation
            act_layer = self.layers_conv[block_idx * sublayers_per_block + 1]
            # print("   ###   Act layer")
            x = act_layer(x)
            # print(f"After Act: {x.shape}")

            # (3) Pool
            pool_layer = self.layers_conv[block_idx * sublayers_per_block + 2]
            # print("   ###   Pool layer ")
            x = periodic_padding_flexible(x, axis=2, padding=1)
            # print(f'cnn layer: periodic padding for pool {x.shape}, axis: (1), padding: (1)')
            x = pool_layer(x)
            # print(f"After Pool: {x.shape}")

            # (4) APS if use_aps
            if self.use_aps:
                aps_layer = self.layers_conv[block_idx * sublayers_per_block + 3]
                #print("   ###   APS layer")
                x = aps_layer(x)
                # print(f"After APS: {x.shape}")
                intermediates.append(x)

        summed = []
        for vector in intermediates:
            # Summation 
            # print("   ###   Summation ")
            vector = vector.sum(dim=3)
            # print(f'Summation {vector.shape}')
            x_csum = self.customsum_list[csum_index](vector)
            #print(f'Summation {vector.shape}')
            #x_csum = self.customsum_list[csum_index](vector.sum(dim=3))
            # print(f"After CustomSum: {x_csum.shape}")
            csum_index += 1
            # Flatten
            x_flat = x_csum.view(x_csum.size(0), -1)
            # print(f"Append flatten representations: {x_flat.shape}")

            # Append to intermediates
            summed.append(x_flat)

        # Concatenate embeddings
        embedding = torch.cat(summed, dim=1)
        print("\nFinal concatenated embedding shape:", embedding.shape)
        return embedding

        
        
    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        """
        This method performs a single forward pass on the input tensor ``x`` of the shape (batch_size, num_channels, height, width) 
        and returns a final prediction tensor with the shape (batch_size, num_outputs).
        """
        # Flip invariance
        emb_orig = self.embedd_single(x)
        
        x_flip = torch.flip(x, dims=[2])    # vertical flip
        emb_flip = self.embedd_single(x_flip)
        
        emb = emb_orig + emb_flip
        # Dense prediction
        out = self.dense1(emb)    # First linear
        out = self.relu(out)      # ReLU
        out = self.dropout_layer(out)   # Dropout
        out = self.dense2(out)    # Final linear layer => shape (N, 2)
        print(out.shape)
        return out
        
    
    def training_step(self, 
                      batch: tuple, 
                      batch_idx: int
                      ) -> torch.Tensor:
        """
        This method implements the calculation of the training loss function which will then internally 
        be used to update the weights of the network with a single training ``batch``.
        """
        x, y_true = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y_true)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = {
            'scheduler': LogLearningRateScheduler(
                optimizer=optimizer,
                lr_start=self.learning_rate,
                lr_stop=self.final_lr,
                epochs=self.epochs,
                epomin=10,
                verbose=False
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    

# == VARIATIONAL AUTOENCODER ==
    
    
class WarmupKLScheduler(pl.Callback):

    def __init__(self, 
                    value_start: float = 1e-10, 
                    value_end: float = 1.0, 
                    warmup_steps: int = 10_000,
                    ) -> None:
        super().__init__()
        self.value_start = value_start
        self.value_end = value_end
        self.warmup_steps = warmup_steps
        self.step = 0

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if self.step < self.warmup_steps:
            kl_factor = self.value_start + (self.value_end - self.value_start) * (self.step / self.warmup_steps)
        else:
            kl_factor = self.value_end
        
        pl_module.kl_factor = kl_factor
        self.step += 1
    
    
class CyclicKLScheduler(pl.Callback):

    def __init__(self, 
                 value_start: float = 1e-10, 
                 value_end: float = 5.0, 
                 frequency: int = 100,
                 warmup_steps: int = 250,
                 ) -> None:
        super().__init__()
        self.value_start = value_start
        self.value_end = value_end
        self.frequency = frequency
        self.warmup_steps = warmup_steps
        self.step = 0

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if self.step < self.warmup_steps:
            kl_factor = self.value_start
        else:
            cycle = (self.step - self.warmup_steps) // self.frequency
            x = ((self.step - self.warmup_steps) % self.frequency) / self.frequency
            kl_factor = self.value_start + (self.value_end - self.value_start) * (0.5 * (1 - math.cos(math.pi * x)))
        
        pl_module.kl_factor = kl_factor
        self.step += 1
    


class ChannelVAE(pl.LightningModule):
    
    def __init__(self,
                 input_channels: int,
                 input_shape: Tuple[int, int],
                 units: List[int],
                 latent_dim: int,
                 stride: int = 2,
                 kernel_size: int = 3,
                 learning_rate: float = 1e-3,
                 kl_factor: float = 0.1,
                 # discriminator related
                 use_discriminator: bool = False,
                 discriminator_units: List[int] = [128, 128, 128, 128, 128],
                 **kwargs,
                 ) -> None:
        super().__init__(**kwargs)
        
        self.input_channels = input_channels
        self.input_shape = input_shape
        self.units = units
        self.latent_dim = latent_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.kl_factor = kl_factor
        self.use_discriminator = use_discriminator
        self.discriminator_units = discriminator_units
        
        # ~ encoder
        _height, _width = self.input_shape
        self.encoder_layers = nn.ModuleList()
        prev_units = input_channels
        for units in units:
            lay = nn.Sequential(
                nn.Conv2d(
                    in_channels=prev_units, 
                    out_channels=units, 
                    kernel_size=self.kernel_size,
                    stride=self.stride, 
                    padding=1,
                    padding_mode='replicate',
                ),
                nn.BatchNorm2d(units),
                nn.ReLU(),
            )
            self.encoder_layers.append(lay)
            prev_units = units
            
            _height = math.ceil(_height / self.stride)
            _width = math.ceil(_width / self.stride)
            
        self.pre_latent_shape = (prev_units, _height, _width)
        self.pre_latent_dim = _height * _width * prev_units
        
        self.lay_mu = nn.Sequential(
            nn.Linear(self.pre_latent_dim, latent_dim),
        )
        self.lay_log_var = nn.Sequential(
            nn.Linear(self.pre_latent_dim, latent_dim),
        )
        
        # ~ decoder
        self.lay_from_latent = nn.Sequential(
            nn.Linear(latent_dim, self.pre_latent_dim),
        )
        self.decoder_layers = nn.ModuleList()
        prev_units = self.units[-1]
        for units in list(reversed(self.units[:-1])):
            lay = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=prev_units, 
                    out_channels=units, 
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    output_padding=1, 
                    padding=1,
                    padding_mode='zeros',
                ),
                nn.BatchNorm2d(units),
                nn.ReLU(),
            )
            self.decoder_layers.append(lay)
            prev_units = units
            
        lay = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=prev_units, 
                out_channels=input_channels, 
                kernel_size=self.kernel_size,
                stride=self.stride,
                output_padding=1, 
                padding=1
            ),
            nn.Sigmoid(),
        )
        self.decoder_layers.append(lay)
        
        # ~ discriminator
        self.automatic_optimization = False
    
        self.discriminator_layers = nn.ModuleList()
        prev_units = input_channels
        _height, _width = self.input_shape
        for units in self.discriminator_units:
            lay = nn.Sequential(
                nn.Conv2d(
                    in_channels=prev_units, 
                    out_channels=units, 
                    kernel_size=3,
                    stride=self.stride, 
                    padding=1,
                    padding_mode='replicate',
                ),
                nn.BatchNorm2d(units),
                nn.ReLU(),
            )
            self.discriminator_layers.append(lay)
            prev_units = units
            
            _height = math.ceil(_height / self.stride)
            _width = math.ceil(_width / self.stride)
            
        lay = nn.Sequential(
            nn.Flatten(),
            nn.Linear(units * _height * _width, 1),
            nn.Sigmoid(),
        )
        self.discriminator_layers.append(lay)
        self.discriminator_shape = (1, _height, _width)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def discriminate(self, x):
        
        for lay in self.discriminator_layers:
            x = lay(x)
        
        return x
    
    def encode(self, x):
        
        for lay in self.encoder_layers:
            x = lay(x)
        
        x = x.view(x.size(0), -1)
        
        mu = self.lay_mu(x)
        log_var = self.lay_log_var(x)
        log_var = F.softplus(log_var)
        
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        
        z = self.lay_from_latent(z)
        
        z = z.view(z.size(0), *self.pre_latent_shape)

        for lay in self.decoder_layers:
            z = lay(z)
            
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def training_step(self, batch, batch_idx):
        
        opt_vae, opt_disc = self.optimizers()
        
        x, _ = batch
        x_recon, mu, log_var = self.forward(x)
        recon_loss = F.l1_loss(x_recon, x, reduction="mean")
        
        kl_loss_ind = -0.1 * (1 + log_var - log_var.exp())
        kl_loss_ind = torch.mean(torch.sum(kl_loss_ind, axis=1))
        
        Z_mean = torch.mean(mu, axis=0)
        Z_log_var = torch.log(torch.var(mu, axis=0))
        
        kl_loss_g = -0.5 * (1 + Z_log_var - Z_mean.pow(2) - Z_log_var.exp())
        kl_loss_g = torch.sum(kl_loss_g, axis=0)
        kl_loss = kl_loss_ind + kl_loss_g
        
        u_fake = self.discriminate(x_recon)
        loss_gen_fake = F.binary_cross_entropy(u_fake, torch.ones_like(u_fake))
        
        loss_vae = 1000 * recon_loss + self.kl_factor * kl_loss + 10 * loss_gen_fake
        
        opt_vae.zero_grad()
        self.manual_backward(loss_vae)
        opt_vae.step()
        
        # ~ discriminator
        x_recon, mu, log_var = self.forward(x)
        
        u_real = self.discriminate(x)
        u_fake = self.discriminate(x_recon)
        
        loss_disc_real = F.binary_cross_entropy(u_real, torch.ones_like(u_real))
        loss_disc_fake = F.binary_cross_entropy(u_fake, torch.zeros_like(u_fake))
        loss_disc = (loss_disc_real + loss_disc_fake)
        
        # ~ gradient updates
    
        opt_disc.zero_grad()
        self.manual_backward(loss_disc)
        opt_disc.step()
        
        self.log("l_recon", recon_loss, prog_bar=True,)
        self.log("l_kl", kl_loss, prog_bar=True)
        self.log('l_gen_fk', loss_gen_fake, prog_bar=True)
        #self.log("kl_factor", self.kl_factor, prog_bar=True)
        self.log("l_disc", loss_disc, prog_bar=True)
        self.log("loss", loss_vae, prog_bar=True)

    def configure_optimizers(self):
        opt_vae = torch.optim.Adam(
            list(self.encoder_layers.parameters()) + 
            list(self.lay_mu.parameters()) + 
            list(self.lay_log_var.parameters()) + 
            list(self.lay_from_latent.parameters()) + 
            list(self.decoder_layers.parameters()), 
            lr=self.learning_rate
        )
        opt_disc = torch.optim.Adam(self.discriminator_layers.parameters(), lr=self.learning_rate * 0.01)
        return opt_vae, opt_disc
    
    def configure_callbacks(self):
        return [
            # CyclicKLScheduler(),
            WarmupKLScheduler(),    
        ]

if __name__ == "__main__":
    # Quick sanity check
    model = InvariantCNN(input_dim=3, input_shape=(64, 64))
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    print("Output shape:", out.shape)

