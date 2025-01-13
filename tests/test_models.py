import os
import torch
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler

from flow_channel_cnn.experiments.models import AbstractCNN
from flow_channel_cnn.experiments.models import InvariantCNN
from flow_channel_cnn.models import ChannelVAE
from pytorch_lightning.utilities.model_summary import ModelSummary

from .utils import ARTIFACTS_PATH


class TestAbstractCNN:
    """
    Unittests for the AbstractCNN class which serves as the base class for all CNN models in the 
    package.
    """
    
    def test_construction_basically_works(self):
        """
        Basic test if instantiation of a new model instance works.
        """
        model = AbstractCNN()
        assert isinstance(model, pl.LightningModule) 
        assert isinstance(model, AbstractCNN)
    
    def test_saving_loading_works(self):
        """
        If it works to save a model to a checkpoint file and then load it again from that.
        """
        model = AbstractCNN()
        
        with tempfile.TemporaryDirectory() as path:
            model_path = os.path.join(ARTIFACTS_PATH, 'model.ckpt')
            model.save(model_path)
        
            model_loaded = model.load(model_path)
            assert isinstance(model_loaded, AbstractCNN)
            
    def test_saving_loading_with_scaler_works(self):
        """
        If it works to save a model to a checkpoint file and then load it again from that. 
        Specifically we want to test if the scaler is also saved and loaded correctly.
        """
        model = AbstractCNN()
        scaler = StandardScaler()
        scaler.fit(np.random.rand(100, 1))
        model.set_scaler(scaler)
        
        with tempfile.TemporaryDirectory() as path:
            model_path = os.path.join(ARTIFACTS_PATH, 'model.ckpt')
            model.save(model_path)
        
            model_loaded = model.load(model_path)
            assert isinstance(model_loaded, AbstractCNN)
            assert isinstance(model_loaded.scaler, StandardScaler)
            assert np.allclose(model_loaded.scaler.mean_, scaler.mean_)


class TestInvariantCNN:
    """
    Unittests for the InvariantCNN class which is a specific CNN model that is invariant to vertical 
    flipping and horizontal shifting of the input data.
    """
    
    def test_construction_basically_works(self):
        """
        Basic test if instantiation of a new model instance works.
        """
        model = InvariantCNN(
            input_dim=3,
            input_shape=(100, 100),
        )
        assert isinstance(model, pl.LightningModule)
        assert isinstance(model, InvariantCNN)

        summary = ModelSummary(model)
        print(summary)
                
    def test_forward_basically_works(self):
        
        # First of all we need to generate some kind of input data with which we can test 
        # the forward method of the model. We will use random data for this purpose.
        batch_size = 32
        num_channels = 1
        width = 400
        height = 100
        data = torch.tensor(np.random.rand(batch_size, num_channels, height, width), dtype=torch.float32)
        
        model = InvariantCNN(
            input_dim=num_channels, 
            input_shape=(height, width),
            dense_units=[64, 2],
        )
        out = model.forward(data)
        assert out.shape == (batch_size, 2)
        
    def test_backward_basically_works(self):
        
        batch_size = 32
        num_channels = 1
        width = 400
        height = 100
        data = torch.tensor(np.random.rand(batch_size, num_channels, height, width), dtype=torch.float32)
        target = torch.tensor(np.random.randint(0, 2, size=(batch_size,)), dtype=torch.long)
        
        model = InvariantCNN(
            input_dim=num_channels, 
            input_shape=(height, width),
            dense_units=[64, 2],
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        
    def test_flip_invariance(self):
        
        # here we initialize a random input tensor and it's flipped version.
        width, height = 400, 100
        x = torch.tensor(np.random.rand(1, 1, height, width), dtype=torch.float32)
        x_flip = torch.flip(x, dims=[2])
        
        model = InvariantCNN(
            input_dim=1, 
            input_shape=(height, width),
            dense_units=[64, 2],
        )
        
        # Then we pass both versions through the model and we expect the result to be the 
        # same
        out = model.forward(x)
        out_flip = model.forward(x_flip)
        
        assert torch.allclose(out, out_flip, atol=1e-6)
        
    def test_shift_invariance(self):
        """
        The forward pass of the model should be inherently shift invariant. This means that if horizontally 
        shifted images are fed into the model the output should not change at all. This should even be 
        true for a non-trained randomly initialized model and is being tested here.
        """
        # here we initialize a random input tensor and we also want all of it's shifted versions
        # as well!
        width, height = 200, 100
        arr = np.random.rand(1, 1, height, width)
        arrs_shift = []
        for shift in range(width):
            arrs_shift.append(np.roll(arr, shift,  axis=3))
            
        arrs_shift = np.concatenate(arrs_shift, axis=0)
        
        x = torch.tensor(arr, dtype=torch.float32)
        x_shift = torch.tensor(arrs_shift, dtype=torch.float32)
        
        model = InvariantCNN(
            input_dim=1, 
            input_shape=(height, width),
            conv_units=[16, 16, 16],
            dense_units=[64, 2],
            kernel_size=6,
        )
    #     model = InvariantCNN(
    #     input_dim=1, 
    #     input_shape=input_shape,
    #     conv_units=e.CONV_UNITS,
    #     dense_units=e.DENSE_UNITS,
    #     learning_rate=e.LEARNING_RATE,
    #     final_lr=e.FINAL_LATE,
    #     kernel_size=e.KERNEL_SIZE,
    #     use_aps=e.USE_APS,
    #     epochs=e.EPOCHS,
    #     dropout=e.DROPOUT,
    #     kernel_size_customsum = e.KERNEL_SIZE_CUSTOMSUM,
    #     n_output_customsum = e.N_OUTPUT_CUSTOMSUM
    # )
        model.eval()
        
        # Then we pass the original through the model and all of the shifted versions and then 
        # we analyze for each shift how much the output has changed in percent w.r.t. to the 
        # original unshifted output.
        out = model.forward(x).detach().numpy()[0]
        outs_shift = model.forward(x_shift).detach().numpy()
        
        diffs = []
        for shift, out_shift in zip(range(0, width), outs_shift):
            diff = out - out_shift
            diff = diff / np.abs(out)
            diff = np.mean(diff)
            diffs.append(diff)
            
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))
        ax.plot(range(width), diffs, label='Shift Difference')
        ax.set_xlabel('Shift [px]')
        ax.set_ylabel('Prediction Difference [%]')
        ax.set_title('Shift Invariance Test')
        if ax.get_ylim()[1] < 0.1:
            y_lo, y_hi = ax.get_ylim()
            ax.set_ylim(y_lo, 0.1)
            
        plt.show()
        fig_path = os.path.join(ARTIFACTS_PATH, 'shift_invariance_test.png')
        fig.savefig(fig_path)
        
    def test_forward_array(self):
        """
        The forward_array method should perform a forward pass on a numpy array instead of a 
        torch tensor for convenience. However the result should be the same.
        """
        width, height = 200, 100
        arr = np.random.rand(32, 1, height, width)
        x = torch.tensor(arr, dtype=torch.float32)
        
        model = InvariantCNN(
            input_dim=1, 
            input_shape=(height, width),
            conv_units=[16, 16, 16],
            dense_units=[64, 1],
        )
        model.eval()
        
        out = model.forward(x).detach().numpy()
        out_array = model.forward_array(arr)
        
        assert np.allclose(out, out_array)
        
    def test_saving_loading_works(self):
        """
        If it works to save a model to a checkpoint file and then load it again from that.
        """
        model = InvariantCNN(
            input_dim=3,
            input_shape=(100, 100),
        )
        
        with tempfile.TemporaryDirectory() as path:
            model_path = os.path.join(path, 'model.ckpt')
            model.save(model_path)
        
            model_loaded = model.load(model_path)
            assert isinstance(model_loaded, AbstractCNN)
            assert isinstance(model_loaded, InvariantCNN)
            
            assert model_loaded.input_dim == 3
            assert model_loaded.input_shape == (100, 100)
            
            
class TestChannelVAE:
    
    def test_construction_basically_works(self):
        
        model = ChannelVAE(
            input_channels=1,
            input_shape=(100, 100),
            units=[16, 16, 16],
            latent_dim=128,
        )
        
        assert isinstance(model, pl.LightningModule)
        assert isinstance(model, ChannelVAE)
        
        summary = ModelSummary(model)
        print(summary)
        
        print('pre latent shape', model.pre_latent_shape)
        print('pre latent dim', model.pre_latent_dim)
        
    def test_encode_basically_works(self):
        
        model = ChannelVAE(
            input_channels=1,
            input_shape=(128, 128),
            units=[16, 16, 16],
            latent_dim=128,
        )
        
        arr = np.random.rand(32, 1, 128, 128)
        x = torch.tensor(arr, dtype=torch.float32)
        
        mu, log_var = model.encode(x)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(log_var, torch.Tensor)
        
        assert mu.shape == (32, 128)
        assert log_var.shape == (32, 128)
        
    def test_encode_decode_basically_works(self):
        
        shape = (1, 128, 384)
        model = ChannelVAE(
            input_channels=shape[0],
            input_shape=(shape[1], shape[2]),
            units=[16, 16, 16],
            latent_dim=128,
        )
        
        arr = np.random.rand(32, *shape)
        x = torch.tensor(arr, dtype=torch.float32)
        
        mu, log_var = model.encode(x)
        
        z = model.reparameterize(mu, log_var)
        x_recon = model.decode(z)
        
        assert x_recon.shape == x.shape