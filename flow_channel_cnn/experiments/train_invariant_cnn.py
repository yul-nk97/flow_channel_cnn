import os
import random
from datetime import datetime
from pickle import dump
from typing import Tuple, List, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from models import InvariantCNN
from utils import EXPERIMENTS_PATH
from utils import render_latex, latex_table 


# == SOURCE PARAMETERS ==
# The path to the source dataset folder

# :param SOURCE_PATH:
#       The path to the source dataset folder. THis folder should contain specific .NPY files which 
#       contain the images and the labels of the dataset.
#SOURCE_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'dataset')
SOURCE_PATH: str = "/home/yuri/Documents/chemegml/DataAug_ChemEngAI/dataset"
# SOURCE_PATH: str = "./dataset/"

# == MODEL PARAMETERS ==
# Parameters related to the construction of the model

# :param CONV_UNITS:
#       The number of convolutional units to use for the model. This should be a list of integers where
#       each integer represents the number of filters to use for each convolutional layer. Each integer
#       in the list will represent/add one layer.
CONV_UNITS: List[int] = [93,93,93,93]
# :param DENSE_UNITS:
#       The number of dense units to use for the model. This should be a list of integers where each integer
#       represents the number of units to use for each dense layer. Each integer in the list will represent/add
#       one layer. The final number of units in this list should match the number of target values that 
#       the model should predict.
DENSE_UNITS: List[int] = 1365
# :param KERNEL_SIZE:
#       The kernel size to use for the convolutional layers.
KERNEL_SIZE: int = 9
# :param USE_APS:
#       Whether to use the Adaptive Polyphase Sampling (APS) layer in the model. If set to True, the model will
#       use the APS layer after each convolutional layer to perform the strided downsampling.
USE_APS: bool = True

KERNEL_SIZE_CUSTOMSUM: int = 4

N_OUTPUT_CUSTOMSUM: int = 12

# == TRANING PARAMETERS ==
# Parameters related to the training process of the model

# :param EPOCHS:
#       The number of epochs to train the model for.
EPOCHS: int = 10 #TODO
# :param BATCH_SIZE:
#       The batch size to use for training.
BATCH_SIZE: int = 4 #74 #TODO
# :param LEARNING_RATE:
#       The learning rate to use for traianing.
LEARNING_RATE: float = 1e-3

FINAL_LATE: float = 0.000004520

DROPOUT:float = 0.3608

# == EVALUATION PARAMETERS ==

# :param NUM_EXAMPLES:
#       The number of examples to use for the shift and flip invariance evaluation.
#       Selecting a high number here may incur a high runtime after the training of the model.
NUM_EXAMPLES: int = 5


__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

def load_dataset(e: Experiment
                 ) -> Tuple[list, list]:
    """
    Loads the dataset and returns it as a tuple (train_list, test_list) of lists where the 
    first element is a list containing tuples (x, y) of the training set samples and the second 
    element is a list containing elements (x, y) of the test set samples.
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
    x_flat_path = os.path.join(e.SOURCE_PATH, 'x_flat.npy')
    x_flat = np.load(x_flat_path)
    
    y_flat_path = os.path.join(e.SOURCE_PATH, 'y_flat.npy')
    y_flat = np.load(y_flat_path)
    
    return (
        [(x, y) for x, y in zip(x_train[:100], y_train[:100])] + [(x, y) for x, y in zip(x_flat[5:], y_flat[5:])],
        [(x, y) for x, y in zip(x_test[:100], y_test[:100])] + [(x, y) for x, y in zip(x_flat[:5], y_flat[:5])],
    )


@experiment.hook('evaluate_shift_invariance', replace=False, default=True)
def evaluate_shift_invariance(e: Experiment,
                              model: InvariantCNN,
                              x: np.array,
                              key: str = 'test',
                              ) -> np.ndarray:
    """
    Given a ``model`` an input element ``x`` and a unique string key, this function will evaluate the shift invariance
    of the model on that given element by applying all possible horizontal shifts of the image and comparing the 
    predictions of the original image with the predictions of the shifted images.
    
    This function will also plot the results and save them to the experiment archive folder.
    """
    # constructing the shifted versions
    width = x.shape[3]
    shifts = list(range(width))
    x_shifted = np.concatenate([np.roll(x, shift, axis=3) for shift in shifts], axis=0)

    # query the model
    out_example = model.forward_array(x) 
    out_shifted = model.forward_array(x_shifted)
    
    # calculate the differences as percentages of the original prediction
    diffs = []
    for shift, out_shift in zip(shifts, out_shifted):
        diff = np.abs(out_example - out_shift)
        diff = diff / np.mean(out_example)*100
        diffs.append(diff)
        
    diffs = np.concatenate(diffs, axis=0)
    fig, (ax_img, ax_cf, ax_st) = plt.subplots(ncols=3, nrows=1, figsize=(20, 6))
    ax_img.imshow(x[0, 0], cmap='gray')
    ax_img.set_title(f'Original Image - {key}\n'
                     f'$C_f$: {out_example[0, 0]:.3f} - $S_t$: {out_example[0, 1]:.3f}')
    
    ax_st.plot(shifts, diffs[:, 0])
    ax_st.set_xlabel('Shift [px]')
    ax_st.set_ylabel('Prediction deviation [%]')
    ax_st.set_title('$C_f$')
    y_lo, y_hi = ax_st.get_ylim()
    if y_hi < 0.1:
        ax_st.set_ylim(-0.02, 0.1)

    ax_cf.plot(shifts, diffs[:, 1])
    ax_cf.set_xlabel('Shift [px]')
    ax_cf.set_ylabel('Prediction deviation [%]')
    ax_cf.set_title('$St$')
    y_lo, y_hi = ax_cf.get_ylim()
    if y_hi < 0.1:
        ax_cf.set_ylim(-0.02, 0.1)
    
    e.commit_fig(f'shift_invariance_{key}.png', fig)
    
    # At the end we want to return a numpy array that contains the differences of 
    return diffs


@experiment.hook('evaluate_flip_invariance', default=False, replace=False)
def evaluate_flip_invariance(e: Experiment,
                             model: InvariantCNN,
                             x: np.ndarray,
                             key: str = 'test'
                             ) -> np.ndarray:
    """
    Given a ``model`` and an input element ``x``, this function will evaluate the vertical flip invariance of the model
    on that given element by comparing the predictions of the original image with the predictions of the flipped image.
    """
    x_flip = np.flip(x, axis=2)
    
    # ~ query the model with both
    out = model.forward_array(x)
    out_flip = model.forward_array(x_flip)

    # calculating the percentage(!) of the deviation w.r.t. the original prediction
    diff = out - out_flip
    diff = diff / np.abs(out)
    
    return diff


@experiment
def experiment(e: Experiment):
    
    e.log("Starting 5-Fold Cross-Validation experiment...")
    
    # ~ 1) Data Loading
    e.log('loading flow channel dataset...')
    train, test = load_dataset(e)
    
    x_train_raw = np.array([x.transpose(2, 0, 1) for x, _ in train])
    y_train_raw = np.array([y for _, y in train])
    
    x_test_raw = np.array([x.transpose(2, 0, 1) for x, _ in test])
    y_test_raw = np.array([y for _, y in test])
    
    # Merge them into full arrays
    X = np.concatenate([x_train_raw, x_test_raw], axis=0)
    Y = np.concatenate([y_train_raw, y_test_raw], axis=0)
    
    # ~ 2) Scale Targets
    e.log('Scaling the target values...')
    scaler = StandardScaler()
    Y = scaler.fit_transform(Y)  # shape still (N_total, 2)
    
    # Save the scaler
    scaler_file = os.path.join(e.path, 'y_scaler.npy')
    with open(scaler_file, 'wb') as f:
        dump(scaler, f)
        
    # Some logging
    e.log(f'Full dataset shape: X={X.shape}, Y={Y.shape}')
    input_shape = (X.shape[2], X.shape[3])  # (H, W)
    e.log(f'Example shape (H, W): {input_shape}')
    
    # ~ 3) Convert to PyTorch Dataset
    X_tensor = torch.tensor(X, dtype=torch.float32)  # shape (N_total, C, H, W)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)  # shape (N_total, 2)
    full_dataset = TensorDataset(X_tensor, Y_tensor)
    
    # ~ 4) Set up 5-fold cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(len(full_dataset))
    
    # Prepare arrays to store performance metrics across folds
    all_r2_cf, all_mae_cf, all_mse_cf = [], [], []
    all_r2_st, all_mae_st, all_mse_st = [], [], []
    
    # ~ 5) Cross-validation loop
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices)):
        e.log(f'\n===== Fold {fold_idx+1}/{n_splits} =====')
        
        # Create subsets for this fold
        train_subset = Subset(full_dataset, train_idx)
        test_subset  = Subset(full_dataset, test_idx)
        
        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=e.BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_subset,  batch_size=e.BATCH_SIZE, shuffle=False)
        
        # ~ Re-initialize the model for each fold
        e.log('Initializing a new model for this fold...')
        model_fold = InvariantCNN(
            input_dim=1,
            input_shape=input_shape,
            conv_units=e.CONV_UNITS,
            dense_units=e.DENSE_UNITS,
            learning_rate=e.LEARNING_RATE,
            final_lr=e.FINAL_LATE,
            kernel_size=e.KERNEL_SIZE,
            use_aps=e.USE_APS,
            epochs=e.EPOCHS,
            dropout=e.DROPOUT,
            kernel_size_customsum=e.KERNEL_SIZE_CUSTOMSUM,
            n_output_customsum=e.N_OUTPUT_CUSTOMSUM
        )
        
        # ~ Train
        e.log('Training model...')
        trainer = pl.Trainer(max_epochs=e.EPOCHS)
        trainer.fit(model_fold, train_loader, test_loader)
        model_fold.eval()
        
        # Attach scaler if needed for inference
        model_fold.set_scaler(scaler)
        
        # Save the model checkpoint for this fold
        fold_model_path = os.path.join(e.path, f'model_fold_{fold_idx+1}.ckpt')
        e.log(f'Saving model to {fold_model_path}')
        model_fold.save(fold_model_path)
        
        # ~ Evaluate on test fold
        e.log('Evaluating model...')
        x_test_list, y_test_list = [], []
        for xb, yb in test_loader:
            x_test_list.append(xb)
            y_test_list.append(yb)
        
        x_test_tensor = torch.cat(x_test_list, dim=0)
        y_test_tensor = torch.cat(y_test_list, dim=0)
        
        # Convert to numpy
        x_test_eval = x_test_tensor.numpy()
        y_test_eval = y_test_tensor.numpy()  # shape (num_test_samples, 2)
        
        # Forward pass (inference)
        y_pred = model_fold.forward_array(x_test_eval, use_scaler=True)
        
        # ~ Calculate metrics per output dimension
        r2_value_cf  = r2_score(y_test_eval[:, 0], y_pred[:, 0])
        mae_value_cf = mean_absolute_error(y_test_eval[:, 0], y_pred[:, 0])
        mse_value_cf = mean_squared_error(y_test_eval[:, 0], y_pred[:, 0])
        
        r2_value_st  = r2_score(y_test_eval[:, 1], y_pred[:, 1])
        mae_value_st = mean_absolute_error(y_test_eval[:, 1], y_pred[:, 1])
        mse_value_st = mean_squared_error(y_test_eval[:, 1], y_pred[:, 1])
        
        # Log fold-specific metrics
        e.log(f' * Fold {fold_idx+1}: C_f - R2={r2_value_cf:.3f}, MAE={mae_value_cf:.3f}, MSE={mse_value_cf:.3f}')
        e.log(f' * Fold {fold_idx+1}: S_t - R2={r2_value_st:.3f}, MAE={mae_value_st:.3f}, MSE={mse_value_st:.3f}')
        
        # Store them for cross-validation summary
        all_r2_cf.append(r2_value_cf)
        all_mae_cf.append(mae_value_cf)
        all_mse_cf.append(mse_value_cf)
        
        all_r2_st.append(r2_value_st)
        all_mae_st.append(mae_value_st)
        all_mse_st.append(mse_value_st)
        
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #           INTEGRATE YOUR PLOTTING AND INVARIANCE CODE HERE
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # 1) Create a parity plot for S_t using Seaborn
        fig, ax_st = plt.subplots(figsize=(6, 6))
        
        # We typically do x=True, y=Pred for a parity plot.
        # However, if you truly want a 2D histogram, you can do it as below.
        sns.histplot(
            x=y_test_eval[:, 1],        # True S_t
            y=y_pred[:, 1],            # Predicted S_t
            bins=50, pmax=0.9, ax=ax_st, cmap="Blues", cbar=True
        )
        # 1:1 line
        min_val = min(y_test_eval[:, 1].min(), y_pred[:, 1].min())
        max_val = max(y_test_eval[:, 1].max(), y_pred[:, 1].max())
        ax_st.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        # Title + labels
        ax_st.set_title(
            f'$S_t$ Regression\nR²: {r2_value_st:.3f}, '
            f'MAE: {mae_value_st:.3f}, MSE: {mse_value_st:.3f}'
        )
        ax_st.set_xlabel('True Values')
        ax_st.set_ylabel('Predicted Values')
        
        # Save the figure for this fold
        reg_plot_path = f'regression_plot_fold_{fold_idx+1}.png'
        e.commit_fig(reg_plot_path, fig)
        plt.close(fig)  # clean up figure
        
        # 2) Evaluate shift invariance
        #    This code snippet picks random images from the test set for invariance tests
        e.log(f'Evaluating shift invariance on {e.NUM_EXAMPLES} examples (Fold {fold_idx+1})...')
        
        x_test_for_invariance = x_test_eval  # shape: (N_test, 1, H, W)
        shift_diffs = []
        flip_diffs  = []
        
        num_tests = x_test_for_invariance.shape[0]
        
        for c in range(e.NUM_EXAMPLES):
            e.log(f' * Example {c+1}/{e.NUM_EXAMPLES}')
            index = random.randint(0, num_tests - 1)
            
            # x_example has shape (1, 1, H, W)
            x_example = x_test_for_invariance[index:index+1]
            
            # shift invariance
            shift_diff = e.apply_hook(
                'evaluate_shift_invariance',
                model=model_fold,
                x=x_example,
                key=c,
            )
            shift_diffs.append(shift_diff)
            
            # flip invariance
            flip_diff = e.apply_hook(
                'evaluate_flip_invariance',
                model=model_fold,
                x=x_example,
                key=c,
            )
            flip_diffs.append(flip_diff)
        
        # Stack results
        shift_diffs = np.stack(shift_diffs, axis=0)  # shape ~ (NUM_EXAMPLES, ???, 2)
        flip_diffs  = np.stack(flip_diffs, axis=0)   # shape ~ (NUM_EXAMPLES, ???, 2)
        
        # 3) Create LaTeX table summarizing invariance results
        _, latex_string = latex_table(
            column_names=[
                '$C_f$ Flip Diff (%)',
                '$S_t$ Flip Diff (%)',
                '$C_f$ Shift Diff (%)',
                '$S_t$ Shift Diff (%)'
            ],
            rows=[[
                flip_diffs[:, :, 0].flatten().tolist(),
                flip_diffs[:, :, 1].flatten().tolist(),
                shift_diffs[:, :, 0].flatten().tolist(),
                shift_diffs[:, :, 1].flatten().tolist(),
            ]],
        )
        
        # Save invariance table for this fold
        tex_path = os.path.join(e.path, f'invariances_fold_{fold_idx+1}.tex')
        e.commit_raw(tex_path, latex_string)
        
        # If you want to render directly to PDF and you have a function like `render_latex`:
        # pdf_path = os.path.join(e.path, f'invariances_fold_{fold_idx+1}.pdf')
        # render_latex({'content': latex_string}, output_path=pdf_path)
        # e.log(f"Invariance PDF saved to {pdf_path}")
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #           END OF PLOTTING AND INVARIANCE INTEGRATION
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # ~ 6) Summarize Performance Across Folds
    e.log('\n===== Cross-validation Summary =====')
    # C_f metrics
    avg_r2_cf  = np.mean(all_r2_cf)
    std_r2_cf  = np.std(all_r2_cf)
    avg_mae_cf = np.mean(all_mae_cf)
    std_mae_cf = np.std(all_mae_cf)
    avg_mse_cf = np.mean(all_mse_cf)
    std_mse_cf = np.std(all_mse_cf)
    
    e.log(f'C_f: R2={avg_r2_cf:.3f} ± {std_r2_cf:.3f}, '
          f'MAE={avg_mae_cf:.3f} ± {std_mae_cf:.3f}, '
          f'MSE={avg_mse_cf:.3f} ± {std_mse_cf:.3f}')
    
    # S_t metrics
    avg_r2_st  = np.mean(all_r2_st)
    std_r2_st  = np.std(all_r2_st)
    avg_mae_st = np.mean(all_mae_st)
    std_mae_st = np.std(all_mae_st)
    avg_mse_st = np.mean(all_mse_st)
    std_mse_st = np.std(all_mse_st)
    
    e.log(f'S_t: R2={avg_r2_st:.3f} ± {std_r2_st:.3f}, '
          f'MAE={avg_mae_st:.3f} ± {std_mae_st:.3f}, '
          f'MSE={avg_mse_st:.3f} ± {std_mse_st:.3f}')
    
    e.log('5-Fold Cross-Validation completed.')

    # If you want to store final CV metrics in your experiment:
    e['cv_metrics/cf/r2_mean'] = avg_r2_cf
    e['cv_metrics/cf/r2_std'] = std_r2_cf
    e['cv_metrics/cf/mae_mean'] = avg_mae_cf
    e['cv_metrics/cf/mse_mean'] = avg_mse_cf
    e['cv_metrics/st/r2_mean'] = avg_r2_st
    e['cv_metrics/st/r2_std'] = std_r2_st
    e['cv_metrics/st/mae_mean'] = avg_mae_st
    e['cv_metrics/st/mse_mean'] = avg_mse_st
    
experiment.run_if_main()