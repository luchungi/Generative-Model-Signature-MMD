import json
from collections import deque
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sigkernel as ksig
from model.generators import *
from utils.data import *
from utils.env import *

N_LAGS = 3 # for tensorboard writer to log autocorrelation coefficients
DATATYPE = torch.float32

def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))

def get_params_dicts(vars):
    '''
    Returns data_params, model_params and train_params dictionaries from vars dictionary
    '''
    data_params = {
        'batch_size': vars['batch_size'],
        'sample_len': vars['sample_len'],
        'seed': vars['seed'],
        'stride': vars['stride'],
        'start_date': vars['start_date'],
        'end_date': vars['end_date'],
        'lead_lag': vars['lead_lag'],
        'lags': vars['lags'],
    }

    model_params = {
        'static_kernel_type': vars['static_kernel_type'], # linear, rbf, rq
        'n_levels': vars['n_levels'], # truncated signature kernel levels
        'hidden_size': vars['hidden_size'],
        'activation': vars['activation'], # pytorch activation function name
        'n_lstm_layers': vars['n_lstm_layers'],
        'noise_dim': vars['noise_dim'],
        'seq_dim': vars['seq_dim'],
        'conditional': vars['conditional'],
        'ma': vars['ma'] if 'ma' in vars else False,
        'ma_p': vars['ma_p'], # to allow reproducibility in dataset class df.dropna() call
    }

    if model_params['conditional'] == True:
        model_params['hist_len'] = vars['hist_len']

    train_params = {
        'epochs': vars['epochs'],
        'start_lr': vars['start_lr'],
        'lr_factor': vars['lr_factor'],
        'patience': vars['patience'],
        'early_stopping': vars['early_stopping'],
        'kernel_sigma': vars['kernel_sigma'],
        'num_losses': vars['num_losses'],
    }

    return data_params, model_params, train_params

def start_writer(data_params, model_params, train_params, rl=False, rl_params=None, env_params=None, dir=None):
    '''
    Starts a tensorboard writer and logs data, model and training parameters
    Returns the writer
    '''
    static_kernel_type = model_params['static_kernel_type']
    n_levels = model_params['n_levels']
    if dir is None: # fresh run from start
        writer = SummaryWriter(comment=f'_{static_kernel_type}_{n_levels}')
        writer.add_text('Data parameters', pretty_json(data_params))
        writer.add_text('Model parameters', pretty_json(model_params))
        writer.add_text('Training parameters', pretty_json(train_params))
    else:
        writer = SummaryWriter(log_dir=dir, comment=f'_{static_kernel_type}_{n_levels}')

    writer.flush()
    return writer

def get_dataloader(**kwargs):
    '''
    Returns a dataloader for the specified sample_model
    '''

    sample_len = kwargs['sample_len']
    stride = kwargs['stride'] if 'stride' in kwargs else None
    start_date = kwargs['start_date'] if 'start_date' in kwargs else None
    end_date = kwargs['end_date'] if 'end_date' in kwargs else None
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else None
    seed = kwargs['seed'] if 'seed' in kwargs else None

    ma = kwargs['ma'] if 'ma' in kwargs else False
    noise_dim = kwargs['noise_dim'] if ma else None
    p = kwargs['ma_p'] if ma else None

    df = pd.read_csv('data/spx.csv', index_col=0, parse_dates=True)
    # NOTE: using the same dataset class even when not using MA to ensure reproducibility as MA dataset drops rows due to return calculation
    dataset = MADataset(df, start_date, end_date, sample_len, ma, noise_dim, p, stride, seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def get_signature_kernel(**kwargs):
    static_kernel_type = kwargs['static_kernel_type']
    kernel_sigma = kwargs['kernel_sigma']
    n_levels = kwargs['n_levels']

    if static_kernel_type == 'linear':
        static_kernel = ksig.kernels.LinearKernel()
    elif static_kernel_type == 'rbf':
        static_kernel = ksig.kernels.RBFKernel(sigma=kernel_sigma)
    elif static_kernel_type == 'rq':
        static_kernel = ksig.kernels.RationalQuadraticKernel(sigma=kernel_sigma)
    kernel = ksig.kernels.SignatureKernel(n_levels=n_levels, static_kernel=static_kernel)

    return kernel

def get_generator(**kwargs):
    sample_len = kwargs['sample_len']
    noise_dim = kwargs['noise_dim']
    seq_dim = kwargs['seq_dim']
    hidden_size = kwargs['hidden_size']
    n_lstm_layers = kwargs['n_lstm_layers'] if 'n_lstm_layers' in kwargs else None
    activation = kwargs['activation'] if 'activation' in kwargs else None

    generator = GenLSTM(noise_dim, seq_dim, sample_len, hidden_size=hidden_size, n_lstm_layers=n_lstm_layers, activation=activation)
    return generator

def compute_mmd_loss(kernel, X, output, lead_lag, lags):
    if lead_lag:
        X = batch_lead_lag_transform(X[:,:,1:], X[:,:,0:1], lags) # inputs are (price series, time dimension, lags to use)
        output = batch_lead_lag_transform(output[:,:,1:], output[:,:,0:1], lags)
    return ksig.loss.mmd_loss(X, output, kernel)

def train(generator, kernel, dataloader, rng, writer, device, checkpoint=None, **kwargs):
    '''
    Trains the generator model using the specified kernel and dataloader
    Args:
        generator (torch.nn.Module): The generator model to be trained.
        kernel (function): The kernel function used to compute the Maximum Mean Discrepancy (MMD) loss.
        dataloader (torch.utils.data.DataLoader): The dataloader object that provides the training data.
        rng (numpy.random.Generator): The random number generator used for generating standard Gaussian noise if moving average noise is not used.
        writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard writer for logging training information.
        device (torch.device): The device (CPU or GPU) on which the training will be performed.
        checkpoint (dict, optional): A dictionary containing the checkpoint information for resuming training.
            Defaults to None.
        **kwargs: Additional keyword arguments for training configuration.

    Keyword Args:
        epochs (int): The maximum number of training epochs.
        batch_size (int): The batch size for training.
        start_lr (float): The initial learning rate for the optimizer.
        lr_factor (float): The factor by which the learning rate is reduced on plateau.
        patience (int): The number of epochs with no improvement after which the learning rate is reduced (measured on average loss of last num_losses).
        early_stopping (int): The number of epochs with no improvement after which training is stopped (measured on average loss of last num_losses).
        num_losses (int): The number of previous losses to consider for computing the average loss.
        lead_lag (bool): Whether to consider lead-lag relationships in the data. Defaults to False.
        lags (list, optional): A list of lag values to consider for lead-lag relationships. Defaults to None.
        sample_len (int): The length of the generated samples.
        noise_dim (int): The dimension of the noise vector.
        ma (bool): Whether to use moving average noise. Defaults to False.
        conditional (bool): Whether to use conditional generation. Defaults to False.
        hist_len (int): The length of the history portion of the path. Required if conditional is True.
    '''
    dtype = DATATYPE
    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    start_lr = kwargs['start_lr']
    lr_factor = kwargs['lr_factor']
    patience = kwargs['patience']
    early_stopping = kwargs['early_stopping']
    num_losses = kwargs['num_losses']
    lead_lag = kwargs['lead_lag'] if 'lead_lag' in kwargs else False
    lags = kwargs['lags'] if 'lags' in kwargs else None
    sample_len = kwargs['sample_len']
    noise_dim = kwargs['noise_dim']
    ma = kwargs['ma'] if 'ma' in kwargs else False
    conditional = kwargs['conditional'] if 'conditional' in kwargs else False
    hist_len = kwargs['hist_len'] if conditional else None

    last_k_losses = deque(maxlen=num_losses) if checkpoint is None else checkpoint['last_k_losses']
    optimizer = torch.optim.Adam(generator.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_factor, verbose=True)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = 0 if checkpoint is None else checkpoint['epoch'] + 1
    best_loss = [np.inf, 0] if checkpoint is None else checkpoint['best_loss']
    for epoch in range(start_epoch, epochs):
        losses = [] # due to legacy code, losses is actually the mmd
        for X in tqdm(dataloader):
            if ma: # ma noise is generated for each batch within dataset class
                X, noise = X
                if noise.device != device: noise = noise.to(device)
            else: # Gaussian noise is generated for each batch
                noise = torch.tensor(rng.normal(size=(batch_size, sample_len-1, noise_dim)), device=device, dtype=dtype, requires_grad=False)
            X = X.to(device)

            t = X[:,:,:1] # time dimension of path is always first series of the last dim
            if conditional:
                hist_x = X[:,:hist_len,1:] # history portion of path
                output = generator(noise, t, hist_x=hist_x)
                output = torch.cat([t, output], axis=-1) # concatenate time and history + generated path along time series value dimension
                # remove history portion from X and output to compute MMD only on generated path
                X = X[:,hist_len:,:]
                output = output[:,hist_len:,:]
            else:
                output = generator(noise, t)
                output = torch.cat([t, output], axis=-1) # concatenate time and generated path along time series value dimension

            # compute loss
            optimizer.zero_grad()
            loss = compute_mmd_loss(kernel, X, output, lead_lag, lags)
            losses.append(loss.item())
            # backpropagate and update weights
            loss.backward()
            optimizer.step()

        # log epoch loss and plot generated samples
        epoch_loss = np.average(losses) # average batch mmd for epoch
        last_k_losses.append(epoch_loss)
        avg_k_losses = np.average(last_k_losses)
        scheduler.step(avg_k_losses)
        writer.add_scalar('Loss/Epoch', epoch_loss, epoch)
        writer.add_scalar('Loss/Epoch_avg_k', avg_k_losses, epoch)
        writer.add_scalar('Param/LR', optimizer.param_groups[0]['lr'], epoch)
        print(f'Epoch {epoch}, loss: {epoch_loss}, avg_last_{num_losses}_loss: {avg_k_losses}')

        # save model if avg_k_losses is the best loss so far
        if avg_k_losses < best_loss[0]:
            best_loss = [avg_k_losses, epoch]
            print(f'Saving model at epoch {epoch}')
            torch.save(generator.state_dict(), f'./{writer.log_dir}/best_model.pt')
        elif epoch - best_loss[1] >= early_stopping:
            print(f'Early stopping at epoch {epoch}')
            break

        # save weights and random states for continued training
        torch.save({'epoch': epoch,
                    'rng_state': rng.bit_generator.state,
                    'generator_state_dict': generator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'torch_rng_state': torch.get_rng_state(),
                    'best_loss': best_loss,
                    'last_k_losses': last_k_losses,
                    'dataset_rng_state': dataloader.dataset.rs.get_state() if ma else None
                    }, f'./{writer.log_dir}/checkpoint.pt')

    torch.save(generator.state_dict(), f'./{writer.log_dir}/generator.pt')
    writer.flush()

def get_trained_generator(model_params, data_params, path, filename='generator.pt', device=None):
    generator = get_generator(**{**model_params, **data_params})
    dict = torch.load(path + filename) if device is None else torch.load(path + filename, map_location=device)
    generator.load_state_dict(dict)
    return generator

def get_rl_env(generator, env_params, seed=None):
    rate = env_params['interest_rate']
    transaction_cost = env_params['transaction_cost']
    cal_name = env_params['trading_calendar']
    window_len = env_params['window_len']
    hist_len = env_params['hist_len']
    n_periods = env_params['n_periods']
    max_long = env_params['max_long']
    max_short = env_params['max_short']

    env = ksig_mmd_sim(n_actions=1,
                       window_len=window_len,
                       n_periods=n_periods,
                       max_long=max_long,
                       max_short=max_short,
                       generator=generator,
                       trading_calendar=cal_name,
                       hist_len=hist_len,
                       r=rate,
                       transaction_cost=transaction_cost,
                       seed=seed)
    return env

def get_spx_data_env(path, env_params):
    rate = env_params['interest_rate']
    transaction_cost = env_params['transaction_cost']
    stride = env_params['stride']
    window_len = env_params['window_len']
    n_periods = env_params['n_periods']
    max_long = env_params['max_long']
    max_short = env_params['max_short']

    env = ksig_mmd_sim(n_actions=1,
                       window_len=window_len,
                       n_periods=n_periods,
                       max_long=max_long,
                       max_short=max_short,
                       df_path=path,
                       stride=stride,
                       r=rate,
                       transaction_cost=transaction_cost)
    return env