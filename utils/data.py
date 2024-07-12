import os
import json
from typing import Optional
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import arch
from arch.univariate import Normal
from .gaussianize import Gaussianize

# Constants for the MA model using arch package
MEAN_MODEL = 'Zero'
Q_CONST = 0
DATATYPE = torch.float32

class MADataset(Dataset):
    '''
    Dataset generating normalised log price series from a dataframe of historical price data
    Each sample is of shape (sample_len, seq_dim+1) where the first column is the time dimension
    Option to generate noise using MA model fitted to the Gaussianized annualised and normalised log returns calculated from the historical data
    When generating noise, the p lags of the processed returns are used to initialise the MA model
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing historical data with a datetime index
    start_date: str
        Start date of data samples to output
    end_date: str
        End date of data samples to output
    sample_len: int
        Length of each sample
    MA: bool
        Whether to use MA model to generate noise (False means no noise is returned)
    p: int
        Number of lags for MA model for squared residuals
    noise_dim: int
        Number of noise dimensions to generate
    stride: int
        Stride for sampling data
    seed: int
        Random seed for MA model
    col_idx: int
        Column index of the data to use
    '''
    def __init__(self, df: pd.DataFrame, start_date: str, end_date: str, sample_len: int,
                 MA: bool, noise_dim: Optional[int]=None, p: Optional[int]=None,
                 stride: int=1, seed: Optional[int]=None, col_idx: int=0):

        self.sample_len = sample_len
        self.stride = stride
        self.seq_dim = 1
        self.MA = MA
        if self.MA:
            assert p is not None, 'p must be specified for MA model'
            assert noise_dim is not None, 'noise_dim must be specified for MA model'
            self.noise_dim = noise_dim
            self.p = p
        self.dtype = DATATYPE

        # Gaussianize and fit data for MA model
        if MA: # add conditional after dropping rows for reproducibility when not using MA
            self.df, _ = gaussianize_data(df, col_idx) # self.df used to set up MA model ending at the start of the item
            ma_model = arch.arch_model(df.loc[:,'gaussianized'], mean=MEAN_MODEL, p=self.p, q=Q_CONST, rescale=True)
            self.res = ma_model.fit(update_freq=0)
            print(self.res.summary())
            self.rs = np.random.RandomState(seed)

        df = df.loc[start_date:end_date].copy()
        df.index = pd.to_datetime(df.index)
        t = torch.zeros(len(df), 1, dtype=self.dtype, requires_grad=False)
        time_series_df = df if col_idx is None else df.iloc[:, col_idx]
        time_series = time_series_df.values.reshape(len(df), -1)
        time_series = np.log(time_series)
        # calculate time dimension as years starting from 0 (365 days per year)
        # first value of t is 0, subsequent values are the difference in days divided by 365
        # first value from index.diff() is NaN, so we start from the second value which is the difference between the first and second index
        t[1:,0] = torch.tensor((df.index.to_series().diff()[1:].dt.days / 365).values.cumsum(), dtype=self.dtype, requires_grad=False)
        self.dataset = torch.cat([t, torch.tensor(time_series, dtype=self.dtype, requires_grad=False)], dim=-1)
        self.shape = self.dataset.shape
        self.len = (int((self.dataset.shape[0] - self.p - self.sample_len)/self.stride) + 1)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        start = idx*self.stride + self.p
        end = start + self.sample_len
        path = self.dataset[start:end] # shape (sample_len, seq_dim+1)
        item = torch.empty_like(path, requires_grad=False)
        # rebase series to start at 0 including time dimension and lead-lag series
        item = path - path[0:1] # log X(t) / X(0) = log X(t) - log X(0)

        if self.MA:
            # MA model inputs end before the start of the item
            ma_model = arch.arch_model(self.df.iloc[start-self.p:start, 0], mean=MEAN_MODEL, p=self.p, q=Q_CONST, rescale=False)
            # set distribution of MA model with random state passed in the construction for reproducibility
            # NOTE: each time forecast is called, the random state will change
            ma_model.distribution = Normal(seed=self.rs)
            self.ma_model = ma_model
            # forecast sample_len-1 steps
            forecasts = ma_model.forecast(params=self.res.params, horizon=self.sample_len-1, method='simulation', simulations=self.noise_dim)
            noise = forecasts.simulations.residuals[0].T
            noise = torch.tensor(noise, dtype=path.dtype, requires_grad=False)
            return item, noise
        else:
            return item

class MA_path_generator():
    '''
    Generate paths based on the following steps:
    1. Create MA model from real data by first Gaussianizing the annualised and normalised log returns with the Lambert transform
    2. Generate MA residuals using MA model build from real data
    3. Feed MA residuals and historical path into generator to produce synthetic paths
    Parameters
    ----------
    generator: nn.Module
        Generator model
    df: pd.DataFrame
        Dataframe containing historical data with a datetime index
    start_date: str
        Start date of data used to train MA model
    end_date: str
        End date of data used to train MA model
    p: int
        Number of lags for MA model for squared residuals
    seed: int
        Random seed for MA model
    random_state: np.random.RandomState
        Random state for MA model
    col_idx: int
        Column index of the data to use
    '''

    def __init__(self, generator: nn.Module, df: pd.DataFrame, start_date: str, end_date: str,
                 p: int, seed: Optional[int]=None, random_state: Optional[np.random.RandomState]=None, col_idx: int=0):

        # assert that not both seed and random_state are passed
        assert (seed is None) or (random_state is None), 'only one of seed or random_state can be passed'
        self.rs = np.random.RandomState(seed) if random_state is None else random_state
        self.dtype = DATATYPE
        self.p = p
        self.noise_dim = generator.noise_dim
        df_gen = df.copy() # df_gen used to generate paths which preserves the original data date range
        df = df.loc[start_date:end_date].copy() # MA model trained on data from start_date to end_date

        # Gaussianize data and train MA model
        df, lambert_transform = gaussianize_data(df, col_idx)
        ma_model = arch.arch_model(df.loc[:,'gaussianized'], mean=MEAN_MODEL, p=self.p, q=Q_CONST, rescale=True)
        self.res = ma_model.fit(update_freq=0)
        print(self.res.summary())

        # prepare data for generator
        self.generator = generator
        df_gen, _ = gaussianize_data(df_gen, col_idx, lambert_transform)
        df_gen['log_path'] = np.log(df_gen.iloc[:, col_idx]) - np.log(df_gen.iloc[:, col_idx].iloc[0])
        df_gen['t'] = np.cumsum(df_gen['dt'])
        df_gen['t'] = df_gen['t'] - df_gen['t'].iloc[0]
        self.df_gen = df_gen

    def generate(self, start_date: str, end_date: str, trading_calendar: str, hist_len: int,
                 batch_size: int, n_batches: int=1,
                 device: str='cpu'):
        '''
        Generate synthetic paths using MA model and generator
        The historical portion of the synthetic paths will be drawn from df_gen i.e. real data
        This means all synthetic paths will have the same historical portion
        Parameters
        ----------
        start_date: str
            Start date of synthetic paths
            MA model will be conditioned based on data up to this date
        end_date: str
            End date of synthetic paths
        trading_calendar: str
            Trading calendar to use for generating time dimension
        hist_len: int
            Length of historical portion drawn from df_gen
        batch_size: int
            Number of paths to generate
        n_batches: int
            Number of batches to generate
        '''
        # set up timeline
        if pd.to_datetime(end_date) > self.df_gen.index[-1]:
            calendar = mcal.get_calendar(trading_calendar)
            schedule = calendar.schedule(start_date=start_date, end_date=end_date)
            t = np.zeros(len(schedule))
            t[1:] = (schedule.index.to_series().diff()[1:].dt.days / 365).values.cumsum()
        else:
            t = self.df_gen[start_date:end_date]['t'].values
        t = t - t[0]
        timeline_whist = t.copy()
        timeline_wohist = timeline_whist[hist_len:]
        timeline_wohist = timeline_wohist - timeline_wohist[0]

        # adjust generator sequence length parameter if necessary
        self.sample_len = len(t)
        if self.generator.seq_len != self.sample_len:
            self.generator.seq_len = self.sample_len

        # prepare time tensor and historical data tensor for generator
        t = torch.tensor(t, dtype=self.dtype, device=device, requires_grad=False)
        t = t.repeat(batch_size, 1).unsqueeze(-1) # shape (batch_size, sample_len, seq_dim)
        if hist_len > 0: # use historical data which will be part of the synthetic paths and used to condition the generator
            hist_x = self.df_gen[start_date:end_date]['log_path'].values
            if len(hist_x) < hist_len:
                raise ValueError('hist_len is longer than the available historical data.')
            hist_x = hist_x[:hist_len]
            hist_x = hist_x - hist_x[0] # rebase historical data to start at 0
            hist_x = torch.tensor(hist_x, dtype=self.dtype, device=device, requires_grad=False)
            hist_x = hist_x.repeat(batch_size, 1).unsqueeze(-1) # shape (batch_size, hist_len, seq_dim)
        else: # no historical data to be used
            hist_x = None

        if len(self.df_gen.loc[:start_date, 'gaussianized']) < self.p:
            raise ValueError('MA initialization period is longer than the available historical data.')

        # create MA model with data up to start_date but model parameters will be from self.res.params trained during init
        ma_model = arch.arch_model(self.df_gen.loc[:start_date, 'gaussianized'], mean=MEAN_MODEL, p=self.p, q=Q_CONST, rescale=False)

        # set distribution of MA model with random state passed in the construction for reproducibility
        ma_model.distribution = Normal(seed=self.rs)

        list_path_whist = []
        list_path_wohist = []
        list_log_returns = []

        # for _ in tqdm(range(n_batches)):
        for _ in range(n_batches):
            forecasts = ma_model.forecast(params=self.res.params, horizon=self.sample_len-1, method='simulation', simulations=self.noise_dim*batch_size)
            noise = forecasts.simulations.residuals[0].T
            noise = noise.reshape(noise.shape[0], self.noise_dim, batch_size).transpose(2, 0, 1) # shape (batch_size, sample_len-1, noise_dim)
            noise = torch.tensor(noise, dtype=self.dtype, device=device, requires_grad=False)

            output_whist = self.generator(noise, t, hist_x).detach().cpu().squeeze()
            if output_whist.ndim == 1:
                output_whist = output_whist.unsqueeze(0)
            path_whist = torch.exp(output_whist)
            output_wohist = output_whist[:, hist_len:]
            output_wohist = output_wohist - output_wohist[:,:1]
            path_wohist = torch.exp(output_wohist)
            log_returns = torch.diff(output_wohist, axis=1)
            list_path_whist.append(path_whist)
            list_path_wohist.append(path_wohist)
            list_log_returns.append(log_returns)

        path_whist = torch.cat(list_path_whist, axis=0).clone().cpu()
        path_wohist = torch.cat(list_path_wohist, axis=0).clone().cpu()
        log_returns = torch.cat(list_log_returns, axis=0).clone().cpu()

        return path_whist, path_wohist, log_returns, timeline_whist, timeline_wohist

def gaussianize_data(df: pd.DataFrame, col_idx: int, lambert_transform=None):
    df['log_returns'] = np.log(df.iloc[:, col_idx]).diff()
    df['dt'] = df.index.to_series().diff().dt.days / 365
    df['cal_ann_returns'] = df.loc[:, 'log_returns'] / df.loc[:, 'dt']
    df['norm_cal_ann_returns'] = (df.loc[:, 'cal_ann_returns'] - df.loc[:, 'cal_ann_returns'].mean()) / df.loc[:, 'cal_ann_returns'].std()
    df.dropna(inplace=True)
    if lambert_transform is None:
        lambert_transform = Gaussianize()
        lambert_transform = lambert_transform.fit(df.loc[:, 'norm_cal_ann_returns'])
    df['gaussianized'] = lambert_transform.transform(df.loc[:, 'norm_cal_ann_returns'])
    return df.copy(), lambert_transform

def get_time_vector(df, difference=False):
    '''
    Get time vector from dataframe index
    Time vector is the number of calendar years from the first date in the index with the first value being 0
    If difference is True, the time vector is the difference in calendar years between consecutive dates
    '''
    t = np.zeros(len(df), dtype=np.float32)
    t[1:] = (df.index.to_series().diff()[1:].dt.days / 365).values.cumsum()
    if difference:
        t = np.diff(t)
    return t

def batch_lead_lag_transform(data: torch.Tensor, t:torch.Tensor, lead_lag: int|list[int]=1):
    '''
    Transform data to lead-lag format
    data is of shape (seq_len, seq_dim)
    '''
    assert data.ndim == 3 and t.ndim == 3, 'data and t must be of shape (batch_size, seq_len, seq_dim)'
    assert data.shape[1] == t.shape[1], 'data and df_index must have the same length'
    if isinstance(lead_lag, int):
        if lead_lag <= 0: raise ValueError('lead_lag must be a positive integer')
    else:
        for lag in lead_lag:
            if lag <= 0: raise ValueError('lead_lag must be a positive integer')

    # get shape of output
    batch_size = data.shape[0]
    seq_len = data.shape[1]
    seq_dim = data.shape[2]
    shape = list(data.shape)
    if isinstance(lead_lag, int):
        lead_lag = [lead_lag]
    max_lag = max(lead_lag)
    shape[1] = shape[1] + max_lag
    shape[2] = (len(lead_lag) + 1) * seq_dim

    # create time dimension t.shape = (batch_size, seq_len, 1)
    # pad latter values with last value, shape (seq_len + max_lag, 1)
    t = torch.cat([t, (torch.ones(batch_size, max_lag, 1, dtype=t.dtype, device=t.device, requires_grad=False) * t[:,-1:,:])], dim=1)

    # create lead-lag series
    lead_lag_data = torch.empty(shape, dtype=data.dtype, device=t.device, requires_grad=False) # shape (seq_len + max_lag, seq_dim * (len(lead_lag) + 1))
    lead_lag_data[:, :seq_len, :seq_dim] = data # fill in original sequence
    lead_lag_data[:, seq_len:, :seq_dim] = data[:,-1:,:] # pad latter values with last value
    for i, lag in enumerate(lead_lag):
        i = i + 1 # skip first seq_dim columns
        lead_lag_data[:, :lag, i*seq_dim:(i+1)*seq_dim] = 0.0 # pad initial values with zeros
        lead_lag_data[:, lag:lag+seq_len, i*seq_dim:(i+1)*seq_dim] = data
        lead_lag_data[:, lag+seq_len-1:, i*seq_dim:(i+1)*seq_dim] = data[:,-1:,:] # pad latter values with last value
    return torch.cat([t, lead_lag_data], axis=2)

def get_hparams(path):
    hparam_type = {}
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    tags = event_acc.Tags()["tensors"]
    for tag in tags:
        name = tag.split('/')[0]
        event_list = event_acc.Tensors(tag)
        param_str = str(event_list[0].tensor_proto.string_val[0])
        param_str = param_str.replace('\\n', '')
        param_str = param_str.replace('\\t', '')
        param_str = param_str.replace('\'', '')
        param_str = param_str.replace('\\', '')
        param_str = param_str.replace('b{', '{')
        if param_str.startswith('{'):
            params = json.loads(param_str)
            hparam_type[name] = params
    return hparam_type

def get_params_from_events(path):
    events_files = []
    for file in os.listdir(path):
        if file.startswith('events'):
            events_files.append(file)
    events_files.sort()
    events_name = events_files[0]
    params = get_hparams(path + events_name)
    return params

def max_drawdown(df, value_col, drawdown_col_name='max_drawdown', drawdown_dur_col_name='max_drawdown_duration', value=True, log_return=False):
    '''
    Calculates the maximum drawdown of a given value column in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the value column and index should be ordered by date.
    value_col : str
        Name of the column containing the value to calculate the drawdown
    drawdown_col_name : str
        Name of the column to store the drawdown values
    drawdown_dur_col_name : str
        Name of the column to store the drawdown duration values
    value : bool
        If True, the value column is assumed to be a price level else it is assumed to be log returns.
    log_return : bool
        If True, the drawdown is calculated in log returns else it is calculated in absolute returns.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with max_drawdown and max_drawdown_duration columns added.
    '''

    for date in df.index:
        # calculate drawdown to future min point from current date in log returns
        if value:
            min = df.loc[date:, value_col].min()
            curr_px = df.loc[date, value_col]
            if min == curr_px:
                df.loc[date, drawdown_col_name] = np.nan
            else:
                df.loc[date, drawdown_col_name] = np.log(min / curr_px) if log_return else min / curr_px - 1
        else:
            temp_df = df.copy()
            temp_df['cum_returns'] = df[value_col].cumsum()
            min = temp_df.loc[date:, 'cum_returns'].min()
            df.loc[date, drawdown_col_name] = min - df.loc[date, 'cum_returns']

        # find date of future min point
        row = df.loc[date:,:][df.loc[date:, value_col] == min]
        # print(date.strftime('%Y-%m-%d'), curr_px, row.index[0].strftime('%Y-%m-%d'), min)
        if len(row) == 0: max_drawdown_date = np.nan
        elif len(row) == 1: max_drawdown_date = row.index[0]
        else: raise ValueError('Multiple min points found')

        # check that there is a max drawdown date to calculate duration
        if not max_drawdown_date == np.nan and not type(max_drawdown_date) == float:
            df.loc[date, drawdown_dur_col_name] = max_drawdown_date - date
        else:
            df.loc[date, drawdown_dur_col_name] = np.nan

    return df

def perf_table(df: pd.DataFrame, dates: list, periods: list):
    metrics = ['Ann. return', 'Volatility', 'Sharpe ratio', 'Max drawdown']
    table = pd.DataFrame(columns=metrics)

    for i in range(len(dates)):
        start = dates[i][0]
        end = dates[i][1]
        temp_df = df.loc[start:end].copy()
        temp_df = max_drawdown(temp_df, 'agent', value=True, log_return=True)
        label = periods[i]
        log_returns = np.diff(np.log(temp_df.loc[:, 'agent'].values.squeeze()))
        table.loc[label, 'Ann. return'] = np.log(temp_df.loc[end, 'agent'] / temp_df.loc[start, 'agent']) * 252 / len(temp_df)
        table.loc[label, 'Volatility'] = np.std(log_returns) * np.sqrt(252)
        table.loc[label, 'Sharpe ratio'] = log_returns.mean() * 252 / table.loc[label, 'Volatility']
        table.loc[label, 'Max drawdown'] = temp_df.loc[:, 'max_drawdown'].min()
    return table