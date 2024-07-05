import os
from typing import Optional, List

import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from .data import MA_path_generator, get_time_vector

TORCH_DTYPE = torch.float32
NP_DTYPE = np.float32

def get_latest_run():
    highest = 0
    for entry in os.scandir('./runs'):
        if entry.is_dir():
            if entry.name.startswith('PPO'):
                number = int(entry.name.split('_')[-1])
                if number > highest:
                    highest = number
    return highest

class pm_env():
    '''
    Portfolio Management Environment
    Parameters:
    n_actions: int
        number of risky assets
    window_len: int
        length of window for observation
    n_periods: int
        number of periods where action can be taken
    baseline_weights: numpy array
        weights for baseline portfolio
    verbose: bool
        whether to print out episode results
    df_path: str
        path to historical data csv file
    stride: int
        number of periods to skip after each episode for historial data
    generator: MA_path_generator
        generator for synthetic data
    hist_len: int
        length of history for each generated path which is extracted from historical data and used at the start of each generated path
    trading_calendar: str
        trading calendar for generating synthetic data which is used if the gen_end_date is beyond the last date in the historical data
    r: float
        risk free rate which is a constant
    seed: int
        random seed
    device: torch.device
        device for torch tensors
    '''

    # Define constants for clearer code
    INITIAL_WEALTH = 1.
    MIN_POS = -1.
    MAX_POS = 1.

    def __init__(self, n_actions: int, window_len: int, n_periods: int,
                 baseline_weights: Optional[np.ndarray|List]=None,
                 verbose: bool=True,
                 df_path: Optional[str]=None, stride: Optional[int]=None,
                 generator: MA_path_generator=None, hist_len: Optional[int]=None, trading_calendar: Optional[str]=None,
                 r: Optional[float]=0., transaction_cost: Optional[float]=0.,
                 seed: Optional[int]=None,
                 device: Optional[torch.device]=torch.device('cpu')):

        assert not (generator is None and df_path is None), 'Must have either generator or df_path'
        assert not (generator is not None and df_path is not None), 'Cannot have both generator and df'

        self.r = r
        self.transaction_cost = transaction_cost
        self.n_actions = n_actions
        self.n_assets = self.n_actions + 1
        if baseline_weights is None:
            self.baseline_weights = np.ones((self.n_assets)) / self.n_actions
            self.baseline_weights[0] = 0.
        else:
            self.baseline_weights = np.array(baseline_weights)
            assert self.baseline_weights.shape == (self.n_assets,), 'Baseline weights must have same shape as number of assets'
            assert np.isclose(self.baseline_weights.sum(), 1.), 'Baseline weights must sum to one'
        self.rng = np.random.default_rng(seed=seed)
        self.verbose = verbose

        self.window_len = window_len # length of window for observation
        self.n_periods = n_periods # number of periods where action can be taken
        self.episode_counter = 0 # keep track of number of episodes
        self.steps_counter = 0 # keep track of number of steps

        # set up for synthetic data
        self.generator = generator
        if generator is not None:
            self.hist_len = hist_len
            self.trading_calendar = trading_calendar
            self.device = device
            self.dtype = TORCH_DTYPE
            self.gen_dates = self.generator.df_gen.index[self.generator.p+1:].to_series() # must leave sufficient data for GARCH(p,q) to generate noise
            self.generate_path()

        # set up for historical data
        if df_path is not None:
            self.start_period = 0
            self.df = pd.read_csv(df_path, index_col=0, parse_dates=True)
            self.time = get_time_vector(self.df)
            self.dts = np.diff(self.time)
            self.data = self.df.values # (len(self.df), n_actions)
            self.stride = stride
        else:
            self.data = None

    def pm_env_reset(self, reset_periods):
        '''
        Reset the environment for a new episode
        '''
        # use synthetic data
        if self.generator is not None:
            self.episode_path = self.batch
            self.batch_period = 0 # keep track of time index for batch
            self.S = self.batch[self.batch_period, :]
            self.B = 1.
            self.curr_step = np.array([self.B] + list(self.S))
            self.dt = self.dts[self.batch_period] # first value of self.dts is for batch_period 0 to 1
            self.generate_path()

        # use historical data
        elif self.data is not None:
            self.S = np.ones((self.n_actions), dtype=NP_DTYPE)
            self.B = 1.
            self.curr_step = np.append(self.B, self.S)
            self.episode_path = [self.S]
            periods_needed = self.window_len + self.n_periods # periods needed for episode is window_len + n_periods NOTE: reset_periods will be window_len - 1
            self.episode_data = self.data[self.start_period:self.start_period + periods_needed] # get data for episode starting from start_period

            # check if sufficient number of periods in episode
            if self.episode_data.shape[0] < periods_needed:
                print(f'Not enough periods in real data for episode. Need {periods_needed} but only have {self.episode_data.shape[0]}')
                self.episode_path = np.empty((periods_needed, self.n_actions))
                self.dt = None
                self.position = [None]
                return

            self.episode_data = self.episode_data / self.episode_data[0, :] # normalise to start at 1
            self.real_data_current_period = 0 # keep track of time index for episode
            self.episode_dts = self.dts[self.start_period:self.start_period + periods_needed - 1] # dts needed for episode is window_len + n_periods - 1
            self.dt = self.episode_dts[self.real_data_current_period] # first value of self.dts is for batch_period 0 to 1
            self.start_period += self.stride # increment start period for next episode

        # set up for episode
        self.episode_wealth = [self.INITIAL_WEALTH]
        self.episode_baseline_wealth = [self.INITIAL_WEALTH]
        self.episode_weights = []
        self.baseline_wealth = self.INITIAL_WEALTH
        self.agent_wealth = self.INITIAL_WEALTH
        self.baseline_bankrupt = False

        # keep track of periods simulated after reset simulation note that simulate_one_step does not increment this
        self.current_period = 0 # equals to number of actions taken
        for t in range(reset_periods): self.simulate_one_step()
        self.position = np.zeros((self.n_actions,), dtype=NP_DTYPE) # for obs if used

    def simulate_one_step(self):
        '''
        Simulate the asset prices in the next step
        '''
        # use synthetic data
        if self.generator is not None:
            self.batch_period += 1 # increment batch_period for next time step in batch, first call increments this from 0 to 1
            next_step = self.episode_path[self.batch_period]
            self.S = next_step
            self.B *= np.exp(self.r * self.dt) # dt is from prev batch_period to current batch_period
            if self.current_period < self.n_periods: # if not at terminal time
                self.dt = self.dts[self.batch_period] # calculates dt from current batch_period to next batch_period NOTE: first dt assigned in pm_env_reset
            next_step = np.array([self.B] + list(self.S)) # don't update self.curr_step until it is saved in self.prev_step

        # use historical data
        elif self.data is not None:
            self.real_data_current_period += 1
            self.B = self.B * np.exp(self.r * self.dt)
            self.S = self.episode_data[self.real_data_current_period]
            next_step = np.append(self.B, self.S)
            self.episode_path.append(self.S)
            if self.real_data_current_period < self.n_periods: # if not at terminal time
                self.dt = self.episode_dts[self.real_data_current_period] # calculates dt from current batch_period to next batch_period

        self.prev_step = self.curr_step
        self.curr_step = next_step

    def pm_env_step(self, action):
        '''
        Simulate the next step and use action to determine reward
        '''
        self.current_period += 1 # value is 1 after first action i.e. self.n_periods = num of actions
        self.simulate_one_step() # simulate next step prices AFTER action is taken
        simple_return = self.curr_step / self.prev_step # return factor for all assets i.e. 1 + % change in price

        if self.baseline_bankrupt:
            self.baseline_wealth = 0.
        else:
            baseline_return = (self.baseline_weights * simple_return).sum()
            if baseline_return <= 0:
                self.baseline_bankrupt = True
                self.baseline_wealth = 0.
            else:
                self.baseline_wealth *= baseline_return
        self.episode_baseline_wealth.append(self.baseline_wealth)

        weights = self.process_action(action) # adds weight for cash where total sums to one
        portfolio_return = (weights * simple_return).sum()

        # deduct transaction cost before wealth is updated to new value based on new stock prices
        if self.transaction_cost > 0:
            prev_weights = self.episode_weights[-1] if len(self.episode_weights) > 0 else np.zeros((self.n_assets,))
            cost = self.transaction_cost * np.abs(weights[1:] - prev_weights[1:]).sum() * self.agent_wealth
        else:
            cost = 0.

        if portfolio_return <= 0: # check for bankruptcy
            done = True # if bankrupt then episode is done
            reward = np.log(np.nextafter(0, 1)) # lowest reward possible based on log of smallest positive number
            self.agent_wealth = 0. # agent bankrupt
        else:
            done = False
            new_wealth = self.agent_wealth * portfolio_return - cost
            reward = np.log(new_wealth / self.agent_wealth) # log return of wealth
            self.agent_wealth = new_wealth

        self.episode_wealth.append(self.agent_wealth)
        self.episode_weights.append(weights)

        if self.current_period == self.n_periods: done = True # reached terminal time

        info = {}
        if done: self.pm_env_done()
        self.position = action.astype(np.float32) # for obs if used

        return reward, done, info

    def pm_env_done(self):
        '''
        actions to complete if env is done
        '''
        if self.current_period < self.n_periods: # if agent is bankrupt but not at terminal time yet
            self.simulate_baseline_to_terminal_time()
        weights = np.array(self.episode_weights)

        self.episode_counter += 1
        self.steps_counter += len(self.episode_weights)
        if self.verbose:
            print(f'E{self.episode_counter} / S{self.steps_counter}: ' +
                  f'Baseline Wealth = {self.baseline_wealth:.4f} / Agent Final Wealth = {self.agent_wealth:.4f} / ' +
                  f'Average Weights and Std: {weights.mean(axis=0)} / {np.sqrt(weights.var(axis=0))}')

    def seed(self, seed):
        pass

    def generate_path(self):
        '''
        Generate a new sequence of prices for the risky assets using generator
        Dates are randomly chosen from the historical data
        '''
        # start date is randomly chosen from the first n of the dates where n = len(gen_dates) - hist_len - n_periods
        # end date is chosen so that there are a total of hist_len + n_periods dates
        start_date_idx = self.rng.integers(0, len(self.gen_dates) - self.hist_len - self.n_periods)
        self.gen_start_date = self.gen_dates[start_date_idx].date().isoformat() # 'yyyy-mm-dd'
        self.gen_end_date = self.gen_dates[start_date_idx + self.hist_len + self.n_periods - 1].date().isoformat()
        gen_output = self.generator.generate(self.gen_start_date, self.gen_end_date, self.trading_calendar, self.hist_len, 1, 1, self.device)
        batch, _, _, self.time, _ = gen_output
        if batch.ndim == 2: batch = batch.unsqueeze(-1) # add channel dimension if 1D so that last dimension indicates number of assets
        # self.batch are paths (non-log) of shape (batch_size, sample_len) sample_len = hist_len + n_periods
        self.batch = batch.detach().numpy()[0]
        # self.time are the corresponding time in calendar year starting from 0 of shape (batch_size, sample_len)
        self.dts = np.diff(self.time)

    def process_action(self, action):
        '''
        Input is action which are weights for risky assets that may not sum to one
        Determine the weight for the risk-free assets so that total weight sums to one
        Return weights of all asset where risk-free asset is the first element
        '''
        weights = np.zeros((self.n_assets))
        weights[0] = 1. - action.sum()
        weights[1:] = action
        return weights

    def simulate_baseline_to_terminal_time(self):
        '''
        Simulate the baseline wealth to maturity when agent is already bankrupt hence episode terminated
        '''
        for t in range(self.n_periods - self.current_period):
            self.simulate_one_step()
            if self.baseline_bankrupt:
                self.baseline_wealth = 0.
            else:
                simple_return = self.curr_step / self.prev_step # return factor for all assets i.e. 1 + % change in price
                baseline_return = (self.baseline_weights * simple_return).sum()
                self.baseline_wealth *= baseline_return
            self.episode_baseline_wealth.append(self.baseline_wealth)
            self.episode_wealth.append(0.)
            self.episode_weights.append(np.zeros((self.n_assets)))
            self.current_period += 1

    def plot(self):
        '''
        Plot results and store data in csv file for historical data only
        '''
        self.df.loc[self.df.index[-len(self.episode_baseline_wealth):], 'baseline'] = self.episode_baseline_wealth
        self.df.loc[self.df.index[-len(self.episode_wealth):], 'agent'] = self.episode_wealth
        weights = np.array(self.episode_weights)
        n = weights.shape[0]
        self.df.loc[self.df.index[-n:], 'cash'] = weights[:,0]
        self.df.loc[self.df.index[-n:], 'index'] = weights[:,1]
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        indices = self.df.index[-n:]
        (self.df.loc[indices, ['agent', 'baseline']] / self.df.loc[indices[0], ['agent', 'baseline']].values.squeeze()).plot(ax=ax[0])
        ax[1].bar(height=self.df.loc[indices, 'cash'].values, x=np.arange(n), bottom=self.df.loc[indices, 'index'].values, width=1)
        ax[1].bar(height=self.df.loc[indices, 'index'].values, x=np.arange(n), width=1)
        plt.tight_layout()
        plt.show()

        num = get_latest_run()
        self.df.to_csv(f'./runs/PPO_{num}/data.csv')

class ksig_mmd_sim(gym.Env, pm_env):
    '''
    Portfolio Management Environment
    Reward is based on log return of weights generated from action
    Observation is based on window length of prices, current wealth, current position and time to next period
    '''
    def __init__(self, n_actions: int, window_len: int, n_periods: int,
                 max_long: Optional[float]=None, max_short: Optional[float]=None,
                 baseline_weights: Optional[np.ndarray|List]=None,
                 verbose: bool=True, df_path: Optional[str]=None, stride: Optional[int]=None,
                 generator: MA_path_generator=None, hist_len: Optional[int]=None,
                 trading_calendar: Optional[str]=None,
                 r: Optional[float]=None, transaction_cost: float=0.,
                 seed: Optional[int]=None,
                 device: Optional[torch.device]=torch.device('cpu')):

        super().__init__(n_actions, window_len, n_periods, baseline_weights, verbose,
                         df_path, stride, generator, hist_len, trading_calendar,
                         r, transaction_cost, seed, device)

        max_long = self.MAX_POS if max_long is None else max_long
        max_short = self.MIN_POS if max_short is None else max_short
        self.action_space = spaces.Box(low=max_short, high=max_long, shape=(self.n_actions,), dtype=NP_DTYPE)
        n_obs_terms = self.n_actions * self.window_len + 2 + self.n_actions # 2 for wealth and dt
        low = np.zeros((n_obs_terms,), dtype=NP_DTYPE)
        low[-self.n_actions:] = max_short
        high = np.inf * np.ones((n_obs_terms,), dtype=NP_DTYPE)
        high[-self.n_actions:] = max_long
        self.observation_space = spaces.Box(low=low, high=high, shape=(n_obs_terms,), dtype=NP_DTYPE)

    def reset(self):
        # observation must be a numpy array
        self.pm_env_reset(self.window_len - 1) # initial price of all 1s can form the first set of values in window_len
        return self.generate_obs()

    def step(self, action):
        reward, done, info = self.pm_env_step(action)
        return self.generate_obs(), reward, done, info

    def render(self):
        pass

    def close(self):
        pass

    def generate_obs(self):
        # Feed in prices of all assets based on window length and position'
        if self.data is None:
            window = np.array(self.episode_path[self.batch_period-(self.window_len-1):self.batch_period+1]).reshape(-1).astype(np.float32)
        else:
            window = np.array(self.episode_path[self.real_data_current_period-(self.window_len-1):self.real_data_current_period+1]).reshape(-1).astype(np.float32)
        # window has shape(window_len * n_actions,)

        obs = np.concatenate((window, [self.dt], [self.agent_wealth], self.position), axis=-1)
        return obs
