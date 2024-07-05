import torch
import torch.nn as nn
class GenBase(nn.Module):
    '''
    This class is effectively redudant but had an influence on the random initialization of the weights in the GenLSTM class.
    Therefore, for reproducibility, it is kept here.
    '''
    def __init__(self, noise_dim, seq_dim, seq_len, hidden_size, n_lstm_layers, activation):
        super().__init__()
        self.seq_dim = seq_dim # dimension of the time series e.g. how many stocks
        self.noise_dim = noise_dim # dimension of the noise vector -> vector of (noise_dim, 1) concatenated with the seq value of dimension seq_dim at each time step
        self.seq_len = seq_len # length of the time series which includes the historical data i.e. generated sequence length = seq_len - hist_len
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers

        activation = getattr(nn, activation)
        self.rnn = nn.LSTM(input_size=seq_dim+noise_dim, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        self.mean_net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                     activation(),
                                     nn.Linear(hidden_size, hidden_size),
                                     activation(),
                                     nn.Linear(hidden_size, seq_dim))
        self.var_net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                     activation(),
                                     nn.Linear(hidden_size, hidden_size),
                                     activation(),
                                     nn.Linear(hidden_size, seq_dim))
class GenLSTM(GenBase):
    '''
    LSTM-based generator model for generating sequences.

    Args:
        noise_dim (int): Dimension of the noise vector.
        seq_dim (int): Dimension of the time series (e.g., number of stocks).
        seq_len (int): Length of the time series including the historical portion (if any).
        hidden_size (int, optional): Size of the hidden state of the LSTM. Defaults to 64.
        n_lstm_layers (int, optional): Number of LSTM layers. Defaults to 1.
        activation (str, optional): Activation function for the LSTM. Defaults to 'Tanh'.

    Methods:
        _condition_lstm: Condition the LSTM with historical data and noise.
        _generate_sequence: Generate the sequence using the LSTM.
        forward: Forward pass of the generator model.
    '''

    def __init__(self, noise_dim, seq_dim, seq_len, hidden_size=64, n_lstm_layers=1, activation='Tanh'):
        super().__init__(noise_dim, seq_dim, seq_len, hidden_size, n_lstm_layers, activation)
        self.seq_dim = seq_dim
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers

        activation = getattr(nn, activation)
        self.rnn = nn.LSTM(input_size=seq_dim+noise_dim+1, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        self.output_net = nn.Linear(hidden_size, seq_dim)

    def _condition_lstm(self, noise, hist_x, t):
        if hist_x is not None: assert hist_x.shape[1] > 1, 'Historical data sequence must have length greater than 1'
        batch_size = noise.shape[0]
        h = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)
        c = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)
        seq = torch.zeros(batch_size, 1, self.seq_dim, requires_grad=False, device=noise.device)

        dts = t.diff(dim=1)
        if hist_x is not None:
            diff_x = hist_x.diff(dim=1)
            input = torch.cat([diff_x, noise[:, :diff_x.shape[1], :], dts[:, :diff_x.shape[1], :]], dim=-1)
            output, (h, c) = self.rnn(input, (h, c))
            noise = noise[:,diff_x.shape[1]:,:]
            dts = dts[:,diff_x.shape[1]:,:]
            seq = torch.cat([seq, diff_x], dim=1)
        else:
            diff_x = torch.zeros(batch_size, 1, self.seq_dim, device=noise.device, requires_grad=False)
            input = torch.cat([diff_x, noise[:, :1, :], dts[:, :1, :]], dim=-1)
            output, (h, c) = self.rnn(input, (h, c))
            noise = noise[:,1:,:]
            dts = dts[:,1:,:]
        return seq, output[:,-1:,:], noise, dts, h, c

    def _generate_sequence(self, seq, output, noise, dts, h, c):
        gen_seq = []
        for i in range(self.seq_len-seq.shape[1]):
            x = self.output_net(output)
            gen_seq.append(x)
            if i < noise.shape[1]:
                input = torch.cat([x, noise[:,i:i+1,:], dts[:,i:i+1,:]], dim=-1)
                output, (h, c) = self.rnn(input, (h, c))
        output_seq = torch.cat(gen_seq, dim=1)
        output_seq = torch.cat([seq, output_seq], dim=1)
        return torch.cumsum(output_seq, dim=1)

    def forward(self, noise, t, hist_x=None):
        seq, output, noise, dts, h, c = self._condition_lstm(noise, hist_x, t)
        return self._generate_sequence(seq, output, noise, dts, h, c)
