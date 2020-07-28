import torch.nn as nn
import torch

from conv_layers import Conv1d
from utils import sequence_mask


class Attention_Assigner(nn.Module):
    """atteniton assigner of CIF including self-attention and feed forward.
    """
    def __init__(self, config):
        super().__init__()
        # parameters
        self.config = config
        self.d_input = config['d_model']
        self.d_model = config['d_model']
        self.n_layers = config['n_layers']
        self.w_context = config['w_context']
        self.dropout_rate = config['dropout']

        self.conv = Conv1d(self.d_input, self.d_model, self.n_layers, self.w_context,
                           pad='same', name='assigner')
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.linear = nn.Linear(self.d_model, 1)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns:
            enc_output: N x T x H
        """
        x, input_lengths = self.conv(padded_input, input_lengths)
        x = self.dropout(x)
        alphas = self.linear(x).squeeze(-1)
        alphas = torch.sigmoid(alphas)
        pad_mask = sequence_mask(input_lengths)

        return alphas * pad_mask
