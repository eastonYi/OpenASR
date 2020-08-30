import logging
import torch
from torch import nn
import random

from loss import cal_ce_square_loss

inf = 1e10

class CPC_Model(nn.Module):
    def __init__(self, encoder, rnn, mappings, cpc_config):
        """
        timestep: predict froward step
        """
        super().__init__()
        self.encoder = encoder
        self.rnn = rnn
        self.mappings = mappings
        self.cpc_config = cpc_config

        # initialize gru
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.rnn.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(self._weights_init)

    def forward(self, x, len_x, hidden):
        device = x.device
        batch, len_x_max = x.size()
        timestep = len(self.mappings)

        # input sequence is N*C*L, e.g. 8*1*20480
        z, len_z = self.encoder(x, len_x)
        t_samples = random.randint(1, (len_z.min().tolist() - timestep - 1))
        # t_samples = torch.randint(len_z.min() - timestep - 1, size=(1,)).long() # randomly pick time stamps
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512

        encode_samples = torch.empty((timestep, batch, 512)).float().to(device) # e.g. size 12*8*512
        for i in range(timestep):
            encode_samples[i] = torch.softmax(z[:, t_samples+i+1, :].view(batch, 512), -1) # z_tk e.g. size 8*512
        forward_seq = z[:, :t_samples+1, :] # e.g. size 8*100*512
        output, hidden = self.rnn(forward_seq, hidden) # output size e.g. 8*100*256
        c_t = output[:, t_samples, :].view(batch, 256) # c_t e.g. size 8*256
        pred = torch.empty((timestep, batch, 512)).float().to(device) # e.g. size 12*8*512
        for i in range(timestep):
            linear = self.mappings[i]
            pred[i] = torch.softmax(linear(c_t), -1) # Wk*c_t e.g. size 8*512

        # target_square = torch.diag(torch.ones(batch)).repeat(timestep, 1, 1).to(device)
        prob_square = torch.bmm(encode_samples, torch.transpose(pred, 1, 2)) # e.g. size 8*8

        loss = 0
        n_correct = 0
        pos_target = torch.ones(timestep).to(device)
        for i in range(batch):
            tmp = list(range(batch)); tmp.pop(i); neg = random.choice(tmp)
            loss += (pos_target - prob_square[:, i, i]).sum()
            loss += prob_square[:, i, neg].sum()
            n_correct += ((prob_square[:, i, i] > 0.5).float().sum() + (prob_square[:, i, neg] < 0.5).float().sum())

        acc = n_correct / (batch * timestep * 2)

        return acc, loss, hidden

    def init_hidden(self, batch_size, device):

        return torch.zeros(self.cpc_config['n_layers'],
                           batch_size,
                           self.cpc_config['d_coding']).to(device)

    def predict(self, x, hidden):
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        output, hidden = self.gru(z, hidden) # output size e.g. 8*128*256

        return output, hidden # return every frame
        #return output[:,-1,:], hidden # only return the last frame per utt

    @classmethod
    def create_model(cls, sp_config, cpc_config):

        from blocks.sp_layers import WavConv

        encoder = WavConv(sp_config)

        d_input = cpc_config['d_input']
        d_coding = cpc_config['d_coding']
        n_layers = cpc_config['n_layers']
        n_steps = cpc_config['n_steps']

        gru = nn.GRU(d_input, d_coding, num_layers=n_layers,
                     bidirectional=False, batch_first=True)
        Wk  = nn.ModuleList([nn.Linear(d_coding, d_input) for i in range(n_steps)])

        model = cls(encoder, gru, Wk, cpc_config)

        return model

    def package(self):
        pkg = {
            "encoder_config": self.encoder.config,
            "encoder_state": self.encoder.state_dict(),
            "cpc_config": self.cpc_config,
            "rnn_state": self.rnn.state_dict(),
            "mappings_state": self.mappings.state_dict()
             }
        return pkg

    def restore(self, pkg):
        # check config
        logging.info("Restore model states...")
        for key in self.encoder.config.keys():
            if (key != "dropout_rate" and
                self.encoder.config[key] != pkg["encoder_config"][key]):
                raise ValueError("encoder_config mismatch.")
        for key in self.cpc.config.keys():
            if (key != "dropout_rate" and
                    self.encoder.config[key] != pkg["encoder_config"][key]):
                raise ValueError("encoder_config mismatch.")

        self.encoder.load_state_dict(pkg["encoder_state"])
        self.rnn.load_state_dict(pkg["rnn_state"])
        self.mappings.load_state_dict(pkg["mappings_state"])

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
