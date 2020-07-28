from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm


class Conv1d(nn.Module):
    # the same as stack frames
    def __init__(self, d_input, d_hidden, n_layers, w_context, pad='same', name=''):
        super().__init__()
        assert n_layers >= 1
        self.n_layers = n_layers
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.w_context = w_context
        self.pad = pad
        self.name = name

        layers = [("{}/conv1d_0".format(name), nn.Conv1d(d_input, d_hidden, w_context, 1)),
                  ("{}/relu_0".format(name), nn.ReLU())]
        for i in range(n_layers-1):
            layers += [
                ("{}/conv1d_{}".format(name, i+1), nn.Conv1d(d_hidden, d_hidden, w_context, 1)),
                ("{}/relu_{}".format(name, i+1), nn.ReLU())
            ]
        layers = OrderedDict(layers)
        self.conv = nn.Sequential(layers)

    def forward(self, feats, feat_lengths):
        if self.pad == 'same':
            input_length = feats.size(1)
            feats = F.pad(feats, (0, 0, 0, self.n_layers * self.w_context))
        outputs = self.conv(feats.permute(0, 2, 1))
        outputs = outputs.permute(0, 2, 1)

        if self.pad == 'same':
            tensor_length = input_length
            assert tensor_length <= outputs.size(1)
            outputs = outputs[:, :tensor_length, :]
            output_lengths = feat_lengths
        else:
            output_lengths = ((feat_lengths + sum(self.padding) -
                               1*(self.w_context-1)-1)/self.subsample + 1).long()

        return outputs, output_lengths


class Conv1dSubsample(torch.nn.Module):
    # the same as stack frames
    def __init__(self, d_input, d_model, w_context, subsample):
        super().__init__()

        self.conv = nn.Conv1d(d_input, d_model, w_context, stride=self.subsample)
        self.conv_norm = LayerNorm(self.d_model)
        self.subsample = subsample
        self.w_context = w_context

    def forward(self, feats, feat_lengths):
        outputs = self.conv(feats.permute(0, 2, 1))
        outputs = outputs.permute(0, 2, 1)
        outputs = self.conv_norm(outputs)
        output_lengths = ((feat_lengths - 1*(self.w_context-1)-1)/self.subsample + 1).long()

        return outputs, output_lengths


class Conv2dSubsample(torch.nn.Module):
    # Follow ESPNet configuration
    def __init__(self, d_input, d_model):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, 2),
            torch.nn.ReLU()
        )
        self.affine = torch.nn.Linear(32 * (((d_input - 1) // 2 - 1) // 2), d_model)

    def forward(self, feats, feat_lengths):
        outputs = feats.unsqueeze(1)  # [B, C, T, D]
        outputs = self.conv(outputs)
        B, C, T, D = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(B, T, C*D)
        outputs = self.affine(outputs)
        output_lengths = (((feat_lengths-1) / 2 - 1) / 2).long()

        return outputs, output_lengths


class Conv2dSubsampleV2(torch.nn.Module):
    def __init__(self, d_input, d_model, layer_num=2):
        super().__init__()
        assert layer_num >= 1
        self.layer_num = layer_num
        layers = [("subsample/conv0", torch.nn.Conv2d(1, 32, 3, (2, 1))),
                ("subsample/relu0", torch.nn.ReLU())]
        for i in range(layer_num-1):
            layers += [
                ("subsample/conv{}".format(i+1), torch.nn.Conv2d(32, 32, 3, (2, 1))),
                ("subsample/relu{}".format(i+1), torch.nn.ReLU())
            ]
        layers = OrderedDict(layers)
        self.conv = torch.nn.Sequential(layers)
        self.affine = torch.nn.Linear(32 * (d_input-2*layer_num), d_model)

    def forward(self, feats, feat_lengths):
        outputs = feats.unsqueeze(1)  # [B, C, T, D]
        outputs = self.conv(outputs)
        B, C, T, D = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(B, T, C*D)
        outputs = self.affine(outputs)
        output_lengths = feat_lengths
        for _ in range(self.layer_num):
            output_lengths = ((output_lengths-1) / 2).long()

        return outputs, output_lengths
