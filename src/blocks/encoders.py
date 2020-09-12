"""
Copyright 2020 Ye Bai by1993@qq.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm

import utils
from third_party import transformer
from blocks.position import PositionalEncoding
from blocks.conv_layers import Conv1dSubsample, Conv2dSubsample, Conv2dSubsampleV2


class TransformerEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_dim = config["input_dim"]
        self.d_output = self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.dim_feedforward = config["dim_feedforward"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.activation = config["activation"]

        self.subconf = config["sub"]
        if self.subconf:
            if self.subconf["type"] == "ConvV1":
                self.sub = Conv2dSubsample(self.input_dim, self.d_model)
            elif self.subconf["type"] == "ConvV2":
                self.sub = Conv2dSubsampleV2(self.input_dim, self.d_model, self.subconf["layer_num"])
            elif self.subconf["type"] == "Stack":
                self.context_width = config["context_width"]
                self.subsample = config["subsample"]
                self.sub = Conv1dSubsample(self.input_dim, self.d_model, self.context_width, self.subsample)
        elif self.input_dim == self.d_model:
            self.affine = lambda x: x
        else:
            self.affine = torch.nn.Linear(self.input_dim, self.d_model)

        self.scale = self.d_model ** 0.5

        self.pe = PositionalEncoding(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)
        encoder_norm = LayerNorm(self.d_model)
        encoder_layer = transformer.TransformerEncoderLayer(d_model=self.d_model,
                nhead=self.nhead, dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_rate, activation=self.activation)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layer, self.num_layers, encoder_norm)

    def forward(self, feats, feat_lengths):
        if self.subconf:
            outputs, output_lengths = self.sub(feats, feat_lengths)
        else:
            outputs, output_lengths = self.affine(feats), feat_lengths

        outputs = self.dropout(self.pe(outputs))

        B, T, D_o = outputs.shape
        src_key_padding_mask = utils.get_transformer_padding_byte_masks(B, T, output_lengths).to(outputs.device)
        outputs = outputs.permute(1, 0, 2)

        outputs = self.transformer_encoder(outputs,
                src_key_padding_mask=src_key_padding_mask)
        outputs = outputs.permute(1, 0, 2)

        return outputs, output_lengths


class GRU_Encoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_input = config["d_input"]
        self.d_model = config["d_model"]
        self.n_layers = config["n_layers"]
        self.dropout = config["dropout"]
        self.bidirectional = False
        self.d_output = self.d_model * self.n_layers

        self.encoder = nn.GRU(self.d_input, self.d_model,
                              num_layers=self.n_layers, dropout=self.dropout,
                              bidirectional=self.bidirectional, batch_first=True)

        for layer_p in self.encoder._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.encoder.__getattr__(p),
                                            mode='fan_out', nonlinearity='relu')

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.n_layers * (2 if self.bidirectional else 1),
                           batch_size, self.d_model).to(device)

    def forward(self, feats, feat_lengths):

        hidden = self.init_hidden(len(feats), feats.device)
        x, state = self.encoder(feats, hidden)
        len_x = feat_lengths

        return x, len_x
