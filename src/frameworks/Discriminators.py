"""
Copyright 2020 Ye Bai by1993@qq.com
"""
import logging
import torch
from torch import nn
import torch.nn.functional as F

import utils
from frameworks import Framework

inf = 1e10


class Discriminator(Framework):
    def __init__(self, encoder):
        torch.nn.Module.__init__(self)
        self.encoder = encoder
        self.ctc_fc = nn.Linear(encoder.d_model, 1, bias=False)
        self._reset_parameters()

    def forward(self, inputs, len_inputs):
        input_mask = utils.sequence_mask(len_inputs, depth=inputs.size(-1))
        inputs = inputs * input_mask
        score = self.get_logits(inputs, len_inputs)

        return score

    def get_logits(self, inputs, len_inputs):
        if inputs.size(1) < 10:
            inputs = F.pad(inputs, [0,0,0,10,0,0])
        encoded, _ = self.encoder(inputs, len_inputs)
        outputs = self.ctc_fc(encoded).mean(1)

        return outputs[:, 0]

    def gradient_penalty(self, real_data, fake_data, lengths):
        device = real_data.device
        B = real_data.size(0)
        alpha = torch.rand(B, 1, 1).to(device)
        alpha = alpha.expand(real_data.size())
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self(interpolates, lengths)
        # TODO: Make ConvBackward diffentiable
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    @classmethod
    def create_model(cls, D_config):
        from blocks.conv_layers import Conv2dSubsampleV2 as Encoder
        encoder = Encoder(D_config['encoder']['d_input'],
                          D_config['encoder']['d_model'],
                          D_config['encoder']['layer_num'])
        model = cls(encoder)
        model.config = D_config

        return model

    def package(self):
        pkg = {
            "encoder_config": self.encoder.config,
            "encoder_state": self.encoder.state_dict(),
            "ctc_fc_state": self.ctc_fc.state_dict()
             }
        return pkg

    def restore(self, pkg):
        # check config
        logging.info("Restore model states...")
        for key in self.encoder.config.keys():
            if (key != "dropout_rate" and
                    self.encoder.config[key] != pkg["encoder_config"][key]):
                raise ValueError("decoder_config mismatch.")

        self.encoder.load_state_dict(pkg["encoder_state"])
        self.ctc_fc.load_state_dict(pkg["ctc_fc_state"])
