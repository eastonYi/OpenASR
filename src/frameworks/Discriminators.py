"""
Copyright 2020 Ye Bai by1993@qq.com
"""
import logging
import torch
from torch import nn

from frameworks import Framework

inf = 1e10


class Discriminator(Framework):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.ctc_fc = nn.Linear(encoder.d_model, 1, bias=False)
        self._reset_parameters()

    def forward(self, inputs, input_mask):
        inputs *= input_mask
        score = self.get_logits(inputs)

        return score

    def get_logits(self, inputs):
        encoded = self.encoder(inputs)
        outputs = self.ctc_fc(encoded).mean(1)

        return outputs[:, 0]

    def gradient_penalty(self, real_data, fake_data):
        device = real_data.device
        B = real_data.size(0)
        alpha = torch.rand(B, 1, 1).to(device)
        alpha = alpha.expand(real_data.size())

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self(interpolates)["score"]
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
    def create_model(cls, en_config):
        from blocks.conv_layers import Conv2dSubsample

        encoder = Conv2dSubsample(en_config)
        model = cls(encoder)
        model.encoder.config = en_config

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
