"""
Copyright 2020 Ye Bai by1993@qq.com
"""
import logging
import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from frameworks import Framework
from .Speech_Models import Conv_CTC
from .Speech_Models import Conv_Transformer
from loss import cal_ce_loss, cal_ctc_loss

inf = 1e10


class Embed_Decoder(Framework):
    def __init__(self, encoder, decoder):
        torch.nn.Module.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self._reset_parameters()

    def forward(self, tokens_input, len_input, target_input, target_output, target_paddings, label_smooth=0.):
        len_targets = torch.sum(1-target_paddings, dim=-1).long()
        logits = self.get_logits(tokens_input, len_input,
                                 target_input, len_targets)
        loss = cal_ce_loss(logits, target_output, target_paddings, label_smooth)

        return loss

    def get_logits(self, tokens_input, len_input, target_input, len_targets):
        encoded = self.encoder(tokens_input)
        outputs = self.decoder(encoded, len_input, target_input, len_targets)

        return outputs

    def get_encoded(self, feats, len_feats):
        encoded, len_encoded = self.encoder(feats, len_feats)

        return encoded, len_encoded

    @staticmethod
    def batch_beam_decode(*inputs, **kwad):

        return Conv_Transformer.batch_beam_decode(*inputs, **kwad)

    @classmethod
    def create_model(cls, en_config, de_config):
        from blocks.decoders import TransformerDecoder

        encoder = nn.Embedding(en_config['vocab_size'], en_config['d_model'])
        decoder = TransformerDecoder(de_config)
        model = cls(encoder, decoder)
        model.encoder.config = en_config

        return model

    def package(self):
        pkg = {
            "encoder_config": self.encoder.config,
            "encoder_state": self.encoder.state_dict(),
            "decoder_config": self.decoder.config,
            "decoder_state": self.decoder.state_dict(),
             }
        return pkg

    def restore(self, pkg):
        # check config
        logging.info("Restore model states...")
        for key in self.encoder.config.keys():
            if (key != "dropout_rate" and
                    self.encoder.config[key] != pkg["encoder_config"][key]):
                raise ValueError("decoder_config mismatch.")
        for key in self.decoder.config.keys():
            if (key != "dropout_rate" and
                    self.decoder.config[key] != pkg["decoder_config"][key]):
                raise ValueError("decoder_config mismatch.")

        self.encoder.load_state_dict(pkg["encoder_state"])
        self.decoder.load_state_dict(pkg["decoder_state"])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class Embed_Decoder_CTC(Embed_Decoder):
    def __init__(self, encoder, decoder, vocab_size):
        super().__init__(encoder, decoder)
        self.ctc_fc = nn.Linear(decoder.d_output, vocab_size, bias=False)
        self._reset_parameters()

    def forward(self, tokens_input, len_input, targets, target_paddings):
        target_lengths = torch.sum(1-target_paddings, dim=-1).long()

        ctc_logits, len_logits_ctc = self.get_logits(tokens_input, len_input)
        loss = cal_ctc_loss(ctc_logits, len_logits_ctc, targets, target_lengths)

        return loss

    def get_logits(self, tokens_input, len_input):
        encoded = self.encoder(tokens_input)
        outputs, len_output = self.decoder(encoded, len_input)
        ctc_logits = self.ctc_fc(outputs)

        return ctc_logits, len_output

    @staticmethod
    def batch_beam_decode(*args, **kwads):

        return Conv_CTC.batch_beam_decode(*args, **kwads)

    @classmethod
    def create_model(cls, en_config, de_config):
        from blocks.encoders import TransformerEncoder

        encoder = nn.Embedding(en_config['vocab_size'], en_config['d_model'])
        decoder = TransformerEncoder(de_config)
        model = cls(encoder, decoder, de_config['vocab_size'])
        model.encoder.config = en_config

        return model

    def package(self):
        pkg = {
            "encoder_config": self.encoder.config,
            "encoder_state": self.encoder.state_dict(),
            "decoder_config": self.decoder.config,
            "decoder_state": self.decoder.state_dict(),
            "ctc_fc_state": self.ctc_fc.state_dict()
             }
        return pkg

    def restore(self, pkg):
        # check config
        logging.info("Restore model states...")
        for key in self.decoder.config.keys():
            if (key != "dropout_rate" and
                    self.decoder.config[key] != pkg["decoder_config"][key]):
                raise ValueError("decoder_config mismatch.")

        self.encoder.load_state_dict(pkg["encoder_state"])
        self.decoder.load_state_dict(pkg["decoder_state"])
        self.ctc_fc.load_state_dict(pkg["ctc_fc_state"])
