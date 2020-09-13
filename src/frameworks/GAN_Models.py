"""
Copyright 2020 Ye Bai by1993@qq.com
"""
import torch
import torch.nn.functional as F

import utils
from frameworks import Framework
from loss import cal_ce_loss, cal_ctc_loss

inf = 1e10
PAD_ID = 2

class GAN_Phone2Char(Framework):
    def __init__(self, G, D):
        torch.nn.Module.__init__(self)
        self.G = G
        self.D = D
        self._reset_parameters()

    def forward(self, x, len_x, y, y_paddings):

        return self.G(x, len_x, y, y_paddings)

    def forward_G(self, x, len_x):
        logits, len_logits = self.G.get_logits(x, len_x)
        blk = logits.size(-1) - 1
        logits_G, len_decode_G = utils.ctc_shrink(logits, pad=PAD_ID, blk=blk)
        probs_G = F.softmax(logits_G, -1)
        loss_G = -self.D(probs_G, len_decode_G).sum()

        return loss_G

    def forward_D(self, x, len_x, text, len_text):
        with torch.no_grad():
            logits, len_logits = self.G.get_logits(x, len_x)
            blk = logits.size(-1) - 1
            logits_G, len_decode_G = utils.ctc_shrink(logits, pad=PAD_ID, blk=blk)
            probs_G = F.softmax(logits_G, -1)
        score_neg = self.D(probs_G, len_decode_G).sum()

        # pos score
        feature_text = F.one_hot(text.long(), probs_G.size(-1)).float()
        score_pos = self.D(feature_text, len_text).sum()

        lengths = torch.min(len_decode_G, len_text)
        min_len = lengths.max()
        if min_len > 0:
            gp = 1.0 * self.D.gradient_penalty(
                real_data=feature_text[:, :min_len, :],
                fake_data=probs_G[:, :min_len, :],
                lengths=lengths)
        else:
            gp = 0

        loss_D = score_neg - score_pos + gp

        return loss_D

    @classmethod
    def create_model(cls, G_config, D_config):
        from frameworks.Text_Models import Embed_Decoder_CTC
        from .Discriminators import Discriminator

        G = Embed_Decoder_CTC.create_model(G_config['encoder'], G_config['decoder'])
        D = Discriminator.create_model(D_config)
        model = cls(G, D)
        model.G.config = G_config
        model.D.config = D_config

        return model

    def package(self):
        pkg = {
            "G_config": self.G.config,
            "G_state": self.G.state_dict(),
            "D_config": self.D.config,
            "D_state": self.D.state_dict(),
             }
        return pkg

    def restore(self, pkg):
        # check config
        self.G.restore(pkg["G"])
        self.D.restore(pkg["D"])

    def restore_G(self, G_path):
        print("Load package from {}.".format(G_path))
        pkg = torch.load(G_path)
        self.G.restore(pkg['model'])
