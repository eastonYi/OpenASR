"""
Copyright 2020 Ye Bai by1993@qq.com
"""
import torch.nn.functional as F

import utils
from frameworks import Framework
from loss import cal_ce_loss, cal_ctc_loss

inf = 1e10


class GAN_Phone2Char(Framework):
    def __init__(self, G, D):
        super().__init__()
        self.G = G
        self.D = D
        self._reset_parameters()

    def forward(self, x, len_x, text, text_paddings, _x, _len_x, _y, _y_paddings):

        # supervise ['encoder_out', 'encoder_padding_mask', 'padding_mask']
        _loss = self.G(_x, _len_x, _y, _y_paddings) #

        # neg score
        logits, len_logits = self.G.get_logits(x, len_x)
        pad = 2
        blk = logits.size(-1) - 1
        logits_G, len_decode_G = utils.ctc_shrink(logits, pad=pad, blk=blk)
        probs_G = F.softmax(logits_G, -1)
        mask = utils.sequence_mask(len_decode_G).unsqueeze(-1).repeat(1, 1, probs_G.size(-1))
        score_neg = self.D(probs_G, mask)

        # pos score
        feature_text = F.one_hot(text.long(), probs_G.size(-1)).float()
        score_pos = self.D(feature_text, 1-text_paddings)

        min_len = min(feature_text.size(1), probs_G.size(1))
        gp = 1.0 * self.D.gradient_penalty(
            real_data=feature_text[:, :min_len, :],
            fake_data=probs_G[:, :min_len, :])

        return _loss, score_neg, (score_pos, gp)

    @classmethod
    def create_model(cls, G_config, D_config):
        from frameworks.Text_Models import Embed_Decoder_CTC
        from .Discriminators import Discriminator

        G = Embed_Decoder_CTC(G_config)
        D = Discriminator(D_config)
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
