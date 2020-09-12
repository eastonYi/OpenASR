import torch
from torch.nn.init import xavier_uniform_


class Framework(torch.nn.Module):
    def __init__(self, splayer, encoder, decoder):
        super().__init__()
        self.splayer = splayer
        self.encoder = encoder
        self.decoder = decoder
        self._reset_parameters()

    def forward(*args, **kwargs):
        raise NotImplementedError("Model must implement the build_model method")

    def get_logits(self, feats, len_feats):
        encoded, len_encoded = self.get_encoded(feats, len_feats)
        outputs, len_decoded = self.decoder(encoded, len_encoded)

        return outputs, len_decoded

    def get_encoded(self, feats, len_feats):
        encoded, len_encoded = self.splayer(feats, len_feats)
        encoded, len_encoded = self.encoder(encoded, len_encoded)

        return encoded, len_encoded

    @staticmethod
    def batch_beam_decode(logits, len_logits, decode_fn):

        raise NotImplementedError("Model must implement the build_model method")

    @classmethod
    def create_model(cls, sp_config, en_config, vocab_size):

        raise NotImplementedError("Model must implement the build_model method")

    def package(self):

        raise NotImplementedError("Model must implement the build_model method")

    def restore(self, pkg, without_fc=False):
        # check config
        raise NotImplementedError("Model must implement the build_model method")

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
