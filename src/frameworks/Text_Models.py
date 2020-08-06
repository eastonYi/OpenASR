"""
Copyright 2020 Ye Bai by1993@qq.com
"""
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from loss import cal_ce_loss

inf = 1e10


class Embed_Decoder(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.emb_input = nn.Embedding(decoder.vocab_size, decoder.d_model)
        self.decoder = decoder
        self._reset_parameters()

    def forward(self, tokens_input, len_input, target_input, target_output, target_paddings, label_smooth=0.):
        len_targets = torch.sum(1-target_paddings, dim=-1).long()
        logits = self.get_logits(tokens_input, len_input,
                                 target_input, len_targets)
        loss = cal_ce_loss(logits, target_output, target_paddings, label_smooth)

        return loss

    def get_logits(self, tokens_input, len_input, target_input, len_targets):
        encoded = self.emb_input(tokens_input)
        outputs = self.decoder(encoded, len_input, target_input, len_targets)

        return outputs

    def batch_beam_decode(self, encoded, len_encoded, sosid, eosid, beam_size=1, max_decode_len=100):
        batch_size = len_encoded.size(0)
        device = encoded.device
        d_output = self.decoder.vocab_size

        # beam search Initialize
        # repeat each sample in batch along the batch axis [1,2,3,4] -> [1,1,2,2,3,3,4,4]
        encoded = encoded[:, None, :, :].repeat(1, beam_size, 1, 1) # [batch_size, beam_size, *, hidden_units]
        encoded = encoded.view(batch_size * beam_size, -1, encoded.size(-1))
        len_encoded = len_encoded[:, None].repeat(1, beam_size).view(-1) # [batch_size * beam_size]

        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        preds = torch.ones([batch_size * beam_size, 1]).long().to(device) * sosid
        logits = torch.zeros([batch_size * beam_size, 0, d_output]).float().to(device)
        len_decoded = torch.ones_like(len_encoded)
        # the score must be [0, -inf, -inf, ...] at init, for the preds in beam is same in init!!!
        scores = torch.tensor([0.0] + [-inf] * (beam_size - 1)).float().repeat(batch_size).to(device)  # [batch_size * beam_size]
        finished = torch.zeros_like(scores).bool().to(device)

        # collect the initial states of lstms used in decoder.
        base_indices = torch.arange(batch_size)[:, None].repeat(1, beam_size).view(-1).to(device)

        for _ in range(max_decode_len):
            # i, preds, scores, logits, len_decoded, finished
            cur_logits = self.decoder(encoded, len_encoded, preds, len_decoded)[:, -1]
            logits = torch.cat([logits, cur_logits[:, None]], 1)  # [batch*beam, size_output]
            z = F.log_softmax(cur_logits) # [batch*beam, size_output]

            # rank the combined scores
            next_scores, next_preds = torch.topk(z, k=beam_size, sorted=True, dim=-1)

            # beamed scores & Pruning
            scores = scores[:, None] + next_scores  # [batch_size * beam_size, beam_size]
            scores = scores.view(batch_size, beam_size * beam_size)

            _, k_indices = torch.topk(scores, k=beam_size)
            k_indices = base_indices * beam_size * beam_size + k_indices.view(-1)  # [batch_size * beam_size]
            # Update scores.
            scores = scores.view(-1)[k_indices]
            # Update predictions.
            next_preds = next_preds.view(-1)[k_indices]

            # k_indices: [0~batch*beam*beam], preds: [0~batch*beam]
            # preds, cache_lm, cache_decoder: these data are shared during the beam expand among vocab
            preds = preds[k_indices // beam_size]
            preds = torch.cat([preds, next_preds[:, None]], axis=1)  # [batch_size * beam_size, i]

            has_eos = next_preds.eq(eosid)
            finished = torch.logical_or(finished, has_eos)
            len_decoded += 1 - finished.int()

            if finished.int().sum() == finished.size(0):
                break

        len_decoded -= 1 - finished.int() # for decoded length cut by encoded length
        preds = preds[:, 1:]
        # tf.nn.top_k is used to sort `scores`
        scores_sorted, sorted = torch.topk(scores.view(batch_size, beam_size),
                                           k=beam_size, sorted=True)
        sorted = base_indices * beam_size + sorted.view(-1)  # [batch_size * beam_size]

        # [batch_size * beam_size, ...] -> [batch_size, beam_size, ...]
        logits_sorted = logits[sorted].view(batch_size, beam_size, -1, d_output)
        preds_sorted = preds[sorted].view(batch_size, beam_size, -1) # [batch_size, beam_size, max_length]
        len_decoded_sorted = len_decoded[sorted].view(batch_size, beam_size)
        scores_sorted = scores[sorted].view(batch_size, beam_size)

        # import pdb; pdb.set_trace()
        # print('here')
        return preds_sorted, len_decoded_sorted, scores_sorted

    @classmethod
    def create_model(cls, de_config):
        from blocks.decoders import TransformerDecoder

        decoder = TransformerDecoder(de_config)
        model = cls(decoder)

        return model

    def package(self):
        pkg = {
            "emb_input": self.emb_input.state_dict(),
            "decoder_config": self.decoder.config,
            "decoder_state": self.decoder.state_dict(),
             }
        return pkg

    def restore(self, pkg):
        # check config
        logging.info("Restore model states...")
        for key in self.decoder.config.keys():
            if (key != "dropout_rate" and
                    self.decoder.config[key] != pkg["decoder_config"][key]):
                raise ValueError("decoder_config mismatch.")

        self.emb_input.load_state_dict(pkg["emb_input"])
        self.decoder.load_state_dict(pkg["decoder_state"])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
