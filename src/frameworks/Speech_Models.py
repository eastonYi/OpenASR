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
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from loss import cal_ctc_loss, cal_ce_loss, cal_qua_loss

inf = 1e10
SOS_ID = 1
EOS_ID = 2

class Conv_Transformer(torch.nn.Module):
    def __init__(self, splayer, encoder, decoder):
        super().__init__()
        self.splayer = splayer
        self.encoder = encoder
        self.decoder = decoder
        self._reset_parameters()

    def forward(self, batch_wave, lengths, target_ids, target_labels=None, target_paddings=None, label_smooth=0.):
        target_lengths = torch.sum(1-target_paddings, dim=-1).long()
        logits = self.get_logits(batch_wave, lengths,
                target_ids, target_lengths)
        loss = cal_ce_loss(logits, target_labels, target_paddings, label_smooth)

        return loss

    def get_logits(self, batch_wave, lengths, target_ids, target_lengths):
        encoder_outputs, encoder_output_lengths = self.splayer(batch_wave, lengths)
        encoder_outputs, encoder_output_lengths = self.encoder(encoder_outputs, encoder_output_lengths)
        outputs = self.decoder(encoder_outputs, encoder_output_lengths, target_ids, target_lengths)

        return outputs

    def get_encoded(self, padded_waveforms, wave_lengths):
        encoded, len_encoded = self.splayer(padded_waveforms, wave_lengths)
        encoded, len_encoded = self.encoder(encoded, len_encoded)

        return encoded, len_encoded

    @staticmethod
    def batch_beam_decode(encoded, len_encoded, step_forward_fn, vocab_size, beam_size=1, max_decode_len=100):
        batch_size = len_encoded.size(0)
        device = encoded.device
        d_output = vocab_size

        # beam search Initialize
        # repeat each sample in batch along the batch axis [1,2,3,4] -> [1,1,2,2,3,3,4,4]
        encoded = encoded[:, None, :, :].repeat(1, beam_size, 1, 1) # [batch_size, beam_size, *, hidden_units]
        encoded = encoded.view(batch_size * beam_size, -1, encoded.size(-1))
        len_encoded = len_encoded[:, None].repeat(1, beam_size).view(-1) # [batch_size * beam_size]

        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        preds = torch.ones([batch_size * beam_size, 1]).long().to(device) * SOS_ID
        logits = torch.zeros([batch_size * beam_size, 0, d_output]).float().to(device)
        len_decoded = torch.ones_like(len_encoded)
        # the score must be [0, -inf, -inf, ...] at init, for the preds in beam is same in init!!!
        scores = torch.tensor([0.0] + [-inf] * (beam_size - 1)).float().repeat(batch_size).to(device)  # [batch_size * beam_size]
        finished = torch.zeros_like(scores).bool().to(device)

        # collect the initial states of lstms used in decoder.
        base_indices = torch.arange(batch_size)[:, None].repeat(1, beam_size).view(-1).to(device)

        for _ in range(max_decode_len):
            # i, preds, scores, logits, len_decoded, finished
            cur_logits = step_forward_fn(encoded, len_encoded, preds, len_decoded)
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

            has_eos = next_preds.eq(EOS_ID)
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

        return preds_sorted, len_decoded_sorted, scores_sorted

    @classmethod
    def create_model(cls, sp_config, en_config, de_config):
        from blocks.sp_layers import SPLayer
        from blocks.encoders import TransformerEncoder
        from blocks.decoders import TransformerDecoder

        splayer = SPLayer(sp_config)
        encoder = TransformerEncoder(en_config)
        decoder = TransformerDecoder(de_config)

        model = cls(splayer, encoder, decoder)

        return model

    def package(self):
        pkg = {
            "splayer_config": self.splayer.config,
            "splayer_state": self.splayer.state_dict(),
            "encoder_config": self.encoder.config,
            "encoder_state": self.encoder.state_dict(),
            "decoder_config": self.decoder.config,
            "decoder_state": self.decoder.state_dict(),
             }
        return pkg

    def restore(self, pkg):
        # check config
        logging.info("Restore model states...")
        for key in self.splayer.config.keys():
            if key == "spec_aug":
                continue
            if self.splayer.config[key] != pkg["splayer_config"][key]:
                raise ValueError("splayer_config mismatch.")
        for key in self.encoder.config.keys():
            if (key != "dropout_rate" and
                    self.encoder.config[key] != pkg["encoder_config"][key]):
                raise ValueError("encoder_config mismatch.")
        for key in self.decoder.config.keys():
            if (key != "dropout_rate" and
                    self.decoder.config[key] != pkg["decoder_config"][key]):
                raise ValueError("decoder_config mismatch.")

        self.splayer.load_state_dict(pkg["splayer_state"])
        self.encoder.load_state_dict(pkg["encoder_state"])
        self.decoder.load_state_dict(pkg["decoder_state"])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class Conv_CTC_Transformer(Conv_Transformer):
    def __init__(self, splayer, encoder, decoder):
        super().__init__(splayer, encoder, decoder)
        self.ctc_fc = nn.Linear(encoder.d_model, decoder.vocab_size, bias=False)
        self._reset_parameters()

    def forward(self, batch_wave, lengths, target_ids, target_labels=None, target_paddings=None, label_smooth=0.):
        target_lengths = torch.sum(1-target_paddings, dim=-1).long()
        ctc_logits, len_logits_ctc, ce_logits = self.get_logits(
            batch_wave, lengths, target_ids, target_lengths)

        ctc_loss = cal_ctc_loss(ctc_logits, len_logits_ctc, target_labels, target_lengths-1) # the target of ctc counts without blk
        ce_loss = cal_ce_loss(ce_logits, target_labels, target_paddings, label_smooth)

        return ctc_loss, ce_loss

    def get_logits(self, batch_wave, lengths, target_ids, target_lengths):
        encoder_outputs, encoder_output_lengths = self.splayer(batch_wave, lengths)
        encoder_outputs, encoder_output_lengths = self.encoder(encoder_outputs, encoder_output_lengths)
        ctc_logits = self.ctc_fc(encoder_outputs)
        ce_logits = self.decoder(encoder_outputs, encoder_output_lengths, target_ids, target_lengths)
        len_logits_ctc = encoder_output_lengths

        return ctc_logits, len_logits_ctc, ce_logits

    @classmethod
    def create_model(cls, sp_config, en_config, de_config):
        from blocks.sp_layers import SPLayer
        from blocks.encoders import TransformerEncoder
        from blocks.decoders import TransformerDecoder

        splayer = SPLayer(sp_config)
        encoder = TransformerEncoder(en_config)
        decoder = TransformerDecoder(de_config)

        model = cls(splayer, encoder, decoder)

        return model

    def package(self):
        pkg = {
            "splayer_config": self.splayer.config,
            "splayer_state": self.splayer.state_dict(),
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
        for key in self.splayer.config.keys():
            if key == "spec_aug":
                continue
            if self.splayer.config[key] != pkg["splayer_config"][key]:
                raise ValueError("splayer_config mismatch.")
        for key in self.encoder.config.keys():
            if (key != "dropout_rate" and
                    self.encoder.config[key] != pkg["encoder_config"][key]):
                raise ValueError("encoder_config mismatch.")
        for key in self.decoder.config.keys():
            if (key != "dropout_rate" and
                    self.decoder.config[key] != pkg["decoder_config"][key]):
                raise ValueError("decoder_config mismatch.")

        self.splayer.load_state_dict(pkg["splayer_state"])
        self.encoder.load_state_dict(pkg["encoder_state"])
        self.decoder.load_state_dict(pkg["decoder_state"])
        self.ctc_fc.load_state_dict(pkg["ctc_fc_state"])


class CIF(Conv_CTC_Transformer):
    def __init__(self, splayer, encoder, assigner, decoder):
        super().__init__(splayer, encoder, decoder)
        self.assigner = assigner
        self._reset_parameters()

    def forward(self, batch_wave, lengths, target_ids, target_labels=None, target_paddings=None, label_smooth=0., threshold=0.95):
        device = batch_wave.device
        target_lengths = torch.sum(1-target_paddings, dim=-1).long()

        encoder_outputs, encoder_output_lengths = self.splayer(batch_wave, lengths)
        encoder_outputs, encoder_output_lengths = self.encoder(encoder_outputs, encoder_output_lengths)
        ctc_logits = self.ctc_fc(encoder_outputs)

        len_logits_ctc = encoder_output_lengths
        alphas = self.assigner(encoder_outputs, encoder_output_lengths)

        # sum
        _num = alphas.sum(-1)
        # scaling
        num = target_lengths.float()
        num_noise = num + 0.9 * torch.rand(alphas.size(0)).to(device) - 0.45
        alphas *= (num_noise / _num)[:, None].repeat(1, alphas.size(1))

        cif_outputs = self.cif(encoder_outputs, alphas, threshold=threshold)

        logits = self.decoder(cif_outputs, target_ids, target_lengths)

        ctc_loss = cal_ctc_loss(ctc_logits, len_logits_ctc, target_labels, target_lengths)
        qua_loss = cal_qua_loss(_num, num)
        ce_loss = cal_ce_loss(logits, target_labels, target_paddings, label_smooth)

        return ctc_loss, qua_loss, ce_loss

    def cif(self, hidden, alphas, threshold=0.95, log=False):
        device = hidden.device
        batch_size, len_time, hidden_size = hidden.size()

        # loop varss
        integrate = torch.zeros([batch_size]).to(device)
        frame = torch.zeros([batch_size, hidden_size]).to(device)
        # intermediate vars along time
        list_fires = []
        list_frames = []

        for t in range(len_time):
            alpha = alphas[:, t]
            distribution_completion = torch.ones([batch_size]).to(device) - integrate

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate > threshold
            integrate = torch.where(fire_place,
                                    integrate - torch.ones([batch_size]).to(device),
                                    integrate)
            cur = torch.where(fire_place,
                              distribution_completion,
                              alpha)
            remainds = alpha - cur

            frame += cur[:, None] * hidden[:, t, :]
            list_frames.append(frame)
            frame = torch.where(fire_place[:, None].repeat(1, hidden_size),
                                remainds[:, None] * hidden[:, t, :],
                                frame)
            if log:
                print('t: {}\t{:.3f} -> {:.3f}|{:.3f}'.format(
                    t, integrate[0].numpy(), cur[0].numpy(), remainds[0].numpy()))

        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        list_ls = []
        len_labels = torch.round(alphas.sum(-1)).int()
        max_label_len = len_labels.max()
        for b in range(batch_size):
            fire = fires[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.where(fire > threshold)[0])
            pad_l = torch.zeros([max_label_len - l.size(0), hidden_size]).to(device)
            list_ls.append(torch.cat([l, pad_l], 0))

        if log:
            print('fire:\n', fires.numpy())

        return torch.stack(list_ls, 0)

    def get_encoded(self, padded_waveforms, wave_lengths):
        encoded, len_encoded = self.splayer(padded_waveforms, wave_lengths)
        encoded, len_encoded = self.encoder(encoded, len_encoded)
        alphas = self.assigner(encoded, len_encoded)

        len_encoded = torch.round(alphas.sum(-1)).int()
        encoded = self.cif(encoded, alphas)

        return encoded, len_encoded

    @classmethod
    def batch_beam_decode(encoded, len_encoded, step_forward_fn, vocab_size, beam_size=1, max_decode_len=None):
        batch_size = len_encoded.size(0)
        device = encoded.device
        d_output = vocab_size

        # beam search Initialize
        # repeat each sample in batch along the batch axis [1,2,3,4] -> [1,1,2,2,3,3,4,4]
        encoded = encoded[:, None, :, :].repeat(1, beam_size, 1, 1) # [batch_size, beam_size, *, hidden_units]
        encoded = encoded.view(batch_size * beam_size, -1, encoded.size(-1))
        len_encoded = len_encoded[:, None].repeat(1, beam_size).view(-1) # [batch_size * beam_size]

        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        preds = torch.ones([batch_size * beam_size, 1]).long().to(device) * SOS_ID
        logits = torch.zeros([batch_size * beam_size, 0, d_output]).float().to(device)
        # the score must be [0, -inf, -inf, ...] at init, for the preds in beam is same in init!!!
        scores = torch.tensor([0.0] + [-inf] * (beam_size - 1)).float().repeat(batch_size).to(device)  # [batch_size * beam_size]

        # collect the initial states of lstms used in decoder.
        base_indices = torch.arange(batch_size)[:, None].repeat(1, beam_size).view(-1).to(device)

        for _ in range(encoded.size(1)):
            # i, preds, scores, logits, len_decoded
            cur_logits = step_forward_fn(encoded, len_encoded, preds)
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

        preds = preds[:, 1:]
        # tf.nn.top_k is used to sort `scores`
        scores_sorted, sorted = torch.topk(scores.view(batch_size, beam_size),
                                           k=beam_size, sorted=True)
        sorted = base_indices * beam_size + sorted.view(-1)  # [batch_size * beam_size]

        # [batch_size * beam_size, ...] -> [batch_size, beam_size, ...]
        logits_sorted = logits[sorted].view(batch_size, beam_size, -1, d_output)
        preds_sorted = preds[sorted].view(batch_size, beam_size, -1) # [batch_size, beam_size, max_length]
        len_decoded_sorted = len_encoded[sorted].view(batch_size, beam_size)
        scores_sorted = scores[sorted].view(batch_size, beam_size)

        # import pdb; pdb.set_trace()
        # print('here')
        return preds_sorted, len_decoded_sorted, scores_sorted

    @classmethod
    def create_model(cls, sp_config, en_config, as_cofig, de_config):
        from blocks.sp_layers import SPLayer
        from blocks.encoders import TransformerEncoder
        from blocks.attention_assigner import Attention_Assigner
        from blocks.decoders import CIF_Decoder

        splayer = SPLayer(sp_config)
        encoder = TransformerEncoder(en_config)
        assigner = Attention_Assigner(as_cofig)
        decoder = CIF_Decoder(de_config)

        model = cls(splayer, encoder, assigner, decoder)

        return model

    def package(self):
        pkg = {
            "splayer_config": self.splayer.config,
            "splayer_state": self.splayer.state_dict(),
            "encoder_config": self.encoder.config,
            "encoder_state": self.encoder.state_dict(),
            "assigner_config": self.assigner.config,
            "assigner_state": self.assigner.state_dict(),
            "decoder_config": self.decoder.config,
            "decoder_state": self.decoder.state_dict(),
            "ctc_fc_state": self.ctc_fc.state_dict(),
             }
        return pkg

    def restore(self, pkg):
        # check config
        logging.info("Restore model states...")
        for key in self.splayer.config.keys():
            if key == "spec_aug":
                continue
            if self.splayer.config[key] != pkg["splayer_config"][key]:
                raise ValueError("splayer_config mismatch.")
        for key in self.encoder.config.keys():
            if (key != "dropout_rate" and
                    self.encoder.config[key] != pkg["encoder_config"][key]):
                raise ValueError("encoder_config mismatch.")
        for key in self.assigner.config.keys():
            if (key != "dropout_rate" and
                    self.assigner.config[key] != pkg["assigner_config"][key]):
                raise ValueError("assigner_config mismatch.")
        for key in self.decoder.config.keys():
            if (key != "dropout_rate" and
                    self.decoder.config[key] != pkg["decoder_config"][key]):
                raise ValueError("decoder_config mismatch.")

        self.splayer.load_state_dict(pkg["splayer_state"])
        self.encoder.load_state_dict(pkg["encoder_state"])
        self.assigner.load_state_dict(pkg["assigner_state"])
        self.decoder.load_state_dict(pkg["decoder_state"])
        self.ctc_fc.load_state_dict(pkg["ctc_fc_state"])


class CIF_FC(CIF):
    def __init__(self, splayer, encoder, assigner, vocab_size):
        torch.nn.Module.__init__(self)
        self.splayer = splayer
        self.encoder = encoder
        self.assigner = assigner
        self.vocab_size = vocab_size
        self.ctc_fc = nn.Linear(encoder.d_model, vocab_size, bias=False)
        self.phone_fc = nn.Linear(encoder.d_model, vocab_size, bias=False)
        self._reset_parameters()

    def forward(self, batch_wave, lengths, phone, len_phone, label_smooth=0., threshold=0.95):
        device = batch_wave.device
        phone_paddings = phone.eq(0).float()

        encoder_outputs, encoder_output_lengths = self.splayer(batch_wave, lengths)
        encoder_outputs, encoder_output_lengths = self.encoder(encoder_outputs, encoder_output_lengths)
        ctc_logits = self.ctc_fc(encoder_outputs)

        len_logits_ctc = encoder_output_lengths
        alphas = self.assigner(encoder_outputs, encoder_output_lengths)

        # sum
        _num = alphas.sum(-1)
        # scaling
        num = len_phone.float()
        num_noise = num + 0.9 * torch.rand(alphas.size(0)).to(device) - 0.45
        alphas *= (num_noise / _num)[:, None].repeat(1, alphas.size(1))

        cif_outputs = self.cif(encoder_outputs, alphas, threshold=threshold)

        logits_IPA = self.phone_fc(cif_outputs)

        ctc_loss = cal_ctc_loss(ctc_logits, len_logits_ctc, phone, len_phone)
        qua_loss = cal_qua_loss(_num, num)
        ce_phone_loss = cal_ce_loss(logits_IPA, phone, phone_paddings, label_smooth)

        return ctc_loss, qua_loss, ce_phone_loss

    @staticmethod
    def batch_beam_decode(encoded, len_encoded, step_forward_fn, vocab_size, beam_size, max_decode_len=100):
        return Conv_Transformer.batch_beam_decode(
            encoded, len_encoded, step_forward_fn, vocab_size, beam_size=beam_size, max_decode_len=max_decode_len)

    @classmethod
    def create_model(cls, sp_config, en_config, as_cofig, vocab_size):
        from blocks.sp_layers import SPLayer
        from blocks.encoders import TransformerEncoder
        from blocks.attention_assigner import Attention_Assigner

        splayer = SPLayer(sp_config)
        encoder = TransformerEncoder(en_config)
        assigner = Attention_Assigner(as_cofig)

        model = cls(splayer, encoder, assigner, vocab_size)

        return model

    def package(self):
        pkg = {
            "splayer_config": self.splayer.config,
            "splayer_state": self.splayer.state_dict(),
            "encoder_config": self.encoder.config,
            "encoder_state": self.encoder.state_dict(),
            "assigner_config": self.assigner.config,
            "assigner_state": self.assigner.state_dict(),
            "ctc_fc_state": self.ctc_fc.state_dict(),
            "phone_fc_state": self.phone_fc.state_dict(),
             }
        return pkg

    def restore(self, pkg):
        # check config
        logging.info("Restore model states...")
        for key in self.splayer.config.keys():
            if key == "spec_aug":
                continue
            if self.splayer.config[key] != pkg["splayer_config"][key]:
                raise ValueError("splayer_config mismatch.")
        for key in self.encoder.config.keys():
            if (key != "dropout_rate" and
                    self.encoder.config[key] != pkg["encoder_config"][key]):
                raise ValueError("encoder_config mismatch.")
        for key in self.assigner.config.keys():
            if (key != "dropout_rate" and
                    self.assigner.config[key] != pkg["assigner_config"][key]):
                raise ValueError("assigner_config mismatch.")

        self.splayer.load_state_dict(pkg["splayer_state"])
        self.encoder.load_state_dict(pkg["encoder_state"])
        self.assigner.load_state_dict(pkg["assigner_state"])
        self.ctc_fc.load_state_dict(pkg["ctc_fc_state"])
        self.phone_fc.load_state_dict(pkg["phone_fc_state"])


class CIF_MIX(CIF_FC):
    def __init__(self, splayer, encoder, assigner, phone_size, decoder):
        super().__init__(splayer, encoder, assigner, phone_size)
        self.decoder = decoder
        self._reset_parameters()

    def forward(self, batch_wave, lengths, phone, len_phone,
                target_input=None, targets=None, target_paddings=None, label_smooth=0., threshold=0.95):
        device = batch_wave.device
        phone_paddings = phone.eq(0).float()

        encoder_outputs, encoder_output_lengths = self.splayer(batch_wave, lengths)
        encoder_outputs, encoder_output_lengths = self.encoder(encoder_outputs, encoder_output_lengths)
        ctc_logits = self.ctc_fc(encoder_outputs)

        len_logits_ctc = encoder_output_lengths
        alphas = self.assigner(encoder_outputs, encoder_output_lengths)

        # sum
        _num = alphas.sum(-1)
        # scaling
        num = len_phone.float()
        num_noise = num + 0.9 * torch.rand(alphas.size(0)).to(device) - 0.45
        alphas *= (num_noise / _num)[:, None].repeat(1, alphas.size(1))

        cif_outputs = self.cif(encoder_outputs, alphas, threshold=threshold)
        cif_outputs_lengths = len_phone

        logits_IPA = self.phone_fc(cif_outputs)

        if target_input is not None:
            target_lengths = torch.sum(1-target_paddings, dim=-1).long()
            logits = self.decoder(cif_outputs, cif_outputs_lengths, target_input, target_lengths)

            ctc_loss = cal_ctc_loss(ctc_logits, len_logits_ctc, phone, len_phone)
            qua_loss = cal_qua_loss(_num, num)
            ce_phone_loss = cal_ce_loss(logits_IPA, phone, phone_paddings, label_smooth)
            ce_target_loss = cal_ce_loss(logits, targets, target_paddings, label_smooth)

            return ctc_loss, qua_loss, ce_phone_loss, ce_target_loss
        else:
            ctc_loss = cal_ctc_loss(ctc_logits, len_logits_ctc, phone, len_phone)
            qua_loss = cal_qua_loss(_num, num)
            ce_phone_loss = cal_ce_loss(logits_IPA, phone, phone_paddings, label_smooth)

            return ctc_loss, qua_loss, ce_phone_loss

    @staticmethod
    def batch_beam_decode(encoded, len_encoded, step_forward_fn, vocab_size, beam_size, max_decode_len=100):
        return Conv_Transformer.batch_beam_decode(
            encoded, len_encoded, step_forward_fn, vocab_size, beam_size=beam_size, max_decode_len=max_decode_len)

    @classmethod
    def create_model(cls, sp_config, en_config, as_cofig, phone_size, de_config):
        from blocks.sp_layers import SPLayer
        from blocks.encoders import TransformerEncoder
        from blocks.attention_assigner import Attention_Assigner
        from blocks.decoders import TransformerDecoder

        splayer = SPLayer(sp_config)
        encoder = TransformerEncoder(en_config)
        assigner = Attention_Assigner(as_cofig)
        decoder = TransformerDecoder(de_config)

        model = cls(splayer, encoder, assigner, phone_size, decoder)

        return model

    def package(self):
        pkg = {
            "splayer_config": self.splayer.config,
            "splayer_state": self.splayer.state_dict(),
            "encoder_config": self.encoder.config,
            "encoder_state": self.encoder.state_dict(),
            "assigner_config": self.assigner.config,
            "assigner_state": self.assigner.state_dict(),
            "decoder_config": self.decoder.config,
            "decoder_state": self.decoder.state_dict(),
            "ctc_fc_state": self.ctc_fc.state_dict(),
            "phone_fc_state": self.phone_fc.state_dict(),
             }
        return pkg

    def restore(self, pkg):
        # check config
        logging.info("Restore model states...")
        for key in self.splayer.config.keys():
            if key == "spec_aug":
                continue
            if self.splayer.config[key] != pkg["splayer_config"][key]:
                raise ValueError("splayer_config mismatch.")
        for key in self.encoder.config.keys():
            if (key != "dropout_rate" and
                    self.encoder.config[key] != pkg["encoder_config"][key]):
                raise ValueError("encoder_config mismatch.")
        for key in self.assigner.config.keys():
            if (key != "dropout_rate" and
                    self.assigner.config[key] != pkg["assigner_config"][key]):
                raise ValueError("assigner_config mismatch.")
        for key in self.decoder.config.keys():
            if (key != "dropout_rate" and
                    self.decoder.config[key] != pkg["decoder_config"][key]):
                raise ValueError("decoder_config mismatch.")

        self.splayer.load_state_dict(pkg["splayer_state"])
        self.encoder.load_state_dict(pkg["encoder_state"])
        self.assigner.load_state_dict(pkg["assigner_state"])
        self.decoder.load_state_dict(pkg["decoder_state"])
        self.ctc_fc.load_state_dict(pkg["ctc_fc_state"])
        self.phone_fc.load_state_dict(pkg["phone_fc_state"])


class Conv_CTC(torch.nn.Module):
    def __init__(self, splayer, encoder, vocab_size):
        super().__init__()
        self.splayer = splayer
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.fc = nn.Linear(encoder.d_model, vocab_size, bias=False)
        self._reset_parameters()

    def forward(self, batch_wave, lengths, targets, target_lengths):
        logits, len_logits = self.get_logits(batch_wave, lengths)
        loss = cal_ctc_loss(logits, len_logits, targets, target_lengths)

        return loss

    def get_logits(self, feats, len_feats):
        encoded, len_encoded = self.get_encoded(feats, len_feats)
        outputs = self.fc(encoded)

        return outputs, len_encoded

    def get_encoded(self, feats, len_feats):
        encoded, len_encoded = self.splayer(feats, len_feats)
        encoded, len_encoded = self.encoder(encoded, len_encoded)

        return encoded, len_encoded

    @staticmethod
    def batch_beam_decode(logits, len_logits, decode_fn, vocab_size, beam_size, max_decode_len):

        prob = torch.softmax(logits, -1)
        beam_results, beam_scores, timesteps, out_seq_len = decode_fn.decode(prob)
        # preds_sorted, len_decoded_sorted, scores_sorted = decode(log_prob, len_logits)

        return beam_results, out_seq_len, beam_scores

    @classmethod
    def create_model(cls, sp_config, en_config, vocab_size):
        from blocks.sp_layers import SPLayer
        from blocks.encoders import TransformerEncoder

        splayer = SPLayer(sp_config)
        encoder = TransformerEncoder(en_config)

        model = cls(splayer, encoder, vocab_size)

        return model

    def package(self):
        pkg = {
            "splayer_config": self.splayer.config,
            "splayer_state": self.splayer.state_dict(),
            "encoder_config": self.encoder.config,
            "encoder_state": self.encoder.state_dict(),
            "vocab_size": self.vocab_size,
            "fc_state": self.fc.state_dict(),
             }
        return pkg

    def restore(self, pkg):
        # check config
        logging.info("Restore model states...")
        for key in self.splayer.config.keys():
            if key == "spec_aug":
                continue
            if self.splayer.config[key] != pkg["splayer_config"][key]:
                raise ValueError("splayer_config mismatch.")
        for key in self.encoder.config.keys():
            if (key != "dropout_rate" and
                    self.encoder.config[key] != pkg["encoder_config"][key]):
                raise ValueError("encoder_config mismatch.")

        self.splayer.load_state_dict(pkg["splayer_state"])
        self.encoder.load_state_dict(pkg["encoder_state"])
        self.fc.load_state_dict(pkg["fc_state"])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
