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
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from loss import cal_ce_loss, cal_qua_loss


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

    def decode(self, batch_wave, lengths, nbest_keep, sosid=1, eosid=2, maxlen=100):
        if type(nbest_keep) != int:
            raise ValueError("nbest_keep must be a int.")
        encoder_outputs, encoder_output_lengths = self._get_acoustic_representations(
                batch_wave, lengths)
        target_ids, scores = self._beam_search(encoder_outputs, encoder_output_lengths, nbest_keep, sosid, eosid, maxlen)

        return target_ids, scores

    def _get_acoustic_representations(self, batch_wave, lengths):
        encoder_outputs, encoder_output_lengths = self.splayer(batch_wave, lengths)
        encoder_outputs, encoder_output_lengths = self.encoder(encoder_outputs, encoder_output_lengths)
        return encoder_outputs, encoder_output_lengths

    def _beam_search(self, encoder_outputs, encoder_output_lengths, nbest_keep, sosid, eosid, maxlen):

        B = encoder_outputs.shape[0]
        # init
        init_target_ids = torch.ones(B, 1).to(encoder_outputs.device).long()*sosid
        init_target_lengths = torch.ones(B).to(encoder_outputs.device).long()
        outputs = (self.decoder(encoder_outputs, encoder_output_lengths, init_target_ids, init_target_lengths)[:, -1, :])
        vocab_size = outputs.size(-1)
        outputs = outputs.view(B, vocab_size)
        log_probs = F.log_softmax(outputs, dim=-1)
        topk_res = torch.topk(log_probs, k=nbest_keep, dim=-1)
        nbest_ids = topk_res[1].view(-1)  #[batch_size*nbest_keep, 1]
        nbest_logprobs = topk_res[0].view(-1)

        target_ids = torch.ones(B*nbest_keep, 1).to(encoder_outputs.device).long()*sosid
        target_lengths = torch.ones(B*nbest_keep).to(encoder_outputs.device).long()

        target_ids = torch.cat([target_ids, nbest_ids.view(B*nbest_keep, 1)], dim=-1)
        target_lengths += 1

        finished_sel = None
        ended = []
        ended_scores = []
        ended_batch_idx = []
        for step in range(1, maxlen):
            (nbest_ids, nbest_logprobs, beam_from) = self._decode_single_step(
                    encoder_outputs, encoder_output_lengths, target_ids, target_lengths, nbest_logprobs, finished_sel)
            batch_idx = (torch.arange(B)*nbest_keep).view(B, -1).repeat(1, nbest_keep).contiguous().to(beam_from.device)
            batch_beam_from = (batch_idx + beam_from.view(-1, nbest_keep)).view(-1)
            nbest_logprobs = nbest_logprobs.view(-1)
            finished_sel = (nbest_ids.view(-1) == eosid)
            target_ids = target_ids[batch_beam_from]
            target_ids = torch.cat([target_ids, nbest_ids.view(B*nbest_keep, 1)], dim=-1)
            target_lengths += 1

            for i in range(finished_sel.shape[0]):
                if finished_sel[i]:
                    ended.append(target_ids[i])
                    ended_scores.append(nbest_logprobs[i])
                    ended_batch_idx.append(i // nbest_keep)
            target_ids = target_ids * (1 - finished_sel[:, None].long()) # mask out finished

        for i in range(target_ids.shape[0]):
            ended.append(target_ids[i])
            ended_scores.append(nbest_logprobs[i])
            ended_batch_idx.append(i // nbest_keep)

        formated = {}
        for i in range(B):
            formated[i] = []
        for i in range(len(ended)):
            if ended[i][0] == sosid:
                formated[ended_batch_idx[i]].append((ended[i], ended_scores[i]))
        for i in range(B):
            formated[i] = sorted(formated[i], key=lambda x:x[1], reverse=True)[:nbest_keep]

        target_ids = torch.zeros(B, nbest_keep, maxlen+1).to(encoder_outputs.device).long()
        scores = torch.zeros(B, nbest_keep).to(encoder_outputs.device)
        for i in range(B):
            for j in range(nbest_keep):
                item = formated[i][j]
                l = min(item[0].shape[0], target_ids[i, j].shape[0])
                target_ids[i, j, :l] = item[0][:l]
                scores[i, j] = item[1]
        return target_ids, scores

    def _decode_single_step(self, encoder_outputs, encoder_output_lengths, target_ids, target_lengths, accumu_scores, finished_sel=None):
        """
        encoder_outputs: [B, T_e, D_e]
        encoder_output_lengths: [B]
        target_ids: [B*nbest_keep, T_d]
        target_lengths: [B*nbest_keep]
        accumu_scores: [B*nbest_keep]
        """
        B, T_e, D_e = encoder_outputs.shape
        B_d, T_d = target_ids.shape
        if B_d % B != 0:
            raise ValueError("The dim of target_ids does not match the encoder_outputs.")
        nbest_keep = B_d // B
        encoder_outputs = (encoder_outputs.view(B, 1, T_e, D_e)
                .repeat(1, nbest_keep, 1, 1).view(B*nbest_keep, T_e, D_e))
        encoder_output_lengths = (encoder_output_lengths.view(B, 1)
                .repeat(1, nbest_keep).view(-1))

        # outputs: [B, nbest_keep, vocab_size]
        outputs = (self.decoder(encoder_outputs, encoder_output_lengths, target_ids, target_lengths)[:, -1, :])
        vocab_size = outputs.size(-1)
        outputs = outputs.view(B, nbest_keep, vocab_size)
        log_probs = F.log_softmax(outputs, dim=-1)  # [B, nbest_keep, vocab_size]
        if finished_sel is not None:
            log_probs = log_probs.view(B*nbest_keep, -1) - finished_sel.view(B*nbest_keep, -1).float()*9e9
            log_probs = log_probs.view(B, nbest_keep, vocab_size)
        this_accumu_scores = accumu_scores.view(B, nbest_keep, 1) + log_probs
        topk_res = torch.topk(this_accumu_scores.view(B, nbest_keep*vocab_size), k=nbest_keep, dim=-1)

        nbest_logprobs = topk_res[0]  # [B, nbest_keep]
        nbest_ids = topk_res[1] % vocab_size # [B, nbest_keep]
        beam_from = (topk_res[1] / vocab_size).long()
        return nbest_ids, nbest_logprobs, beam_from

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


class CIF(Conv_Transformer):
    def __init__(self, splayer, encoder, assigner, decoder):
        torch.nn.Module.__init__(self)
        self.splayer = splayer
        self.encoder = encoder
        self.assigner = assigner
        self.decoder = decoder
        self._reset_parameters()

    def forward(self, batch_wave, lengths, target_ids, target_labels=None, target_paddings=None, label_smooth=0., threshold=0.95):
        device = batch_wave.device
        target_lengths = torch.sum(1-target_paddings, dim=-1).long()

        encoder_outputs, encoder_output_lengths = self.splayer(batch_wave, lengths)
        encoder_outputs, encoder_output_lengths = self.encoder(encoder_outputs, encoder_output_lengths)
        alphas = self.assigner(encoder_outputs, encoder_output_lengths)

        # sum
        _num = alphas.sum(-1)
        # scaling
        num = (target_labels > 0).float().sum(-1)
        num_noise = num + 0.9 * torch.rand(alphas.size(0)).to(device) - 0.45
        alphas *= (num_noise / _num)[:, None].repeat(1, alphas.size(1))

        cif_outputs = self.cif(encoder_outputs, alphas, threshold=threshold)

        logits = self.decoder(cif_outputs, target_ids, target_lengths)

        qua_loss = cal_qua_loss(_num, num)
        ce_loss = cal_ce_loss(logits, target_labels, target_paddings, label_smooth)

        return qua_loss, ce_loss

    def cif(self, hidden, alphas, threshold, log=False):
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

    def decode(self, batch_wave, lengths, nbest_keep, sosid=1, eosid=2, maxlen=100):
        if type(nbest_keep) != int:
            raise ValueError("nbest_keep must be a int.")
        encoder_outputs, encoder_output_lengths = self._get_acoustic_representations(
                batch_wave, lengths)
        target_ids, scores = self._beam_search(encoder_outputs, encoder_output_lengths, nbest_keep, sosid, eosid, maxlen)

        return target_ids, scores

    def _get_acoustic_representations(self, batch_wave, lengths):
        encoder_outputs, encoder_output_lengths = self.splayer(batch_wave, lengths)
        encoder_outputs, encoder_output_lengths = self.encoder(encoder_outputs, encoder_output_lengths)

        return encoder_outputs, encoder_output_lengths

    def _beam_search(self, encoder_outputs, encoder_output_lengths, nbest_keep, sosid, eosid, maxlen):

        B = encoder_outputs.shape[0]
        # init
        init_target_ids = torch.ones(B, 1).to(encoder_outputs.device).long()*sosid
        init_target_lengths = torch.ones(B).to(encoder_outputs.device).long()
        outputs = (self.decoder(encoder_outputs, encoder_output_lengths, init_target_ids, init_target_lengths)[:, -1, :])
        vocab_size = outputs.size(-1)
        outputs = outputs.view(B, vocab_size)
        log_probs = F.log_softmax(outputs, dim=-1)
        topk_res = torch.topk(log_probs, k=nbest_keep, dim=-1)
        nbest_ids = topk_res[1].view(-1)  #[batch_size*nbest_keep, 1]
        nbest_logprobs = topk_res[0].view(-1)

        target_ids = torch.ones(B*nbest_keep, 1).to(encoder_outputs.device).long()*sosid
        target_lengths = torch.ones(B*nbest_keep).to(encoder_outputs.device).long()

        target_ids = torch.cat([target_ids, nbest_ids.view(B*nbest_keep, 1)], dim=-1)
        target_lengths += 1

        finished_sel = None
        ended = []
        ended_scores = []
        ended_batch_idx = []
        for step in range(1, maxlen):
            (nbest_ids, nbest_logprobs, beam_from) = self._decode_single_step(
                    encoder_outputs, encoder_output_lengths, target_ids, target_lengths, nbest_logprobs, finished_sel)
            batch_idx = (torch.arange(B)*nbest_keep).view(B, -1).repeat(1, nbest_keep).contiguous().to(beam_from.device)
            batch_beam_from = (batch_idx + beam_from.view(-1, nbest_keep)).view(-1)
            nbest_logprobs = nbest_logprobs.view(-1)
            finished_sel = (nbest_ids.view(-1) == eosid)
            target_ids = target_ids[batch_beam_from]
            target_ids = torch.cat([target_ids, nbest_ids.view(B*nbest_keep, 1)], dim=-1)
            target_lengths += 1

            for i in range(finished_sel.shape[0]):
                if finished_sel[i]:
                    ended.append(target_ids[i])
                    ended_scores.append(nbest_logprobs[i])
                    ended_batch_idx.append(i // nbest_keep)
            target_ids = target_ids * (1 - finished_sel[:, None].long()) # mask out finished

        for i in range(target_ids.shape[0]):
            ended.append(target_ids[i])
            ended_scores.append(nbest_logprobs[i])
            ended_batch_idx.append(i // nbest_keep)

        formated = {}
        for i in range(B):
            formated[i] = []
        for i in range(len(ended)):
            if ended[i][0] == sosid:
                formated[ended_batch_idx[i]].append((ended[i], ended_scores[i]))
        for i in range(B):
            formated[i] = sorted(formated[i], key=lambda x:x[1], reverse=True)[:nbest_keep]

        target_ids = torch.zeros(B, nbest_keep, maxlen+1).to(encoder_outputs.device).long()
        scores = torch.zeros(B, nbest_keep).to(encoder_outputs.device)
        for i in range(B):
            for j in range(nbest_keep):
                item = formated[i][j]
                l = min(item[0].shape[0], target_ids[i, j].shape[0])
                target_ids[i, j, :l] = item[0][:l]
                scores[i, j] = item[1]

        return target_ids, scores

    def _decode_single_step(self, encoder_outputs, encoder_output_lengths, target_ids, target_lengths, accumu_scores, finished_sel=None):
        """
        encoder_outputs: [B, T_e, D_e]
        encoder_output_lengths: [B]
        target_ids: [B*nbest_keep, T_d]
        target_lengths: [B*nbest_keep]
        accumu_scores: [B*nbest_keep]
        """
        B, T_e, D_e = encoder_outputs.shape
        B_d, T_d = target_ids.shape
        if B_d % B != 0:
            raise ValueError("The dim of target_ids does not match the encoder_outputs.")
        nbest_keep = B_d // B
        encoder_outputs = (encoder_outputs.view(B, 1, T_e, D_e)
                .repeat(1, nbest_keep, 1, 1).view(B*nbest_keep, T_e, D_e))
        encoder_output_lengths = (encoder_output_lengths.view(B, 1)
                .repeat(1, nbest_keep).view(-1))

        # outputs: [B, nbest_keep, vocab_size]
        outputs = (self.decoder(encoder_outputs, encoder_output_lengths, target_ids, target_lengths)[:, -1, :])
        vocab_size = outputs.size(-1)
        outputs = outputs.view(B, nbest_keep, vocab_size)
        log_probs = F.log_softmax(outputs, dim=-1)  # [B, nbest_keep, vocab_size]
        if finished_sel is not None:
            log_probs = log_probs.view(B*nbest_keep, -1) - finished_sel.view(B*nbest_keep, -1).float()*9e9
            log_probs = log_probs.view(B, nbest_keep, vocab_size)
        this_accumu_scores = accumu_scores.view(B, nbest_keep, 1) + log_probs
        topk_res = torch.topk(this_accumu_scores.view(B, nbest_keep*vocab_size), k=nbest_keep, dim=-1)

        nbest_logprobs = topk_res[0]  # [B, nbest_keep]
        nbest_ids = topk_res[1] % vocab_size # [B, nbest_keep]
        beam_from = (topk_res[1] / vocab_size).long()

        return nbest_ids, nbest_logprobs, beam_from

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

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
