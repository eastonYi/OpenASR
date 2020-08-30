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

from third_party import kaldi_signal as ksp


class SPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_type = config["feature_type"]
        if "spec_aug" in config:
            self.spec_aug_conf = {
                "freq_mask_num": config["spec_aug"]["freq_mask_num"],
                "freq_mask_width": config["spec_aug"]["freq_mask_width"],
                "time_mask_num": config["spec_aug"]["time_mask_num"],
                "time_mask_width": config["spec_aug"]["time_mask_width"],
                }

        self.num_ceps = None
        if self.feature_type == "offline":
            feature_func = None
        elif self.feature_type == "fbank":
            def feature_func(waveform):
                return ksp.fbank(
                    waveform,
                    sample_frequency=float(config["sample_rate"]),
                    use_energy=config["use_energy"],
                    num_mel_bins=int(config["num_mel_bins"])
                 )
        else:
            raise ValueError("Unknown feature type.")
        self.func = feature_func

    def spec_aug(self, padded_features, feature_lengths):
        freq_means = torch.mean(padded_features, dim=-1)
        time_means = (torch.sum(padded_features, dim=1)
                /feature_lengths[:, None].float()) # Note that features are padded with zeros.

        B, T, V = padded_features.shape
        # mask freq
        for _ in range(self.spec_aug_conf["freq_mask_num"]):
            fs = (self.spec_aug_conf["freq_mask_width"]*torch.rand(size=[B],
                device=padded_features.device, requires_grad=False)).long()
            f0s = ((V-fs).float()*torch.rand(size=[B],
                device=padded_features.device, requires_grad=False)).long()
            for b in range(B):
                padded_features[b, :, f0s[b]:f0s[b]+fs[b]] = freq_means[b][:, None]

        # mask time
        for _ in range(self.spec_aug_conf["time_mask_num"]):
            ts = (self.spec_aug_conf["time_mask_width"]*torch.rand(size=[B],
                device=padded_features.device, requires_grad=False)).long()
            t0s = ((feature_lengths-ts).float()*torch.rand(size=[B],
                device=padded_features.device, requires_grad=False)).long()
            for b in range(B):
                padded_features[b, t0s[b]:t0s[b]+ts[b], :] = time_means[b][None, :]
        return padded_features, feature_lengths

    def forward(self, wav_batch, lengths):
        batch_size, batch_length = wav_batch.shape[0], wav_batch.shape[1]
        if self.func is not None:
            features = []
            feature_lengths = []
            for i in range(batch_size):
                feature = self.func(wav_batch[i, :lengths[i]].view(1, -1))
                features.append(feature)
                feature_lengths.append(feature.shape[0])

            # pad to max_length
            max_length = max(feature_lengths)
            padded_features = torch.zeros(batch_size, max_length, feature.shape[-1]).to(feature.device)
            for i in range(batch_size):
                l = feature_lengths[i]
                padded_features[i, :l, :] += features[i]
        else:
            padded_features = wav_batch
            feature_lengths = lengths

        feature_lengths = feature_lengths.long().to(padded_features.device)

        if self.training and self.spec_aug_conf is not None:
            padded_features, feature_lengths = self.spec_aug(padded_features, feature_lengths)

        return padded_features, feature_lengths


class WavConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.d_model = config["d_model"]
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(1, self.d_model, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats, feat_lengths):
        len_x = feat_lengths // 160
        x = self.encoder(feats.unsqueeze(1)).transpose(1, 2)
        x = x[:, :len_x.max(), :]

        return x, len_x
