"""
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

from .data_utils import gen_casual_targets, load_wave_batch, load_feat_batch, pad_list
import utils


class TextCollate(object):
    def __init__(self, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __call__(self, batch):
        timer = utils.Timer()
        timer.tic()
        rawids_list = [self.tokenizer.encode(t) for t in batch]
        ids, labels, paddings = gen_casual_targets(rawids_list)
        logging.debug("Text Processing Time: {}s".format(timer.toc()))
        return ids, labels, paddings


def waveCollate(batch):
    utts = [d["uttid"] for d in batch]
    paths = ['flac:' + d["path"] for d in batch]
    padded_waveforms, wave_lengths = load_wave_batch(paths)

    return utts, (padded_waveforms, wave_lengths)


class WaveSampleCollate(object):
    def __init__(self, tokenizer, add_eos=False, label_type='tokens'):
        self.tokenizer = tokenizer
        self.label_type = label_type
        self.add_eos = add_eos

    def __call__(self, batch):
        utts = [d["uttid"] for d in batch]
        paths = ['flac:' + d["feat"] for d in batch]
        if self.label_type == 'tokens':
            trans = [d["tokens"] for d in batch]
        elif self.label_type == 'phones':
            trans = [d["phones"] for d in batch]
        timer = utils.Timer()
        timer.tic()
        padded_waveforms, wave_lengths = load_wave_batch(paths)
        logging.debug("Wave Loading Time: {}s".format(timer.toc()))
        timer.tic()
        rawids_list = [self.tokenizer.encode(t) for t in trans]
        ids, labels, paddings = gen_casual_targets(rawids_list, self.add_eos)
        logging.debug("Transcription Processing Time: {}s".format(timer.toc()))

        return utts, (padded_waveforms, wave_lengths, ids, labels, paddings)


class FeatureCollate(object):
    def __init__(self, tokenizer, add_eos=False, label_type='tokens'):
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.label_type = label_type

    def __call__(self, batch):
        utts = [d["uttid"] for d in batch]
        paths = [d["feat"] for d in batch]
        if self.label_type == 'tokens':
            trans = [d["tokens"] for d in batch]
        elif self.label_type == 'phones':
            trans = [d["phones"] for d in batch]
        else:
            raise NotImplementedError(self.label_type)
        timer = utils.Timer()
        timer.tic()
        padded_features, feature_lengths = load_feat_batch(paths)
        logging.debug("Feature Loading Time: {}s".format(timer.toc()))
        timer.tic()
        rawids_list = [self.tokenizer.encode(t) for t in trans]
        ids, labels, paddings = gen_casual_targets(rawids_list, self.add_eos)
        logging.debug("Transcription Processing Time: {}s".format(timer.toc()))

        return utts, (padded_features, feature_lengths, ids, labels, paddings)


class Phone_Char_Collate(object):
    def __init__(self, tokenizer_phone, tokenizer_char, add_eos=False):
        self.tokenizer_phone = tokenizer_phone
        self.tokenizer_char = tokenizer_char
        self.add_eos = add_eos

    def __call__(self, batch):
        utts = [d["uttid"] for d in batch]

        phones = [torch.tensor(self.tokenizer_phone.encode(d["phones"])).long() for d in batch]
        xs_in, len_xs = pad_list(phones, pad_value=0, return_length=True)

        tokens = [self.tokenizer_char.encode(d["tokens"]) for d in batch]
        target_in, target_out, paddings = gen_casual_targets(tokens, self.add_eos)

        return utts, (xs_in, len_xs, target_in, target_out, paddings)


class Feat_Phone_Collate(Phone_Char_Collate):
    def __init__(self, tokenizer_phone):
        self.tokenizer_phone = tokenizer_phone

    def __call__(self, batch):
        utts = [d["uttid"] for d in batch]
        paths = [d["feat"] for d in batch]
        feats, len_feat = load_feat_batch(paths)
        phones = [torch.tensor(self.tokenizer_phone.encode(d["phones"])).long() for d in batch]
        phones, len_phone = pad_list(phones, pad_value=0, return_length=True)

        return utts, (feats, len_feat, phones, len_phone)


class Feat_Phone_Char_Collate(Phone_Char_Collate):
    def __init__(self, tokenizer_phone, tokenizer_char, add_eos=False):
        self.tokenizer_phone = tokenizer_phone
        self.tokenizer_char = tokenizer_char
        self.add_eos = add_eos

    def __call__(self, batch):
        utts = [d["uttid"] for d in batch]
        paths = [d["feat"] for d in batch]
        feats, len_feat = load_feat_batch(paths)
        phones = [torch.tensor(self.tokenizer_phone.encode(d["phones"])).long() for d in batch]
        phones, len_phone = pad_list(phones, pad_value=0, return_length=True)

        tokens = [self.tokenizer_char.encode(d["tokens"]) for d in batch]
        target_in, target_out, paddings = gen_casual_targets(tokens, self.add_eos)

        return utts, (feats, len_feat, phones, len_phone, target_in, target_out, paddings)
