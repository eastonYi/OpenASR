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
import json
import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

import utils
import third_party.kaldi_io as kio

IGNORE_ID = -1

SOS_SYM = "<sos>"
EOS_SYM = "<eos>"
UNK_SYM = "<unk>"
BLK_SYM = '<blk>'
SPECIAL_SYM_SET = {SOS_SYM, EOS_SYM, UNK_SYM, BLK_SYM,
                   '[VOCALIZED-NOISE]', '[NOISE]', '[LAUGHTER]'}


class CharTokenizer(object):
    def __init__(self, fn_vocab, add_blk=False):
        with open(fn_vocab, 'r') as f:
            units = f.read().strip().split('\n')
        units = [UNK_SYM, SOS_SYM, EOS_SYM] + units
        if add_blk:
            units += [BLK_SYM]
        self.unit2id = {k:v for v,k in enumerate(units)}
        self.id2unit = units

    def to_id(self, unit):
        return self.unit2id[unit]

    def to_unit(self, id):
        return self.id2unit[id]

    def encode(self, textline):
        return [self.unit2id[char]
            if char in self.unit2id
            else self.unit2id[UNK_SYM]
            for char in list(textline.strip().split())]

    def decode(self, ids, split_token=True, remove_special_sym=True):
        syms = [self.id2unit[i] for i in ids]
        if remove_special_sym:
            syms = [sym for sym in syms if sym not in SPECIAL_SYM_SET]
        if split_token:
            return " ".join(syms)
        return "".join(syms)

    def unit_num(self):
        return len(self.unit2id)


def gen_casual_targets(idslist, maxlen, sos_id, eos_id, add_eos):
    if add_eos:
        ids_with_sym_list = [[sos_id]+ids+[eos_id] for ids in idslist]
    else:
        ids_with_sym_list = [[sos_id]+ids for ids in idslist]

    list_tokens = []
    list_paddings = []
    for b, ids in enumerate(ids_with_sym_list):
        if len(ids) > maxlen:
            logging.warn("ids length {} vs. maxlen {}, cut it.".format(len(ids), maxlen))
        l = min(len(ids), maxlen)
        list_tokens.append(torch.tensor(ids).long()[:l])
        list_paddings.append(torch.zeros(l).long())

    padded_rawids = pad_list(list_tokens, eos_id)
    paddings = pad_list(list_paddings, 1)

    labels = padded_rawids[:, 1:]
    ids = padded_rawids[:, :-1]
    paddings = paddings[:, 1:] # the padding is for labels

    return ids, labels, paddings


def pad_list(xs, pad_value, max_len=None, return_length=False):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    lengths = torch.tensor([x.size(0) for x in xs]).long()
    max_len = lengths.max() if not max_len else max_len
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    if return_length:
        return pad, pad.ne(pad_value).long().sum(-1)
    else:
        return pad


class TextLineByLineDataset(data.Dataset):
    def __init__(self, fn):
        super(TextLineByLineDataset, self).__init__()
        with open(fn, 'r') as f:
            self.data = f.read().strip().split('\n')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SpeechDataset(data.Dataset):
    def __init__(self, data_json_path, reverse=False):
        super().__init__()
        with open(data_json_path, 'rb') as f:
            data = json.load(f)
        self.data = sorted(data, key=lambda x: float(x["duration"]))
        if reverse:
            self.data.reverse()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ArkDataset(SpeechDataset):
    def __init__(self, json_path, reverse=False, rate_in_out=(4,999)):
        try:
            # json_path is a single file
            with open(json_path) as f:
                data = json.load(f)
        except:
            # json_path is a dir where *.json in
            data = []
            for dir, _, fs in os.walk(os.path.dirname(json_path)):   # os.walk获取所有的目录
                for f in fs:
                    if f.endswith('.json'):  # 判断是否是".json"结尾
                        filename = os.path.join(dir, f)
                        with open(filename) as f:
                            data.extend(json.load(f))
        # filter
        if rate_in_out:
            list_to_pop = []
            for i, sample in enumerate(data):
                len_x = sample['feat_length']
                len_y = sample['token_length']
                if len_x < 1 or len_y < 1 or not (rate_in_out[0] <= (len_x / len_y) <= rate_in_out[1]):
                    list_to_pop.append(i)
            list_to_pop.reverse()
            [data.pop(i) for i in list_to_pop]

        self.data = sorted(data, key=lambda x: float(x["feat_length"]))
        if reverse:
            self.data.reverse()


class TimeBasedSampler(Sampler):
    def __init__(self, dataset, duration=200, ngpu=1, shuffle=False): # 200s
        self.dataset = dataset
        self.dur = duration
        self.shuffle = shuffle

        batchs = []
        batch = []
        batch_dur = 0.
        for idx in range(len(self.dataset)):
            batch.append(idx)
            batch_dur += self.dataset[idx]["duration"]
            if batch_dur >= self.dur and len(batch)%ngpu==0:
                # To make the numbers of batchs are equal for each GPU.
                batchs.append(batch)
                batch = []
                batch_dur = 0.
        if batch:
            if len(batch)%ngpu==0:
                batchs.append(batch)
            else:
                b = len(batch)
                batchs.append(batch[b//ngpu*ngpu:])
        self.batchs = batchs

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batchs)
        for b in self.batchs:
            yield b

    def __len__(self):
        return len(self.batchs)


class FrameBasedSampler(TimeBasedSampler):
    def __init__(self, dataset, frames=200, ngpu=1, shuffle=False):
        self.dataset = dataset
        self.frames = frames
        self.shuffle = shuffle

        batchs = []
        batch = []
        batch_frames = 0
        for idx in range(len(self.dataset)):
            batch.append(idx)
            batch_frames += self.dataset[idx]["feat_length"]
            if batch_frames >= self.frames and len(batch)%ngpu==0:
                # To make the numbers of batchs are equal for each GPU.
                batchs.append(batch)
                batch = []
                batch_frames = 0
        if batch:
            if len(batch)%ngpu==0:
                batchs.append(batch)
            else:
                b = len(batch)
                batchs.append(batch[b//ngpu*ngpu:])
        self.batchs = batchs


def load_wave_batch(paths):
    waveforms = []
    lengths = []
    for path in paths:
        sample_rate, waveform = utils.load_wave(path)
        waveform = torch.from_numpy(waveform)
        waveforms.append(waveform)
        lengths.append(waveform.shape[0])
    max_length = max(lengths)
    padded_waveforms = torch.zeros(len(lengths), max_length)
    for i in range(len(lengths)):
        padded_waveforms[i, :lengths[i]] += waveforms[i]
    return padded_waveforms, torch.tensor(lengths).long()


def load_feat_batch(paths):
    features = []
    lengths = []
    for path in paths:
        feature = kio.read_mat(path)
        feature = torch.from_numpy(feature)
        features.append(feature)
        lengths.append(feature.shape[0])
    max_length = max(lengths)
    dim = feature.shape[1]
    padded_features = torch.zeros(len(lengths), max_length, dim)
    for i in range(len(lengths)):
        padded_features[i, :lengths[i], :] += features[i]
    return padded_features, torch.tensor(lengths).long()


class TextCollate(object):
    def __init__(self, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __call__(self, batch):
        timer = utils.Timer()
        timer.tic()
        rawids_list = [self.tokenizer.encode(t) for t in batch]
        ids, labels, paddings = gen_casual_targets(rawids_list, self.maxlen,
                self.tokenizer.to_id(SOS_SYM), self.tokenizer.to_id(EOS_SYM))
        logging.debug("Text Processing Time: {}s".format(timer.toc()))
        return ids, labels, paddings


class WaveCollate(object):
    def __init__(self, tokenizer, maxlen, add_eos=False):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.add_eos = add_eos

    def __call__(self, batch):
        utts = [d["utt"] for d in batch]
        paths = [d["path"] for d in batch]
        trans = [d["transcript"] for d in batch]
        timer = utils.Timer()
        timer.tic()
        padded_waveforms, wave_lengths = load_wave_batch(paths)
        logging.debug("Wave Loading Time: {}s".format(timer.toc()))
        timer.tic()
        rawids_list = [self.tokenizer.encode(t) for t in trans]
        ids, labels, paddings = gen_casual_targets(rawids_list, self.maxlen,
                self.tokenizer.to_id(SOS_SYM), self.tokenizer.to_id(EOS_SYM), self.add_eos)
        logging.debug("Transcription Processing Time: {}s".format(timer.toc()))

        return utts, (padded_waveforms, wave_lengths, ids, labels, paddings)


class FeatureCollate(object):
    def __init__(self, tokenizer, maxlen, add_eos=False, label_type='tokens'):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
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
        ids, labels, paddings = gen_casual_targets(rawids_list, self.maxlen,
                self.tokenizer.to_id(SOS_SYM), self.tokenizer.to_id(EOS_SYM), self.add_eos)
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
        target_in, target_out, paddings = gen_casual_targets(tokens, 999,
                self.tokenizer_char.to_id(SOS_SYM), self.tokenizer_char.to_id(EOS_SYM), self.add_eos)

        return utts, (xs_in, len_xs, target_in, target_out, paddings)


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
        target_in, target_out, paddings = gen_casual_targets(tokens, 999,
                self.tokenizer_char.to_id(SOS_SYM), self.tokenizer_char.to_id(EOS_SYM), self.add_eos)

        return utts, (feats, len_feat, phones, len_phone, target_in, target_out, paddings)


def kaldi_wav_collate(batch):
    utts = [d[0] for d in batch]
    paths = [d[1] for d in batch]
    padded_data, lengths = load_wave_batch(paths)

    return utts, padded_data, lengths


def kaldi_feat_collate(batch):
    utts = [d[0] for d in batch]
    paths = [d[1] for d in batch]
    padded_data, lengths = load_feat_batch(paths)
    return utts, padded_data, lengths


if __name__ == '__main__':
    paths = ['/Users/easton/Projects/OpenASR_BaiYe/egs/callhome_hkust/data/feats_cmvn.ark:102502',
             '/Users/easton/Projects/OpenASR_BaiYe/egs/callhome_hkust/data/feats_cmvn.ark:102502',
             '/Users/easton/Projects/OpenASR_BaiYe/egs/callhome_hkust/data/feats_cmvn.ark:102502',
             '/Users/easton/Projects/OpenASR_BaiYe/egs/callhome_hkust/data/feats_cmvn.ark:24']
    load_feat_batch(paths)
