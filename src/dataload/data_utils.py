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
import numpy as np
import torch
import third_party.kaldi_io as kio

import utils
from dataload.datasets import ArkDataset

IGNORE_ID = -1

SOS_SYM = "<sos>"
EOS_SYM = "<eos>"
UNK_SYM = "<unk>"
BLK_SYM = '<blk>'
SPECIAL_SYM_SET = {SOS_SYM, EOS_SYM, UNK_SYM, BLK_SYM,
                   '[VOCALIZED-NOISE]', '[NOISE]', '[LAUGHTER]'}


class CharTokenizer(object):
    def __init__(self, fn_vocab, add_blk=False):
        units = [UNK_SYM, SOS_SYM, EOS_SYM]
        with open(fn_vocab, 'r') as f:
            for line in f:
                unit = line.strip().split()[0]
                units.append(unit)
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


class SubwordTokenizer(CharTokenizer):
    def __init__(self, fn_vocab, add_blk=False):
        units = [UNK_SYM, SOS_SYM, EOS_SYM]
        with open(fn_vocab, 'r') as f:
            for line in f:
                unit = line.strip().split()[0]
                units.append(unit)
        if add_blk:
            units += [BLK_SYM]
        self.unit2id = {k:v for v,k in enumerate(units)}
        self.id2unit = units

    def decode(self, ids, split_token=True, remove_special_sym=True):
        syms = [self.id2unit[i] for i in ids]
        if remove_special_sym:
            syms = [sym for sym in syms if sym not in SPECIAL_SYM_SET]
        if split_token:
            return " ".join(syms).replace('@@ ')
        return "".join(syms)


def gen_casual_targets(idslist, add_eos, sos_id=1, eos_id=2):
    if add_eos:
        ids_with_sym_list = [[sos_id]+ids+[eos_id] for ids in idslist]
    else:
        ids_with_sym_list = [[sos_id]+ids for ids in idslist]

    list_tokens = []
    list_paddings = []
    for b, ids in enumerate(ids_with_sym_list):
        l = len(ids)
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
        feature = np.copy(kio.read_mat(path))
        feature = torch.tensor(feature)
        features.append(feature)
        lengths.append(feature.shape[0])
    max_length = max(lengths)
    dim = feature.shape[1]
    padded_features = torch.zeros(len(lengths), max_length, dim)
    for i in range(len(lengths)):
        padded_features[i, :lengths[i], :] += features[i]
    return padded_features, torch.tensor(lengths).long()


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
    # paths = ['/Users/easton/Projects/OpenASR_BaiYe/egs/callhome_hkust/data/feats_cmvn.ark:102502',
    #          '/Users/easton/Projects/OpenASR_BaiYe/egs/callhome_hkust/data/feats_cmvn.ark:102502',
    #          '/Users/easton/Projects/OpenASR_BaiYe/egs/callhome_hkust/data/feats_cmvn.ark:102502',
    #          '/Users/easton/Projects/OpenASR_BaiYe/egs/callhome_hkust/data/feats_cmvn.ark:24']
    # load_feat_batch(paths)
    dataset = ArkDataset('/data3/easton/data/CALLHOME_Multilingual/jsons/train',
                         feat_range=(1, 1000), label_range=(0, 50))
    from collections import defaultdict
    counter = defaultdict(lambda: 0)
    threash = 1000
    for sample in dataset:
        if sample['feat_length'] > threash:
            utt = sample['uttid'].split('_')[0]
            counter[utt] += 1
            print(sample['uttid'])
    import pdb; pdb.set_trace()
    print('')
