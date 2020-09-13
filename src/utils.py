import os
import io
import logging
import subprocess
import time
import numpy as np
import torch
from collections import defaultdict
import soundfile as sf
import editdistance as ed

from third_party import wavfile

TENSORBOARD_LOGGING = 0

def cycle(iterable):
    while True:
        print('new loop on', iterable)
        for x in iterable:
            yield x


def load_vocab(path, vocab_size=None):
    with open(path, encoding='utf8') as f:
        vocab = [line.strip().split()[0] for line in f]
    vocab = vocab[:vocab_size] if vocab_size else vocab
    id_unk = vocab.index('<unk>')
    token2idx = defaultdict(lambda: id_unk)
    idx2token = defaultdict(lambda: '<unk>')
    token2idx.update({token: idx for idx, token in enumerate(vocab)})
    idx2token.update({idx: token for idx, token in enumerate(vocab)})
    idx2token[token2idx['<pad>']] = ''
    idx2token[token2idx['<blk>']] = ''
    idx2token[token2idx['<unk>']] = '<UNK>'
    idx2token[token2idx['<sos>']] = ''
    idx2token[token2idx['<eos>']] = ''

    assert len(token2idx) == len(idx2token)

    return token2idx, idx2token


def cleanup_ckpt(expdir, num_last_ckpt_keep):
    ckptlist = [t for t in os.listdir(expdir) if t.endswith('.pt') and t != 'last.pt']
    ckptlist = sorted(ckptlist)
    ckptlist_rm = ckptlist[:-num_last_ckpt_keep]
    logging.info("Clean up checkpoints. Remain the last {} checkpoints.".format(num_last_ckpt_keep))
    for name in ckptlist_rm:
       os.remove(os.path.join(expdir, name))


def get_command_stdout(command, require_zero_status=True):
    """ Executes a command and returns its stdout output as a string.  The
        command is executed with shell=True, so it may contain pipes and
        other shell constructs.

        If require_zero_stats is True, this function will raise an exception if
        the command has nonzero exit status.  If False, it just prints a warning
        if the exit status is nonzero.

        See also: execute_command, background_command
    """
    p = subprocess.Popen(command, shell=True,
                         stdout=subprocess.PIPE)

    stdout = p.communicate()[0]
    if p.returncode != 0:
        output = "Command exited with status {0}: {1}".format(
            p.returncode, command)
        if require_zero_status:
            raise Exception(output)
        else:
            logging.warning(output)
    return stdout


def load_wave(path):
    """
    path can be wav filename or pipeline
    """
    # parse path
    items = path.strip().split(":", 1)
    if len(items) != 2:
        raise ValueError("Unknown path format.")
    tag = items[0]
    path = items[1]
    if tag == "file":
        sample_rate, data = wavfile.read(path)
    elif tag == "pipe":
        path = path[:-1]
        out = get_command_stdout(path, require_zero_status=True)
        sample_rate, data = wavfile.read(io.BytesIO(out))
    elif tag == "ark":
        fn, offset = path.split(":", 1)
        offset = int(offset)
        with open(fn, 'rb') as f:
            f.seek(offset)
            sample_rate, data = wavfile.read(f, offset=offset)
    elif tag == 'flac':
        data, sample_rate = sf.read(path)
    else:
        raise ValueError("Unknown file tag.")
    data = data.astype(np.float32)
    return sample_rate, data


def parse_scp(fn):
    dic = {}
    with open(fn, 'r') as f:
        cnt = 0
        for line in f:
            cnt += 1
            items = line.strip().split(' ', 1)
            if len(items) != 2:
                logging.warning('Wrong formated line {} in scp {}, skip it.'.format(cnt, fn))
                continue
            dic[items[0]] = items[1]
    return dic


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise 'Unsupported value encountered.'


def batch_distance(hpys, len_hyps, refs, len_refs):
    total_dist = 0
    for hpy, len_hyp, ref, len_ref in zip(hpys, len_hyps, refs, len_refs):
        total_dist += ed.eval(hpy[:len_hyp], ref[:len_ref])

    return total_dist


class Timer(object):
    def __init__(self):
        self.start = 0.

    def tic(self):
        self.start = time.time()

    def toc(self):
        return time.time() - self.start

# ==========================================
# auxilary functions for sequence
# ==========================================

def freeze(module):
    for name, param in module.named_parameters():
        param.requires_grad = False
        print('{}\tfreezed'.format(name))


def sequence_mask(lengths, maxlen=None, depth=None, dtype=torch.float):
    if maxlen is None:
        maxlen = lengths.max()
    mask = torch.ones((len(lengths), maxlen),
                      device=lengths.device,
                      dtype=torch.uint8).cumsum(dim=1) <= lengths.unsqueeze(0).t()
    if depth:
        mask = mask.unsqueeze(-1).repeat(1, 1, depth)

    return mask.type(dtype)


def get_paddings(src, lengths):
    paddings = torch.zeros_like(src).to(src.device)
    for b in range(lengths.shape[0]):
        paddings[b, lengths[b]:, :] = 1
    return paddings


def get_paddings_by_shape(shape, lengths, device="cpu"):
    paddings = torch.zeros(shape).to(device)
    if shape[0] != lengths.shape[0]:
        raise ValueError("shape[0] does not match lengths.shape[0]:"
            " {} vs. {}".format(shape[0], lengths.shape[0]))
    T = shape[1]
    for b in range(shape[0]):
        if lengths[b] < T:
            l = lengths[b]
            paddings[b, l:] = 1
    return paddings


def get_transformer_padding_byte_masks(B, T, lengths):
    masks = get_paddings_by_shape([B, T], lengths).bool()
    return masks


def get_transformer_casual_masks(T):
    masks = -torch.triu(
            torch.ones(T, T), diagonal=1)*9e20
    return masks

# == ctc related ==

def ctc_reduce(align):
    tmp = None
    res = []
    for i in align:
        if i != tmp:
            res.append(i)
            tmp = i
    return res


def ctc_shrink(logits, pad, blk):
    """only count the first one for the repeat freams
    """
    device = logits.device
    B, T, V = logits.size()
    tokens = torch.argmax(logits, -1)
    # intermediate vars along time
    list_fires = []
    token_prev = torch.ones(B).to(device) * -1
    blk_batch = torch.ones(B).to(device) * blk
    pad_batch = torch.ones(B).to(device) * pad

    for t in range(T):
        token = tokens[:, t]
        fire_place = torch.logical_and(token != blk_batch, token != token_prev)
        fire_place = torch.logical_and(fire_place, token != pad_batch)
        list_fires.append(fire_place)
        token_prev = token

    fires = torch.stack(list_fires, 1)
    len_decode = fires.sum(-1)
    max_decode_len = len_decode.max()
    list_ls = []

    for b in range(B):
        l = logits[b, :, :].index_select(0, torch.where(fires[b])[0])
        pad_l = torch.zeros([max_decode_len - l.size(0), V]).to(device)
        list_ls.append(torch.cat([l, pad_l], 0))

    logits_shrunk = torch.stack(list_ls, 0)

    return logits_shrunk, len_decode


def ctc_decode_fn(units, beam_width, blank_id, num_processes=2):
    from ctcdecode import CTCBeamDecoder

    fn = CTCBeamDecoder(units,
                        beam_width=beam_width,
                        blank_id=blank_id,
                        num_processes=num_processes)
    return fn
