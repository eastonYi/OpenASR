import os
import io
import logging
import subprocess
import time
import numpy as np
import torch
from collections import defaultdict

from third_party import wavfile

TENSORBOARD_LOGGING = 0


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    demo:
    a ={'length': 10, 'shape': (2,3)}
    config = AttrDict(a)
    config.length #10

    here we can recurrently use attribute to access confis
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        try:
            if type(self[item]) is dict:
                self[item] = AttrDict(self[item])
            res = self[item]
        except:

            print('not found {}'.format(item))
            res = None
        return res


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
    if p.returncode is not 0:
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

def sequence_mask(lengths, maxlen=None, dtype=torch.float):
    if maxlen is None:
        maxlen = lengths.max()
    mask = torch.ones((len(lengths), maxlen),
                      device=lengths.device,
                      dtype=torch.uint8).cumsum(dim=1) <= lengths.unsqueeze(0).t()

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
