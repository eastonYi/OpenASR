#/usr/bin/python
#encoding=utf-8

#greedy decoder and beamsearch decoder for ctc

import torch
import numpy as np
import math

LOG_ZERO = -1e-9
LOG_ONE = 0.0


class Decoder(object):
    "解码器基类定义，作用是将模型的输出转化为文本使其能够与标签计算正确率"
    def __init__(self, blank_index=0):
        '''
        blank_index  :     空白类的索引，默认设置为0
        '''
        self.blank_index = blank_index

    def __call__(self, prob_tensor, frame_seq_len=None):
        return self.decode(prob_tensor, frame_seq_len)

    def decode(self):
        "解码函数，在GreedyDecoder和BeamDecoder继承类中实现"
        raise NotImplementedError;

    def ctc_reduce_map(self, batch_samples, lengths):
        """
        inputs:
            batch_samples: size x time
        return:
            (padded_samples, mask): (size x max_len, size x max_len)
                                     max_len <= time
        """
        sents = []
        for align, length in zip(batch_samples, lengths):
            sent = []
            tmp = None
            for token in align[:length]:
                if token != self.blank_index and token != tmp:
                    sent.append(token)
                tmp = token
            sents.append(sent)

        return self.padding_list_seqs(sents, dtype=np.int32, pad=0)

    def padding_list_seqs(self, list_seqs, dtype=np.float32, pad=0.):
        len_x = [len(s) for s in list_seqs]

        size_batch = len(list_seqs)
        maxlen = max(len_x)

        shape_feature = tuple()
        for s in list_seqs:
            if len(s) > 0:
                shape_feature = np.asarray(s).shape[1:]
                break

        x = (np.ones((size_batch, maxlen) + shape_feature) * pad).astype(dtype)
        for idx, s in enumerate(list_seqs):
            x[idx, :len(s)] = s

        return x, len_x


class GreedyDecoder(Decoder):
    "直接解码，把每一帧的输出概率最大的值作为输出值，而不是整个序列概率最大的值"
    def decode(self, prob_tensor, frame_seq_len):
        '''解码函数
        Args:
            prob_tensor   :   网络模型输出
            frame_seq_len :   每一样本的帧数
        Returns:
            解码得到的string，即识别结果
        '''
        _, decoded = torch.max(prob_tensor, -1)

        return self.ctc_reduce_map(decoded, frame_seq_len)


class BeamDecoder(Decoder):
    "Beam search 解码。解码结果为整个序列概率的最大值"
    def __init__(self, beam_width=200, blank_index=0):
        super().__init__(blank_index=blank_index)
        # import ctcdecode

        self.beam_width = beam_width
        self.blank_index = blank_index
        # self.decode = ctcdecode.CTCBeamDecoder(
        #     self.vocab_list, beam_width=beam_width,
        #     blank_id=blank_index, num_processes=5)

    def decode(self, logits, frame_seq_len=None):
        '''解码函数
        Args:
            prob_tensor   :   网络模型输出
            frame_seq_len :   每一样本的帧数
        Returns:
            res           :   解码得到的string，即识别结果
        '''
        log_probs = torch.log_softmax(logits, -1)
        # beam_result, beam_scores, timesteps, out_seq_len = self.decode.decode(distributions)
        # res = self._decoder.decode(distributions, frame_seq_len)
        for log_prob, length in zip(log_probs, frame_seq_len):
            import pdb; pdb.set_trace()
            res = ctcBeamSearch(log_prob[:length])

        return res


class BeamEntry:
	"information about one single beam at specific time-step"
	def __init__(self):
		self.prTotal = 0 # blank and non-blank
		self.prNonBlank = 0 # non-blank
		self.prBlank = 0 # blank
		self.prText = 1 # LM score
		self.lmApplied = False # flag if LM was already applied to this beam
		self.labeling = () # beam-labeling


class BeamState:
	"information about the beams at specific time-step"
	def __init__(self):
		self.entries = {}

	def norm(self):
		"length-normalise LM score"
		for (k, _) in self.entries.items():
			labelingLen = len(self.entries[k].labeling)
			self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

	def sort(self):
		"return beam-labelings, sorted by probability"
		beams = [v for (_, v) in self.entries.items()]
		sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
		return [x.labeling for x in sortedBeams]


def addBeam(beamState, labeling):
	"add beam if it does not yet exist"
	if labeling not in beamState.entries:
		beamState.entries[labeling] = BeamEntry()


def ctcBeamSearch(mat, blank_index=0, beamWidth=25):
	"beam search as described by the paper of Hwang et al. and the paper of Graves et al."
	maxT, maxC = mat.shape

	# initialise beam state
	last = BeamState()
	labeling = ()
	last.entries[labeling] = BeamEntry()
	last.entries[labeling].prBlank = 1
	last.entries[labeling].prTotal = 1

	# go over all time-steps
	for t in range(maxT):
		curr = BeamState()

		# get beam-labelings of best beams
		bestLabelings = last.sort()[0:beamWidth]

		# go over best beams
		for labeling in bestLabelings:

			# probability of paths ending with a non-blank
			prNonBlank = 0
			# in case of non-empty beam
			if labeling:
				# probability of paths with repeated last char at the end
				prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

			# probability of paths ending with a blank
			prBlank = (last.entries[labeling].prTotal) * mat[t, blank_index]

			# add beam at current time-step if needed
			addBeam(curr, labeling)

			# fill in data
			curr.entries[labeling].labeling = labeling
			curr.entries[labeling].prNonBlank += prNonBlank
			curr.entries[labeling].prBlank += prBlank
			curr.entries[labeling].prTotal += prBlank + prNonBlank
			curr.entries[labeling].prText = last.entries[labeling].prText # beam-labeling not changed, therefore also LM score unchanged from
			curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling

			# extend current beam-labeling
			for c in range(maxC - 1):
				# add new char to current beam-labeling
				newLabeling = labeling + (c,)

				# if new labeling contains duplicate char at the end, only consider paths ending with a blank
				if labeling and labeling[-1] == c:
					prNonBlank = mat[t, c] * last.entries[labeling].prBlank
				else:
					prNonBlank = mat[t, c] * last.entries[labeling].prTotal

				# add beam at current time-step if needed
				addBeam(curr, newLabeling)

				# fill in data
				curr.entries[newLabeling].labeling = newLabeling
				curr.entries[newLabeling].prNonBlank += prNonBlank
				curr.entries[newLabeling].prTotal += prNonBlank

		# set new beam state
		last = curr

	# normalise LM scores according to beam-labeling-length
	last.norm()

	 # sort by probability
	bestLabeling = last.sort()[0] # get most probable labeling

	return bestLabeling


if __name__ == '__main__':
    decoder = Decoder('abcde', 1, 2)
    print(decoder._convert_to_strings([[1,2,1,0,3],[1,2,1,1,1]]))
