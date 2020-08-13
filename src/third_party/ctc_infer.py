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
        _, decoded = torch.max(prob_tensor, 2)
        decoded = decoded.view(decoded.size(0), decoded.size(1))

        return self.ctc_reduce_map(decoded, frame_seq_len)


class BeamDecoder(Decoder):
    "Beam search 解码。解码结果为整个序列概率的最大值"
    def __init__(self, beam_width=200, blank_index=0):
        self.beam_width = beam_width
        super().__init__(blank_index=blank_index)

        self._decoder = ctcBeamSearch(beam_width, blank_index=blank_index)

    def decode(self, prob_tensor, frame_seq_len=None):
        '''解码函数
        Args:
            prob_tensor   :   网络模型输出
            frame_seq_len :   每一样本的帧数
        Returns:
            res           :   解码得到的string，即识别结果
        '''
        probs = torch.exp(prob_tensor)
        res = self._decoder.decode(probs, frame_seq_len)
        return res


class BeamEntry:
    "information about one single beam at specific time-step"
    def __init__(self):
        self.prTotal=LOG_ZERO      # blank and non-blank
        self.prNonBlank=LOG_ZERO   # non-blank
        self.prBlank=LOG_ZERO      # blank
        self.y=()                  # labelling at current time-step


class BeamState:
    "information about beams at specific time-step"
    def __init__(self):
        self.entries={}

    def norm(self):
        "length-normalise probabilities to avoid penalising long labellings"
        for (k,v) in self.entries.items():
            labellingLen=len(self.entries[k].y)
            self.entries[k].prTotal=self.entries[k].prTotal*(1.0/(labellingLen if labellingLen else 1))

    def sort(self):
        "return beams sorted by probability"
        u=[v for (k,v) in self.entries.items()]
        s=sorted(u, reverse=True, key=lambda x:x.prTotal)
        return [x.y for x in s]


class ctcBeamSearch(object):
    def __init__(self, beam_width, blank_index=0):
        self.beamWidth = beam_width
        self.blank_index = blank_index

    def log_add_prob(self, log_x, log_y):
        if log_x <= LOG_ZERO:
            return log_y
        if log_y <= LOG_ZERO:
            return log_x
        if (log_y - log_x) > 0.0:
            log_y, log_x = log_x, log_y
        return log_x + math.log(1 + math.exp(log_y - log_x))

    def calcExtPr(self, k, y, t, mat, beamState):
        "probability for extending labelling y to y+k"

        # language model (char bigrams)
        bigramProb=LOG_ONE

        # optical model (RNN)
        if len(y) and y[-1]==k and mat[t-1, self.blank_index] < 0.9:
            return math.log(mat[t, k]) + bigramProb + beamState.entries[y].prBlank
        else:
            return math.log(mat[t, k]) + bigramProb + beamState.entries[y].prTotal

    def addLabelling(self, beamState, y):
        "adds labelling if it does not exist yet"
        if y not in beamState.entries:
            beamState.entries[y]=BeamEntry()

    def decode(self, inputs, inputs_list):
        '''
        mat : FloatTesnor batch * timesteps * class
        '''
        batches, maxT, maxC = inputs.size()
        res = []

        for batch in range(batches):
            mat = inputs[batch].numpy()
            # Initialise beam state
            last=BeamState()
            y=()
            last.entries[y]=BeamEntry()
            last.entries[y].prBlank=LOG_ONE
            last.entries[y].prTotal=LOG_ONE

            # go over all time-steps
            for t in range(inputs_list[batch]):
                curr=BeamState()
                #跳过概率很接近1的blank帧，增加解码速度
                if (1 - mat[t, self.blank_index]) < 0.1:
                    continue
                #取前beam个最好的结果
                BHat=last.sort()[0:self.beamWidth]
                # go over best labellings
                for y in BHat:
                    prNonBlank=LOG_ZERO
                    # if nonempty labelling
                    if len(y)>0:
                        #相同的y两种可能，加入重复或者加入空白,如果之前没有字符，在NonBlank概率为0
                        prNonBlank=last.entries[y].prNonBlank + math.log(mat[t, y[-1]])

                    # calc probabilities
                    prBlank = (last.entries[y].prTotal) + math.log(mat[t, self.blank_index])
                    # save result
                    self.addLabelling(curr, y)
                    curr.entries[y].y=y
                    curr.entries[y].prNonBlank = self.log_add_prob(curr.entries[y].prNonBlank, prNonBlank)
                    curr.entries[y].prBlank = self.log_add_prob(curr.entries[y].prBlank, prBlank)
                    prTotal = self.log_add_prob(prBlank, prNonBlank)
                    curr.entries[y].prTotal = self.log_add_prob(curr.entries[y].prTotal, prTotal)

                    #t时刻加入其它的label,此时Blank的概率为0，如果加入的label与最后一个相同，因为不能重复，所以上一个字符一定是blank
                    for k in range(maxC):
                        if k != self.blank_index:
                            newY=y+(k,)
                            prNonBlank=self.calcExtPr(k, y, t, mat, last)

                            # save result
                            self.addLabelling(curr, newY)
                            curr.entries[newY].y=newY
                            curr.entries[newY].prNonBlank = self.log_add_prob(curr.entries[newY].prNonBlank, prNonBlank)
                            curr.entries[newY].prTotal = self.log_add_prob(curr.entries[newY].prTotal, prNonBlank)

                # set new beam state
                last=curr

            BHat=last.sort()[0:self.beamWidth]
            # go over best labellings
            curr = BeamState()
            for y in BHat:
                newY = y
                prNonBlank = last.entries[newY].prTotal
                self.addLabelling(curr, newY)
                curr.entries[newY].y=newY
                curr.entries[newY].prNonBlank = self.log_add_prob(curr.entries[newY].prNonBlank, prNonBlank)
                curr.entries[newY].prTotal = self.log_add_prob(curr.entries[newY].prTotal, prNonBlank)

            last = curr
            # normalise probabilities according to labelling length
            last.norm()

            # sort by probability
            bestLabelling=last.sort()[0] # get most probable labelling

            res.append(bestLabelling)

        return res


if __name__ == '__main__':
    decoder = Decoder('abcde', 1, 2)
    print(decoder._convert_to_strings([[1,2,1,0,3],[1,2,1,1,1]]))
