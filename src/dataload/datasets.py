import json
import os
import torch.utils.data as data


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
    def __init__(self, json_path, reverse=False,
                 feat_range=(1, 99999), label_range=(1, 100), rate_in_out=(4,999)):
        try:
            # json_path is a single file
            with open(json_path) as f:
                data = json.load(f)
        except:
            # json_path is a dir where *.json in
            data = []
            for dir, _, fs in os.walk(json_path):   # os.walk获取所有的目录
                for f in fs:
                    if f.endswith('.json'):  # 判断是否是".json"结尾
                        filename = os.path.join(dir, f)
                        print('loading json file :', filename)
                        with open(filename) as f:
                            add = json.load(f)
                            data.extend(add)
                        print('loaded {} samples'.format(len(add)))

        # filter
        list_to_pop = []
        for i, sample in enumerate(data):
            len_x = sample['feat_length']
            len_y = sample['token_length']
            if not (feat_range[0] <= len_x <= feat_range[1]) or \
               not (label_range[0] <= len_y <= label_range[1]) or \
               not (rate_in_out[0] <= (len_x / len_y) <= rate_in_out[1]):
                list_to_pop.append(i)
        print('filtered {}/{} samples\n'.format(len(list_to_pop), len(data)))
        list_to_pop.reverse()
        [data.pop(i) for i in list_to_pop]

        self.data = sorted(data, key=lambda x: float(x["feat_length"]))
        if reverse:
            self.data.reverse()
