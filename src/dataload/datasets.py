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
    def __init__(self, data_file, reverse=False,
                 feat_range=(1, 99999), label_range=(1, 100), rate_in_out=(4,99999)):
        super().__init__()

        if data_file.endswith('.flist'):  # 判断是否是".json"结尾
            self.data = self.load_flist(data_file,
                x_range=feat_range)

        elif data_file.endswith('.json'):  # 判断是否是".json"结尾
            self.data = self.load_json(data_file,
                x_range=feat_range, y_range=label_range, rate=rate_in_out)

        self.data = sorted(data, key=lambda x: x['feat_length'])
        if reverse:
            self.data.reverse()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_flist(data_file, x='feat_length', x_range=(1,9999)):
        data = []
        with open(data_file) as f:
            for i, line in enumerate(f):
                f_path, duration = line.strip().split()
                sample = {'uttid': i, 'path': f_path, 'feat_length': int(duration)}
                data.append(sample)

        list_to_pop = []
        for i, sample in enumerate(data):
            len_x = sample[x]
            if not (x_range[0] <= len_x <= x_range[1]):
                list_to_pop.append(i)

        # filter
        print('filtered {}/{} samples\n'.format(len(list_to_pop), len(data)))
        list_to_pop.reverse()
        [data.pop(i) for i in list_to_pop]

        return data

    @staticmethod
    def load_json(json_path,
                  x='feat_length', y='token_length',
                  x_range=(1, 9999), y_range=(1,999), rate=(1,99)):
        # json
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

        list_to_pop = []
        for i, sample in enumerate(data):
            len_x = sample[x]
            len_y = sample[y]
            if not (x_range[0] <= len_x <= x_range[1]) or \
               not (y_range[0] <= len_y <= y_range[1]) or \
               not (rate[0] <= (len_x / len_y) <= rate[1]):
                list_to_pop.append(i)

        # filter
        print('filtered {}/{} samples\n'.format(len(list_to_pop), len(data)))
        list_to_pop.reverse()
        [data.pop(i) for i in list_to_pop]

        return data


class ArkDataset(SpeechDataset):
    def __init__(self, json_path, reverse=False,
                 feat_range=(1, 99999), label_range=(1, 100), rate_in_out=(4,999)):
        data = self.load_json(json_path,
            x_range=feat_range, y_range=label_range, rate=rate_in_out)

        self.data = sorted(data, key=lambda x: float(x['feat_length']))
        if reverse:
            self.data.reverse()


class PhoneCharDataset(SpeechDataset):
    def __init__(self, json_path, sort=True, reverse=False, multi=1,
                 feat_range=(1, 99999), label_range=(1, 100), rate_in_out=(2,999)):
        data = self.load_json(json_path,
            x='phone_length', x_range=feat_range, y_range=label_range, rate=rate_in_out)

        if sort:
            self.data = sorted(data, key=lambda x: float(x['phone_length']))
            if reverse:
                self.data.reverse()
        if multi > 1:
            self.data *= multi


class TokenDataset(SpeechDataset):
    def __init__(self, phone_path, multi=1):
        # phone
        self.data = self.load_tokens(phone_path)
        if multi > 1:
            self.data *= multi

    @staticmethod
    def load_tokens(token_file):
        list_tokens = []
        with open(token_file) as f:
            for i, line in enumerate(f):
                try:
                    tokens = line.strip().split(maxsplit=1)[1]
                    list_tokens.append(tokens)
                except:
                    pass
        print('load {}/{} samples from {}\n'.format(len(list_tokens), i+1, token_file))

        return list_tokens


class Semi_PhoneCharDataset(PhoneCharDataset):
    def __init__(self, phone_path, text_path, json_path,
                 feat_range=(1, 99999), label_range=(1, 100), rate_in_out=(2,999)):
        """
        phone       : uttid a b c
        text        : uttid m1 m2 m3
        json_path   :
        """
        data = self.load_json(json_path,
            x='phone_length', x_range=feat_range, y_range=label_range, rate=rate_in_out)

        self.data = sorted(data, key=lambda x: float(x['phone_length']))

        # phone
        self.phone_data = self.load_tokens(phone_path)

        # text
        self.text_data = self.load_tokens(text_path)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return {'paired': len(self.data),
                'text': len(self.text_data),
                'phone': len(self.phone_data)}
