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
import os
import argparse
import logging
import yaml
import torch
from torch.utils.data import DataLoader

import utils
from dataload import datasets, collates, samplers, data_utils

if "LAS_LOG_LEVEL" in os.environ:
    LOG_LEVEL = os.environ["LAS_LOG_LEVEL"]
else:
    LOG_LEVEL = "INFO"
if LOG_LEVEL == "DEBUG":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
else:
     logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def get_args():
    parser = argparse.ArgumentParser(description="""
     Usage: train.py <config>""")
    parser.add_argument("config", help="path to config file")
    parser.add_argument('--continue-training', type=utils.str2bool, default=False,
                        help='Continue training from last_model.pt.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    timer = utils.Timer()

    args = get_args()
    timer.tic()
    config = utils.AttrDict(yaml.load(open(args.config)))
    dataconfig = config["data"]
    trainingconfig = config["training"]
    modelconfig = config["model"]

    feat_range = [int(i) for i in dataconfig['feat_range'].split(',')]
    label_range = [int(i) for i in dataconfig['label_range'].split(',')]

    ngpu = 1
    if "multi_gpu" in trainingconfig and trainingconfig["multi_gpu"] == True:
        ngpu = torch.cuda.device_count()

    tokenizer_phone = data_utils.CharTokenizer(dataconfig["vocab_phone"], add_blk=True)
    tokenizer_char = data_utils.CharTokenizer(dataconfig["vocab_char"], add_blk=modelconfig['add_blk'])

    # model type
    if modelconfig['type'] == 'Embed_Decoder':
        from frameworks.Text_Models import Embed_Decoder as Model

        modelconfig["encoder"]["vocab_size"] = tokenizer_phone.unit_num()
        modelconfig["decoder"]["vocab_size"] = tokenizer_char.unit_num()
        model = Model.create_model(modelconfig["encoder"], modelconfig["decoder"])

    elif modelconfig['type'] == 'Conv_CTC_Transformer':
        from frameworks.Speech_Models import Conv_CTC as Model

        modelconfig["decoder"]["vocab_size"] = tokenizer_char.unit_num()
        model = Model.create_model(modelconfig["signal"],
                                   modelconfig["encoder"],
                                   tokenizer_phone.unit_num(),
                                   modelconfig["decoder"])

    elif modelconfig['type'] == 'CIF_MIX':
        from frameworks.Speech_Models import CIF_MIX as Model

        modelconfig["decoder"]["vocab_size"] = tokenizer_char.unit_num()
        model = Model.create_model(modelconfig["signal"],
                                   modelconfig["encoder"],
                                   modelconfig["assigner"],
                                   tokenizer_phone.unit_num(),
                                   modelconfig["decoder"])

    # solver type
    if trainingconfig['type'] == 'phone2char':
        from solvers import Phone2Char_Solver as Solver

        acoustic_set = datasets.ArkDataset(dataconfig["acoustic"], feat_range=feat_range, label_range=label_range)
        training_set = datasets.ArkDataset(dataconfig["trainset"], feat_range=feat_range, label_range=label_range)
        valid_set = datasets.ArkDataset(dataconfig["devset"], reverse=True)

        collate = collates.Phone_Char_Collate(tokenizer_phone, tokenizer_char, modelconfig["add_eos"])
        tr_loader = DataLoader(training_set,
            collate_fn=collate, batch_size=trainingconfig['batch_size'], shuffle=False, num_workers=2)
        cv_loader = DataLoader(valid_set,
            collate_fn=collate, batch_size=trainingconfig['batch_size'], shuffle=False, num_workers=1)

        solver = Solver(model, trainingconfig, tr_loader, cv_loader)

    elif trainingconfig['type'] == 'mix':
        from solvers import CIF_MIX_Solver as Solver

        acoustic_set = datasets.ArkDataset(
            dataconfig["acoustic"], feat_range=feat_range, label_range=label_range)
        training_set = datasets.ArkDataset(
            dataconfig["trainset"], feat_range=feat_range, label_range=label_range)
        valid_set = datasets.ArkDataset(
            dataconfig["devset"], reverse=True)

        collate_acoustic = collates.Feat_Phone_Collate(tokenizer_phone)
        sampler_acoustic = samplers.FrameBasedSampler(
            acoustic_set, trainingconfig["batch_acoustic_frames"]*ngpu, ngpu, shuffle=True)
        batchiter_acoustic = DataLoader(
            acoustic_set, collate_fn=collate_acoustic, batch_sampler=sampler_acoustic,
            shuffle=False, num_workers=dataconfig["fetchworker_num"])

        collate = collates.Feat_Phone_Char_Collate(
            tokenizer_phone, tokenizer_char, modelconfig["add_eos"])
        sampler_training = samplers.FrameBasedSampler(
            training_set, trainingconfig["batch_frames"]*ngpu, ngpu, shuffle=True)
        sampler_valid = samplers.FrameBasedSampler(
            valid_set, trainingconfig["batch_frames"]*ngpu, ngpu, shuffle=False) # for plot longer utterance
        batchiter_train = DataLoader(training_set,
            collate_fn=collate, batch_sampler=sampler_training, shuffle=False,
            num_workers=1)
        batchiter_dev = DataLoader(valid_set,
            collate_fn=collate, batch_sampler=sampler_valid, shuffle=False,
            num_workers=dataconfig["fetchworker_num"])

        solver = Solver(model, trainingconfig, batchiter_acoustic, batchiter_train, batchiter_dev)

    logging.info("\nModel info:\n{}".format(model))

    if args.continue_training:
        logging.info("Load package from {}.".format(os.path.join(trainingconfig["exp_dir"], "last.pt")))
        pkg = torch.load(os.path.join(trainingconfig["exp_dir"], "last.pt"))
        model.restore(pkg["model"])

    if torch.cuda.is_available():
        model = model.cuda()

    if args.continue_training:
        logging.info("Restore solver states...")
        solver.restore(pkg)
    logging.info("Start training...")
    solver.train()
    logging.info("Total time: {:.4f} secs".format(timer.toc()))
