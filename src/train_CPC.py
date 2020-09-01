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
from dataload import datasets, samplers, data_utils, collates


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
    parser.add_argument('--type', type=str, default='pretrain',
                        help='Continue training from last_model.pt.')
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

    ngpu = 1
    if "multi_gpu" in trainingconfig and trainingconfig["multi_gpu"] == True:
        ngpu = torch.cuda.device_count()

    if args.type == 'pretrain':
        training_set = datasets.SpeechDataset(dataconfig["trainset"], feat_range=feat_range)
        valid_set = datasets.SpeechDataset(dataconfig["devset"], reverse=True, feat_range=feat_range)
        trainingsampler = samplers.TimeBasedSampler(training_set, trainingconfig["batch_time"]*ngpu, ngpu, shuffle=True)
        validsampler = samplers.TimeBasedSampler(valid_set, trainingconfig["batch_time"]*ngpu, ngpu, shuffle=False) # for plot longer utterance

        tr_loader = DataLoader(training_set, collate_fn=collates.waveCollate,
                               batch_sampler=trainingsampler, shuffle=False,
                               num_workers=dataconfig["fetchworker_num"])
        cv_loader = DataLoader(valid_set, collate_fn=collates.waveCollate,
                               batch_sampler=validsampler, shuffle=False,
                               num_workers=dataconfig["fetchworker_num"])

        from frameworks.CPC_Models import CPC_Model as Model
        from solvers import CPC_Solver as Solver

        model = Model.create_model(modelconfig['sp'], modelconfig['cpc'])

    elif args.type == 'finetune':

        tokenizer = data_utils.SubwordTokenizer(dataconfig["vocab_path"], add_blk=modelconfig['add_blk'])
        modelconfig["decoder"]["vocab_size"] = tokenizer.unit_num()
        label_range = [int(i) for i in dataconfig['label_range'].split(',')]

        training_set = datasets.SpeechDataset(dataconfig["trainset"], feat_range=feat_range, label_range=label_range)
        valid_set = datasets.SpeechDataset(dataconfig["devset"], reverse=True, feat_range=feat_range, label_range=label_range)
        trainingsampler = samplers.TimeBasedSampler(training_set, trainingconfig["batch_time"]*ngpu, ngpu, shuffle=True)
        validsampler = samplers.TimeBasedSampler(valid_set, trainingconfig["batch_time"]*ngpu, ngpu, shuffle=False) # for plot longer utterance
        collect = collates.WaveSampleCollate(tokenizer, add_eos=modelconfig["add_eos"],
                                             label_type=trainingconfig["label_type"])
        tr_loader = DataLoader(training_set, collate_fn=collect, batch_sampler=trainingsampler,
                               shuffle=False, num_workers=dataconfig["fetchworker_num"])
        cv_loader = DataLoader(valid_set, collate_fn=collect, batch_sampler=validsampler,
                               shuffle=False, num_workers=dataconfig["fetchworker_num"])

        from frameworks.Speech_Models import GRU_CTC_Model as Model
        from solvers import CTC_Solver as Solver

        model = Model.create_model(modelconfig["signal"],
                                   modelconfig["encoder"],
                                   modelconfig["decoder"]["vocab_size"])

        if trainingconfig['load_splayer']:
            logging.info("Load pretrained splayer from {}.".format(trainingconfig["load_splayer"]))
            pkg = torch.load(trainingconfig["load_splayer"])
            model.load_splayer(pkg["model"])
            utils.freeze(model.splayer)

    logging.info("\nModel info:\n{}".format(model))

    if args.continue_training:
        logging.info("Load package from {}.".format(os.path.join(trainingconfig["exp_dir"], "last.pt")))
        pkg = torch.load(os.path.join(trainingconfig["exp_dir"], "last.pt"))
        model.restore(pkg["model"])

    if "multi_gpu" in trainingconfig and trainingconfig["multi_gpu"] == True:
        logging.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    solver = Solver(model, trainingconfig, tr_loader, cv_loader)

    if args.continue_training:
        logging.info("Restore solver states...")
        solver.restore(pkg)
    logging.info("Start training...")
    solver.train()
    logging.info("Total time: {:.4f} secs".format(timer.toc()))
