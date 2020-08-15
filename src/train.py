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
import utils
import data


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

    tokenizer = data.CharTokenizer(dataconfig["vocab_path"], add_blk=modelconfig['add_blk'])
    modelconfig["decoder"]["vocab_size"] = tokenizer.unit_num()
    if modelconfig['signal']["feature_type"] == 'offline':
        training_set = data.ArkDataset(dataconfig["trainset"], feat_range=feat_range, label_range=label_range)
        valid_set = data.ArkDataset(dataconfig["devset"], reverse=True)
        collate = data.FeatureCollate(tokenizer, modelconfig["add_eos"], trainingconfig["label_type"])
        trainingsampler = data.FrameBasedSampler(training_set, trainingconfig["batch_frames"]*ngpu, ngpu, shuffle=True)
        validsampler = data.FrameBasedSampler(valid_set, trainingconfig["batch_frames"]*ngpu, ngpu, shuffle=False) # for plot longer utterance
    else:
        training_set = data.SpeechDataset(dataconfig["trainset"])
        valid_set = data.SpeechDataset(dataconfig["devset"], reverse=True)
        collate = data.WaveCollate(tokenizer, dataconfig["maxlen"], modelconfig["add_eos"])
        trainingsampler = data.TimeBasedSampler(training_set, trainingconfig["batch_time"]*ngpu, ngpu, shuffle=True)
        validsampler = data.TimeBasedSampler(valid_set, trainingconfig["batch_time"]*ngpu, ngpu, shuffle=False) # for plot longer utterance

    tr_loader = torch.utils.data.DataLoader(training_set,
        collate_fn=collate, batch_sampler=trainingsampler, shuffle=False, num_workers=dataconfig["fetchworker_num"])
    cv_loader = torch.utils.data.DataLoader(valid_set,
        collate_fn=collate, batch_sampler=validsampler, shuffle=False, num_workers=dataconfig["fetchworker_num"])

    if modelconfig['type'] == 'conv-transformer':
        from frameworks.Speech_Models import Conv_Transformer as Model
        from solvers import CE_Solver as Solver

        model = Model.create_model(modelconfig["signal"],
                                   modelconfig["encoder"],
                                   modelconfig["decoder"])

    elif modelconfig['type'] == 'conv-ctc-transformer':
        from frameworks.Speech_Models import Conv_CTC_Transformer as Model
        from solvers import CTC_CE_Solver as Solver

        model = Model.create_model(modelconfig["signal"],
                                   modelconfig["encoder"],
                                   modelconfig["decoder"])

    elif modelconfig['type'] == 'CIF':
        from frameworks.Speech_Models import CIF as Model
        from solvers import CIF_Solver as Solver

        model = Model.create_model(modelconfig["signal"],
                                   modelconfig["encoder"],
                                   modelconfig["assigner"],
                                   modelconfig["decoder"])

    elif modelconfig['type'] == 'Conv_CTC':
        from frameworks.Speech_Models import Conv_CTC as Model
        from solvers import CTC_Solver as Solver

        model = Model.create_model(modelconfig["signal"],
                                   modelconfig["encoder"],
                                   vocab_size=modelconfig["decoder"]["vocab_size"])

    logging.info("\nModel info:\n{}".format(model))

    if args.continue_training:
        logging.info("Load package from {}.".format(os.path.join(trainingconfig["exp_dir"], "last.pt")))
        pkg = torch.load(os.path.join(trainingconfig["exp_dir"], "last.pt"))
        model.restore(pkg["model"])
    elif trainingconfig['pretrained_model']:
        logging.info("Load package from {}.".format(trainingconfig['pretrained_model']))
        pkg = torch.load(trainingconfig['pretrained_model'])
        model.restore_without_fc(pkg["model"])
        trainingconfig['init_lr'] *= 0.1

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
