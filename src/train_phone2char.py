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

    ngpu = 1
    if "multi_gpu" in trainingconfig and trainingconfig["multi_gpu"] == True:
        ngpu = torch.cuda.device_count()

    tokenizer_phone = data.CharTokenizer(dataconfig["vocab_phone"], add_blk=True)
    tokenizer_char = data.CharTokenizer(dataconfig["vocab_char"], add_blk=modelconfig['add_blk'])
    modelconfig["encoder"]["vocab_size"] = tokenizer_phone.unit_num()
    modelconfig["decoder"]["vocab_size"] = tokenizer_char.unit_num()

    training_set = data.ArkDataset(dataconfig["trainset"], rate_in_out=None)
    valid_set = data.ArkDataset(dataconfig["devset"], reverse=True, rate_in_out=None)

    if modelconfig['type'] == 'Embed_Decoder':
        from frameworks.Text_Models import Embed_Decoder as Model
        from solvers import Phone2Char_Solver as Solver

        model = Model.create_model(modelconfig["encoder"], modelconfig["decoder"])

        collate = data.Phone_Char_Collate(tokenizer_phone, tokenizer_char, modelconfig["add_eos"])
        tr_loader = torch.utils.data.DataLoader(training_set,
            collate_fn=collate, batch_size=trainingconfig['batch_size'], shuffle=False, num_workers=2)
        cv_loader = torch.utils.data.DataLoader(valid_set,
            collate_fn=collate, batch_size=trainingconfig['batch_size'], shuffle=False, num_workers=1)

    elif modelconfig['type'] == 'CIF_MIX':
        from frameworks.Speech_Models import CIF_MIX as Model
        from solvers import CIF_MIX_Solver as Solver

        model = Model.create_model(modelconfig["signal"],
                                   modelconfig["encoder"],
                                   modelconfig["assigner"],
                                   modelconfig["decoder"])

        collate = data.Feat_Phone_Char_Collate(tokenizer_phone, tokenizer_char, modelconfig["add_eos"])
        trainingsampler = data.FrameBasedSampler(training_set, trainingconfig["batch_frames"]*ngpu, ngpu, shuffle=True)
        validsampler = data.FrameBasedSampler(valid_set, trainingconfig["batch_frames"]*ngpu, ngpu, shuffle=False) # for plot longer utterance
        tr_loader = torch.utils.data.DataLoader(training_set,
            collate_fn=collate, batch_sampler=trainingsampler, shuffle=False, num_workers=dataconfig["fetchworker_num"])
        cv_loader = torch.utils.data.DataLoader(valid_set,
            collate_fn=collate, batch_sampler=validsampler, shuffle=False, num_workers=dataconfig["fetchworker_num"])

    logging.info("\nModel info:\n{}".format(model))

    if args.continue_training:
        logging.info("Load package from {}.".format(os.path.join(trainingconfig["exp_dir"], "last.pt")))
        pkg = torch.load(os.path.join(trainingconfig["exp_dir"], "last.pt"))
        model.restore(pkg["model"])

    if torch.cuda.is_available():
        model = model.cuda()

    solver = Solver(model, trainingconfig, tr_loader, cv_loader)

    if args.continue_training:
        logging.info("Restore solver states...")
        solver.restore(pkg)
    logging.info("Start training...")
    solver.train()
    logging.info("Total time: {:.4f} secs".format(timer.toc()))
