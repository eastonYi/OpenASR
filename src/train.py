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
    with open(args.config) as f:
        config = yaml.load(f)
    dataconfig = config["data"]
    trainingconfig = config["training"]
    modelconfig = config["model"]

    ngpu = 1
    if "multi_gpu" in trainingconfig and trainingconfig["multi_gpu"] == True:
        ngpu = torch.cuda.device_count()

    tokenizer = data.CharTokenizer(dataconfig["vocab_path"])
    if modelconfig['signal']["feature_type"] == 'offline':
        training_set = data.ArkDataset(dataconfig["trainset"])
        valid_set = data.ArkDataset(dataconfig["devset"], reverse=True)
        collate = data.FeatureCollate(tokenizer, dataconfig["maxlen"], modelconfig["no_eos"])
        trainingsampler = data.FrameBasedSampler(training_set, trainingconfig["batch_frames"]*ngpu, ngpu, shuffle=True)
        validsampler = data.FrameBasedSampler(valid_set, trainingconfig["batch_frames"]*ngpu, ngpu, shuffle=False) # for plot longer utterance
    else:
        training_set = data.SpeechDataset(dataconfig["trainset"])
        valid_set = data.SpeechDataset(dataconfig["devset"], reverse=True)
        collate = data.WaveCollate(tokenizer, dataconfig["maxlen"], modelconfig["no_eos"])
        trainingsampler = data.TimeBasedSampler(training_set, trainingconfig["batch_time"]*ngpu, ngpu, shuffle=True)
        validsampler = data.TimeBasedSampler(valid_set, trainingconfig["batch_time"]*ngpu, ngpu, shuffle=False) # for plot longer utterance

    tr_loader = torch.utils.data.DataLoader(training_set,
        collate_fn=collate, batch_sampler=trainingsampler, shuffle=False, num_workers=dataconfig["fetchworker_num"])
    cv_loader = torch.utils.data.DataLoader(valid_set,
        collate_fn=collate, batch_sampler=validsampler, shuffle=False, num_workers=dataconfig["fetchworker_num"])

    if modelconfig['type'] == 'conv-transformer':
        import sp_layers
        import encoder_layers
        import decoder_layers
        from models import Conv_Transformer as Model

        from trainer import Trainer

        splayer = sp_layers.SPLayer(modelconfig["signal"])
        encoder = encoder_layers.Transformer(modelconfig["encoder"])
        modelconfig["decoder"]["vocab_size"] = tokenizer.unit_num()
        decoder = decoder_layers.TransformerDecoder(modelconfig["decoder"])

        model = Model(splayer, encoder, decoder)

    elif modelconfig['type'] == 'CIF':
        import sp_layers
        import encoder_layers
        import attention_assigner
        import decoder_layers
        from models import CIF as Model

        from trainer import CIF_Trainer as Trainer

        splayer = sp_layers.SPLayer(modelconfig["signal"])
        encoder = encoder_layers.Transformer(modelconfig["encoder"])
        assigner = attention_assigner.Attention_Assigner(modelconfig["assigner"])
        modelconfig["decoder"]["vocab_size"] = tokenizer.unit_num()
        decoder = decoder_layers.CIF_Decoder(modelconfig["decoder"])

        model = Model(splayer, encoder, assigner, decoder)

    logging.info("\nModel info:\n{}".format(model))

    if args.continue_training:
        logging.info("Load package from {}.".format(os.path.join(trainingconfig["exp_dir"], "last-ckpt.pt")))
        pkg = torch.load(os.path.join(trainingconfig["exp_dir"], "last-ckpt.pt"))
        model.restore(pkg["model"])

    if "multi_gpu" in trainingconfig and trainingconfig["multi_gpu"] == True:
        logging.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    trainer = Trainer(model, trainingconfig, tr_loader, cv_loader)

    if args.continue_training:
        logging.info("Restore trainer states...")
        trainer.restore(pkg)
    logging.info("Start training...")
    trainer.train()
    logging.info("Total time: {:.4f} secs".format(timer.toc()))
