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
import time
import logging
import torch
from torch.nn.utils import clip_grad_norm_
import utils
import schedule
import metric


class Trainer(object):
    def __init__ (self, model, config, tr_loader, cv_loader):
        self.config = config
        self.tr_loader = tr_loader
        self.cv_loader = cv_loader

        self.model = model
        if config["multi_gpu"] == True:
            self.model_to_pack = self.model.module
        else:
            self.model_to_pack = self.model

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.num_epoch = config["num_epoch"]
        self.exp_dir = config["exp_dir"]
        self.print_inteval = config["print_inteval"]

        self.accumulate_grad_batch = config["accumulate_grad_batch"]
        self.init_lr = config["init_lr"]
        self.grad_max_norm = config["grad_max_norm"]
        self.label_smooth = config["label_smooth"]

        self.num_last_ckpt_keep = None
        if "num_last_ckpt_keep" in config:
            self.num_last_ckpt_keep = config["num_last_ckpt_keep"]

        self.lr_scheduler = schedule.get_scheduler(config["lr_scheduler"])
        self.metric_summarizer = metric.MetricSummarizer()
        self.metric_summarizer.register_metric("per_token_loss", display=True, visual=True, optim=True)
        self.metric_summarizer.register_metric("avg_token_loss", display=True, visual=True, optim=False)
        self.metric_summarizer.register_metric("learning_rate", display=True, visual=True, optim=False)
        self.metric_summarizer.register_metric("sequence_per_sec", display=True, visual=True, optim=False)

        # trainer state
        self.epoch = 0
        self.step = 0
        self.tr_loss = []
        self.cv_loss = []
        self.lr = self.init_lr

        if config["optimtype"] == "sgd":
            self.optimizer = torch.optim.SGD(self.model_to_pack.parameters(), lr=self.lr, momentum=0.9)
        elif config["optimtype"] == "adam":
            self.optimizer = torch.optim.Adam(self.model_to_pack.parameters(), lr=self.lr,
                betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            raise ValueError("Unknown optimizer.")
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)

    def training_state(self):
        return {
            "epoch": self.epoch,
            "step": self.step,
            "tr_loss": self.tr_loss,
            "cv_loss": self.cv_loss,
            "lr": self.lr,
            }

    def restore_training_state(self, state):
        self.epoch = state["epoch"]
        self.step = state["step"]
        self.tr_loss = state["tr_loss"]
        self.cv_loss = state["cv_loss"]
        self.lr = state["lr"]

    def package(self):
        return {
            "model": self.model_to_pack.package(),
            "trainer_config": self.config,
            "trainer_state": self.training_state(),
            "optim_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.pack_state()
            }

    def save(self, path):
        pkg = self.package()
        torch.save(pkg, path)
        logging.info("Saving model to {}".format(path))

    def restore(self, pkg):
        self.restore_training_state(pkg["trainer_state"])
        self.optimizer.load_state_dict(pkg['optim_state'])
        self.lr_scheduler.restore_state(pkg["scheduler_state"])

    def train(self):
        timer = utils.Timer()
        self.best_cvloss = 9e20
        if self.cv_loss:
            self.best_cvloss = min(self.cv_loss)

        while self.epoch < self.num_epoch:
            timer.tic()
            self.epoch += 1
            logging.info("Training")
            tr_loss = self.iter_one_epoch()
            tr_msg = ("tr loss: {:.4f}").format(tr_loss)
            msg = "\n" + "-"*85 + "\n"
            msg += "Epoch {} Training Summary:\n{}\n".format(self.epoch, tr_msg)
            msg += "-"*85
            logging.info(msg)
            self.save(os.path.join(self.exp_dir, "ep-{:04d}.pt".format(self.epoch)))
            self.save(os.path.join(self.exp_dir, "last-ckpt.pt"))
            logging.info("Validation")
            cv_loss = self.iter_one_epoch(cross_valid=True)

            if self.best_cvloss > cv_loss:
                self.best_cvloss = cv_loss
            train_time = timer.toc()
            cv_msg = ("cv loss: {:.4f} | best cv loss {:.4f}").format(cv_loss, self.best_cvloss)
            msg = "\n" + "-"*85 + "\n"
            msg += "Epoch {} Validation Summary:\n{}\n".format(self.epoch, cv_msg)
            msg += "Time cost: {:.4f} min".format(train_time/60.)
            msg += "\n" + "-"*85 + '\n'
            logging.info(msg)
            self.tr_loss.append(tr_loss)
            self.cv_loss.append(cv_loss)

            if self.num_last_ckpt_keep:
                utils.cleanup_ckpt(self.exp_dir, self.num_last_ckpt_keep)

    def iter_one_epoch(self, cross_valid=False):

        if cross_valid:
            loader = self.cv_loader
            self.model.eval()
        else:
            loader = self.tr_loader
            self.model.train()

        timer = utils.Timer()
        timer.tic()
        tot_loss = 0.
        tot_token = 0
        tot_sequence = 0

        n_accu_batch = self.accumulate_grad_batch

        tot_iter_num = len(loader)
        for niter, data in enumerate(loader):
            niter += 1
            utts, padded_waveforms, wave_lengths, ids, labels, paddings = data

            if cross_valid:
                with torch.no_grad():
                    this_loss = self.model(padded_waveforms.to(self.device),
                            wave_lengths.long().to(self.device),
                            ids.long().to(self.device),
                            labels.long().to(self.device),
                            paddings.long().to(self.device))
            else:
                this_loss = self.model(padded_waveforms.to(self.device),
                        wave_lengths.long().to(self.device),
                        ids.long().to(self.device),
                        labels.long().to(self.device),
                        paddings.long().to(self.device),
                        label_smooth=self.label_smooth)

            batch_loss = torch.sum(this_loss)
            n_token = torch.sum(1-paddings).float()
            n_sequence = len(utts)

            tot_loss = tot_loss + batch_loss
            tot_token = tot_token + n_token
            tot_sequence = tot_sequence + n_sequence

            self.metric_summarizer.reset_metrics()
            self.metric_summarizer.update_metric("per_token_loss", batch_loss, 1.0/n_token)
            self.metric_summarizer.update_metric("avg_token_loss", tot_loss, 1.0/tot_token)
            self.metric_summarizer.update_metric("learning_rate", list(self.optimizer.param_groups)[0]["lr"], 1.0)
            self.metric_summarizer.update_metric("sequence_per_sec", tot_sequence, 1.0/timer.toc())
            self.metric_summarizer.summarize()

            loss =  self.metric_summarizer.collect_loss()
            loss = loss/self.accumulate_grad_batch

            # compute gradients
            if not cross_valid:
                if n_accu_batch == self.accumulate_grad_batch:
                    self.optimizer.zero_grad()
                loss.backward()
                n_accu_batch -= 1
                if n_accu_batch == 0 or niter == tot_iter_num:
                    self.step += 1  # to be consistant with metric
                    clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
                    self.lr_scheduler.step()   # then, update learning rate
                    self.lr_scheduler.set_lr(self.optimizer, self.init_lr)
                    self.optimizer.step()
                    n_accu_batch = self.accumulate_grad_batch
                else:
                    continue

            timer.toc()
            if niter % self.print_inteval == 0:
                todisp = self.metric_summarizer.fetch_scalers(use="display")
                todispmsg = self.metric_summarizer.display_msg(todisp)
                del todisp
                msg = ("\nEpoch {} | Step {} | Iter {}:\n").format(self.epoch, self.step, niter)
                msg += todispmsg
                logging.info("Progress:\n" + msg.strip())

        self.metric_summarizer.reset_metrics()
        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_token).item()


class CIF_Trainer(Trainer):
    def __init__ (self, model, config, tr_loader, cv_loader):
        self.config = config
        self.tr_loader = tr_loader
        self.cv_loader = cv_loader

        self.model = model
        if config["multi_gpu"] == True:
            self.model_to_pack = self.model.module
        else:
            self.model_to_pack = self.model

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.num_epoch = config["num_epoch"]
        self.exp_dir = config["exp_dir"]
        self.print_inteval = config["print_inteval"]

        self.accumulate_grad_batch = config["accumulate_grad_batch"]
        self.init_lr = config["init_lr"]
        self.grad_max_norm = config["grad_max_norm"]
        self.label_smooth = config["label_smooth"]
        self.lambda_qua = config["lambda_qua"]

        self.num_last_ckpt_keep = None
        if "num_last_ckpt_keep" in config:
            self.num_last_ckpt_keep = config["num_last_ckpt_keep"]

        self.lr_scheduler = schedule.get_scheduler(config["lr_scheduler"])

        # trainer state
        self.epoch = 0
        self.step = 0
        self.tr_loss = []
        self.cv_loss = []
        self.lr = self.init_lr

        if config["optimtype"] == "sgd":
            self.optimizer = torch.optim.SGD(self.model_to_pack.parameters(), lr=self.lr, momentum=0.9)
        elif config["optimtype"] == "adam":
            self.optimizer = torch.optim.Adam(self.model_to_pack.parameters(), lr=self.lr,
                betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            raise ValueError("Unknown optimizer.")
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)

    def iter_one_epoch(self, cross_valid=False):

        if cross_valid:
            loader = self.cv_loader
            self.model.eval()
        else:
            loader = self.tr_loader
            self.model.train()

        timer = utils.Timer()
        timer.tic()
        tot_loss = 0.
        tot_token = 0
        tot_sequence = 0

        n_accu_batch = self.accumulate_grad_batch

        tot_iter_num = len(loader)
        for niter, data in enumerate(loader):
            niter += 1
            utts, padded_waveforms, wave_lengths, ids, labels, paddings = data

            if cross_valid:
                with torch.no_grad():
                    qua_loss, ce_loss = self.model(padded_waveforms.to(self.device),
                            wave_lengths.long().to(self.device),
                            ids.long().to(self.device),
                            labels.long().to(self.device),
                            paddings.long().to(self.device))
            else:
                qua_loss, ce_loss = self.model(padded_waveforms.to(self.device),
                        wave_lengths.long().to(self.device),
                        ids.long().to(self.device),
                        labels.long().to(self.device),
                        paddings.long().to(self.device),
                        label_smooth=self.label_smooth)

            n_token = torch.sum(1-paddings).float()
            tot_token += n_token
            n_sequence = len(utts)
            tot_sequence += n_sequence


            loss = (self.lambda_qua * qua_loss + ce_loss).sum() / n_token
            tot_loss += ce_loss

            # compute gradients
            if not cross_valid:
                if n_accu_batch == self.accumulate_grad_batch:
                    self.optimizer.zero_grad()
                loss.backward()
                n_accu_batch -= 1
                if n_accu_batch == 0 or niter == tot_iter_num:
                    self.step += 1  # to be consistant with metric
                    clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
                    self.lr_scheduler.step()   # then, update learning rate
                    self.lr_scheduler.set_lr(self.optimizer, self.init_lr)
                    self.optimizer.step()
                    n_accu_batch = self.accumulate_grad_batch
                else:
                    continue

            timer.toc()
            if niter % self.print_inteval == 0:
                print('per_token_loss: {:.3f} avg_token_ce_loss: {:.3f} learning_rate: {:.3e} sequence_per_sec: {:.3f}'.format(
                    loss, tot_loss / tot_token, list(self.optimizer.param_groups)[0]["lr"], tot_sequence/timer.toc()
                ), flush=True)

                print("Epoch {} | Step {} | Iter {}:\n".format(self.epoch, self.step, niter),
                      flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_token).item()
