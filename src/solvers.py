"""
"""
import os
import time
import logging
import torch
from torch.nn.utils import clip_grad_norm_

import utils
import schedule
from utils import cycle


class Solver(object):
    def __init__(self, model, config, tr_loader, cv_loader):
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
        # Solver state
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
            "Solver_config": self.config,
            "Solver_state": self.training_state(),
            "optim_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.pack_state()
            }

    def save(self, path):
        pkg = self.package()
        torch.save(pkg, path)
        logging.info("Saving model to {}".format(path))

    def restore(self, pkg):
        self.restore_training_state(pkg["Solver_state"])
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
            self.save(os.path.join(self.exp_dir, "last.pt"))
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


class CE_Solver(Solver):

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
        for niter, (utts, data) in enumerate(loader):
            niter += 1
            feats, len_feat, target_in, targets, paddings = \
                (i.to(self.device) for i in data)

            if niter == 1 and self.epoch == 1:
                print('feats:\t{}\nlen_feat:\t{}\ntarget_in:\t{}\ntargets:\t{}\npaddings:\t{}'.format(
                    feats.size(), len_feat.size(), target_in.size(), targets.size(), paddings.size()))
                print('feats:\n{}\nlen_feat:\t{}\ntarget_in:\t{}\ntargets:\t{}\npaddings:\t{}'.format(
                    feats[0], len_feat[0], target_in[0], targets[0], paddings[0]))

            if cross_valid:
                with torch.no_grad():
                    ce_loss = self.model(feats, len_feat, target_in, targets, paddings)
            else:
                ce_loss = self.model(feats, len_feat, target_in, targets, paddings,
                        label_smooth=self.label_smooth)

            n_token = torch.sum(1-paddings).float()
            tot_token += n_token
            n_sequence = len(utts)
            tot_sequence += n_sequence

            loss = ce_loss.sum()/n_token
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
                print('Epoch {} | Step {} | Batch {}/{} {} \ncur_all_loss: {:.3f} ce_loss: {:.3f} lr: {:.3e} sent/sec: {:.3f}\n'.format(
                    self.epoch, self.step, niter, tot_iter_num, list(feats.size()),
                    loss, tot_loss / tot_token, list(self.optimizer.param_groups)[0]["lr"], tot_sequence/timer.toc()
                ), flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_token).item()


class CTC_CE_Solver(Solver):
    def __init__ (self, model, config, tr_loader, cv_loader):
        super().__init__(model, config, tr_loader, cv_loader)
        self.lambda_ctc = config["lambda_ctc"]

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
        tot_ctc_loss = 0.
        tot_token = 0
        tot_sequence = 0

        n_accu_batch = self.accumulate_grad_batch

        tot_iter_num = len(loader)
        for niter, data in enumerate(loader):
            niter += 1
            utts, padded_waveforms, wave_lengths, ids, labels, paddings = data

            if cross_valid:
                with torch.no_grad():
                    ctc_loss, ce_loss = self.model(padded_waveforms,
                            wave_lengths.long(),
                            ids.long(),
                            labels.long(),
                            paddings.long())
            else:
                ctc_loss, ce_loss = self.model(padded_waveforms,
                        wave_lengths.long(),
                        ids.long(),
                        labels.long(),
                        paddings.long(),
                        label_smooth=self.label_smooth)

            n_token = torch.sum(1-paddings).float()
            tot_token += n_token
            n_sequence = len(utts)
            tot_sequence += n_sequence

            loss = ce_loss.sum()/n_token + self.lambda_ctc * ctc_loss.sum()/n_sequence

            tot_ctc_loss += ctc_loss
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
                print('Epoch {} | Step {} | Iter {} batch {} \ncur_all_loss: {:.3f} ce_loss: {:.3f} ctc_loss/sent: {:.3f} lr: {:.3e} sent/sec: {:.3f}\n'.format(
                    self.epoch, self.step, niter, list(padded_waveforms.size()),
                    loss, tot_loss / tot_token, tot_ctc_loss / tot_sequence,
                    list(self.optimizer.param_groups)[0]["lr"], tot_sequence/timer.toc()
                ), flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_token).item()


class CTC_Solver(CE_Solver):

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
        for niter, (utts, data) in enumerate(loader):
            niter += 1
            feats, len_feat, _, targets, paddings = (i.to(self.device) for i in data)
            # import pdb; pdb.set_trace()

            if niter == 1 and self.epoch == 1:
                print('feats:\t{}\nlen_feat:\t{}\ntargets:\t{}\npaddings:\t{}'.format(
                    feats.size(), len_feat.size(), targets.size(), paddings.size()))
                print('feats:\n{}\nlen_feat:\t{}\ntargets:\t{}\npaddings:\t{}'.format(
                    feats[0][:100], len_feat[0], targets[0], paddings[0]))

            len_target = (1-paddings).int().sum(-1)
            n_token = len_target.sum().float()
            tot_token += n_token
            n_sequence = len(utts)
            tot_sequence += n_sequence

            if cross_valid:
                with torch.no_grad():
                    ctc_loss = self.model(feats, len_feat, targets, len_target)
                    if niter == 1:
                        logits, len_logits = self.model.get_logits(feats[:1], len_feat[:1])
                        blk_idx = logits.size(-1) - 1
                        align = torch.argmax(logits[0], -1)[:len_logits[0]]
                        print('infer:\n', utils.ctc_reduce(align[align<blk_idx].tolist()))
                        print('target:\n', targets[0][:len_target[0]].tolist())
            else:
                ctc_loss = self.model(feats, len_feat, targets, len_target)

            loss = ctc_loss.sum()/n_sequence
            tot_loss += ctc_loss

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
                print('Epoch {} | Step {} | Batch {}/{} {} \ncur_loss: {:.3f} avg_loss: {:.3f} lr: {:.3e} sent/sec: {:.3f}\n'.format(
                    self.epoch, self.step, niter, tot_iter_num, list(feats.size()),
                    loss, tot_loss / tot_sequence,
                    list(self.optimizer.param_groups)[0]["lr"],
                    tot_sequence/timer.toc()
                ), flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_sequence).item()


class CIF_Solver(Solver):
    def __init__(self, model, config, tr_loader, cv_loader):
        super().__init__(model, config, tr_loader, cv_loader)
        self.lambda_qua = config["lambda_qua"]

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
        tot_qua_loss = 0.
        tot_token = 0
        tot_sequence = 0

        n_accu_batch = self.accumulate_grad_batch

        tot_iter_num = len(loader)
        for niter, (utts, data) in enumerate(loader):
            niter += 1
            padded_waveforms, wave_lengths, target_input, target, paddings = \
                (i.to(self.device) for i in data)

            if cross_valid:
                with torch.no_grad():
                    qua_loss, ce_loss = self.model(padded_waveforms,
                            wave_lengths.long(),
                            target_input.long(),
                            target.long(),
                            paddings.long())
            else:
                qua_loss, ce_loss = self.model(padded_waveforms,
                        wave_lengths.long(),
                        target_input.long(),
                        target.long(),
                        paddings.long(),
                        label_smooth=self.label_smooth)

            n_token = torch.sum(1-paddings).float()
            tot_token += n_token
            n_sequence = len(utts)
            tot_sequence += n_sequence

            loss = ce_loss.sum()/n_token + \
                   self.lambda_qua * qua_loss.sum()/n_sequence

            tot_qua_loss += qua_loss
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
                print('Epoch {} | Step {} | Iter {} batch {} \ncur_all_loss: {:.3f} ce_loss: {:.3f} qua_loss: {:.3f} lr: {:.3e} sec/sent: {:.3f}s\n'.format(
                    self.epoch, self.step, niter, list(padded_waveforms.size()),
                    loss, tot_loss/tot_token, tot_qua_loss/tot_sequence,
                    list(self.optimizer.param_groups)[0]["lr"], tot_sequence/timer.toc()
                ), flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_token).item()


class CIF_CTC_Solver(Solver):
    def __init__ (self, model, config, tr_loader, cv_loader):
        super().__init__(model, config, tr_loader, cv_loader)
        self.lambda_ctc = config["lambda_ctc"]

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
        tot_ctc_loss = 0.
        tot_qua_loss = 0.
        tot_token = 0
        tot_sequence = 0

        n_accu_batch = self.accumulate_grad_batch

        tot_iter_num = len(loader)
        for niter, data in enumerate(loader):
            niter += 1
            utts, padded_waveforms, wave_lengths, target_input, target, paddings = data

            if cross_valid:
                with torch.no_grad():
                    ctc_loss, qua_loss, ce_loss = self.model(padded_waveforms,
                            wave_lengths.long(),
                            target_input.long(),
                            target.long(),
                            paddings.long())
            else:
                ctc_loss, qua_loss, ce_loss = self.model(padded_waveforms,
                        wave_lengths.long(),
                        target_input.long(),
                        target.long(),
                        paddings.long(),
                        label_smooth=self.label_smooth)

            n_token = torch.sum(1-paddings).float()
            tot_token += n_token
            n_sequence = len(utts)
            tot_sequence += n_sequence

            loss = ce_loss.sum()/n_token + \
                   self.lambda_qua * qua_loss.sum()/n_sequence + \
                   self.lambda_ctc * ctc_loss.sum()/n_sequence

            tot_ctc_loss += ctc_loss
            tot_qua_loss += qua_loss
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
                print('Epoch {} | Step {} | Iter {} batch {} \ncur_all_loss: {:.3f} ce_loss: {:.3f} ctc_loss: {:.3f} qua_loss: {:.3f} lr: {:.3e} sec/sent: {:.3f}s\n'.format(
                    self.epoch, self.step, niter, list(padded_waveforms.size()),
                    loss, tot_loss/tot_token, tot_ctc_loss/tot_sequence, tot_qua_loss/tot_sequence,
                    list(self.optimizer.param_groups)[0]["lr"], tot_sequence/timer.toc()
                ), flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_token).item()


class Phone2Char_Solver(Solver):

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
        for niter, (utts, data) in enumerate(loader):
            niter += 1
            xs_in, len_xs, target_in, target, paddings = (i.to(self.device) for i in data)

            if cross_valid:
                with torch.no_grad():
                    ce_loss = self.model(xs_in, len_xs, target_in, target, paddings)
            # elif cross_valid == 'performance':
            #     self.model.eval()
            #     with torch.no_grad():
            #         encoded, len_encoded = self.model.get_encoded(xs_in, len_xs)
            #         pred_ids, len_decodeds, scores = self.model.batch_beam_decode(
            #             encoded, len_encoded, beam_size=self.nbest, max_decode_len=self.maxlen)
            #
            #     self.model.train()
            else:
                ce_loss = self.model(xs_in, len_xs, target_in, target, paddings,
                                     label_smooth=self.label_smooth)

            n_token = torch.sum(1-paddings).float()
            tot_token += n_token
            n_sequence = len(utts)
            tot_sequence += n_sequence

            loss = ce_loss.sum()/n_token
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
                print('Epoch {} | Step {} | Iter {} batch {} \ncur_all_loss: {:.3f} ce_loss: {:.3f} lr: {:.3e} sent/sec: {:.3f}\n'.format(
                    self.epoch, self.step, niter, list(xs_in.size()),
                    loss, tot_loss / tot_token, list(self.optimizer.param_groups)[0]["lr"], tot_sequence/timer.toc()
                ), flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_token).item()

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
            self.save(os.path.join(self.exp_dir, "last.pt"))
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


class CIF_FC_Solver(CIF_Solver):

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
        tot_sequence = 0
        tot_phone_acoustic = 0

        n_accu_batch = self.accumulate_grad_batch

        tot_iter_num = len(loader)
        for niter, (utts, data_acoustic) in enumerate(loader):
            niter += 1
            if n_accu_batch == self.accumulate_grad_batch:
                self.optimizer.zero_grad()

            feats_acoustic, len_feat_acoustic, phones_acoustic, len_phone_acoustic = \
                (i.to(self.device) for i in data_acoustic)

            # general acoustic loss
            n_sequence = len(utts)
            n_phone_acoustic = len_phone_acoustic.sum()
            tot_phone_acoustic += n_phone_acoustic
            tot_sequence+= n_sequence

            loss_qua_acoustic, loss_ce_phone_acoustic = \
                self.model(feats_acoustic, len_feat_acoustic, phones_acoustic, len_phone_acoustic,
                           label_smooth=self.label_smooth)

            loss_ce_phone_acoustic = loss_ce_phone_acoustic.sum() / n_phone_acoustic
            loss_qua_acoustic = loss_qua_acoustic.sum() / n_sequence
            loss_acoustic = loss_ce_phone_acoustic + \
                   self.lambda_qua * loss_qua_acoustic
            loss_acoustic.backward()

            tot_loss += loss_acoustic

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
                print('''Epoch {} | Step {} | Iter {} acoustic {} | lr: {:.3e} | sent/sec: {:.1f}
acoustic cur_all_loss: {:.3f} loss_ce_phone: {:.3f} loss_qua: {:.3f}
                      '''.format(
                    self.epoch, self.step, niter, list(feats_acoustic.size()),
                    list(self.optimizer.param_groups)[0]["lr"], tot_sequence/timer.toc(),
                    loss_acoustic, loss_ce_phone_acoustic, loss_qua_acoustic
                ), flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_phone_acoustic).item()


class CIF_CTC_FC_Solver(CIF_FC_Solver):

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
        tot_sequence = 0
        tot_phone_acoustic = 0

        n_accu_batch = self.accumulate_grad_batch

        tot_iter_num = len(loader)
        for niter, (utts, data_acoustic) in enumerate(loader):
            niter += 1
            if n_accu_batch == self.accumulate_grad_batch:
                self.optimizer.zero_grad()

            feats_acoustic, len_feat_acoustic, phones_acoustic, len_phone_acoustic = \
                (i.to(self.device) for i in data_acoustic)

            # general acoustic loss
            n_sequence = len(utts)
            n_phone_acoustic = len_phone_acoustic.sum()
            tot_phone_acoustic += n_phone_acoustic
            tot_sequence+= n_sequence

            loss_ctc_acoustic, loss_qua_acoustic, loss_ce_phone_acoustic = \
                self.model(feats_acoustic, len_feat_acoustic, phones_acoustic, len_phone_acoustic,
                           label_smooth=self.label_smooth)

            loss_ce_phone_acoustic = loss_ce_phone_acoustic.sum() / n_phone_acoustic
            loss_ctc_acoustic = loss_ctc_acoustic.sum() / n_sequence
            loss_qua_acoustic = loss_qua_acoustic.sum() / n_sequence
            loss_acoustic = loss_ce_phone_acoustic + \
                   self.lambda_qua * loss_qua_acoustic + \
                   self.lambda_ctc * loss_ctc_acoustic
            loss_acoustic.backward()

            tot_loss += loss_acoustic

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
                print('''Epoch {} | Step {} | Iter {} acoustic {} | lr: {:.3e} | sent/sec: {:.1f}
acoustic cur_all_loss: {:.3f} loss_ce_phone: {:.3f} loss_ctc: {:.3f} loss_qua: {:.3f}
                      '''.format(
                    self.epoch, self.step, niter, list(feats_acoustic.size()),
                    list(self.optimizer.param_groups)[0]["lr"], tot_sequence/timer.toc(),
                    loss_acoustic, loss_ce_phone_acoustic, loss_ctc_acoustic, loss_qua_acoustic
                ), flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_phone_acoustic).item()


class CIF_MIX_Solver(CIF_Solver):

    def __init__ (self, model, config, batchiter_acoustic, batchiter_train, batchiter_dev):
        self.config = config
        self.batchiter_acoustic = batchiter_acoustic
        self.batchiter_train = batchiter_train
        self.batchiter_dev = batchiter_dev

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
        self.lambda_ctc = config["lambda_ctc"]

        self.num_last_ckpt_keep = None
        if "num_last_ckpt_keep" in config:
            self.num_last_ckpt_keep = config["num_last_ckpt_keep"]

        self.lr_scheduler = schedule.get_scheduler(config["lr_scheduler"])
        # Solver state
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

    def train(self):
        timer = utils.Timer()
        self.best_cvloss = 9e20
        if self.cv_loss:
            self.best_cvloss = min(self.cv_loss)

        while self.epoch < self.num_epoch:
            timer.tic()
            self.epoch += 1
            logging.info("Training")
            tr_loss = self.iter_train_epoch()
            tr_msg = ("tr loss: {:.4f}").format(tr_loss)
            msg = "\n" + "-"*85 + "\n"
            msg += "Epoch {} Training Summary:\n{}\n".format(self.epoch, tr_msg)
            msg += "-"*85
            logging.info(msg)
            self.save(os.path.join(self.exp_dir, "ep-{:04d}.pt".format(self.epoch)))
            self.save(os.path.join(self.exp_dir, "last.pt"))
            logging.info("Validation")
            cv_loss = self.iter_dev_epoch()

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

    def iter_train_epoch(self):
        loader_acoustic = self.batchiter_acoustic
        loader = iter(cycle(self.batchiter_train))
        self.model.train()

        timer = utils.Timer()
        timer.tic()
        tot_loss = 0.
        tot_phone = 0
        tot_token = 0
        tot_sequence = 0

        tot_phone_acoustic = 0
        tot_sequence_acoustic = 0

        n_accu_batch = self.accumulate_grad_batch

        tot_iter_num = len(loader_acoustic)
        for niter, ((utts_acoustic, data_acoustic), (utts, data)) in enumerate(zip(loader_acoustic, loader)):
            niter += 1
            if n_accu_batch == self.accumulate_grad_batch:
                self.optimizer.zero_grad()

            feats_acoustic, len_feat_acoustic, phones_acoustic, len_phone_acoustic = \
                (i.to(self.device) for i in data_acoustic)
            feats, len_feat, phones, len_phone, target_in, targets, paddings = \
                (i.to(self.device) for i in data)

            if niter == 1 and self.epoch == 1:
                print('feats_acoustic:\t{}\nlen_feat_acoustic:\t{}\nphones_acoustic:\t{}\nlen_phone_acoustic:\t{}'.format(
                    feats_acoustic.size(), len_feat_acoustic.size(), phones_acoustic.size(), len_phone_acoustic.size()))
                print('feats_acoustic:\n{}\nlen_feat_acoustic:\t{}\nphones_acoustic:\t{}\nlen_phone_acoustic:\t{}'.format(
                    feats_acoustic[0], len_feat_acoustic[0], phones_acoustic[0], len_phone_acoustic[0]))
                print('feats:\t{}\nlen_feat:\t{}\nphones:\t{}\nlen_phone:\t{}\ntargets:\t{}\npaddings:\t{}'.format(
                    feats.size(), len_feat.size(), phones.size(), len_phone.size(), target_in.size(), targets.size(), paddings.size()))
                print('feats:\n{}\nlen_feat:\t{}\nphones:\t{}\nlen_phone:\t{}\ntarget_in:\t{}\ntargets:\t{}\npaddings:\t{}'.format(
                    feats[0], len_feat[0], phones[0], len_phone[0], target_in[0], targets[0], paddings[0]))

            timer.tic()
            # general acoustic loss
            n_phone_acoustic = len_phone_acoustic.sum()
            tot_phone_acoustic += n_phone_acoustic
            n_sequence_acoustic = len(utts_acoustic)
            tot_sequence_acoustic += n_sequence_acoustic

            loss_ctc_acoustic, loss_qua_acoustic, loss_ce_phone_acoustic = \
                self.model(feats_acoustic, len_feat_acoustic, phones_acoustic, len_phone_acoustic,
                           label_smooth=self.label_smooth)

            print(timer.toc())
            loss_ce_phone_acoustic = loss_ce_phone_acoustic.sum() / n_phone_acoustic
            loss_ctc_acoustic = loss_ctc_acoustic.sum() / n_sequence_acoustic
            loss_qua_acoustic = loss_qua_acoustic.sum() / n_sequence_acoustic
            loss_acoustic = loss_ce_phone_acoustic + \
                   self.lambda_qua * loss_qua_acoustic + \
                   self.lambda_ctc * loss_ctc_acoustic
            loss_acoustic.backward()

            loss_ctc, loss_qua, loss_ce_phone, loss_ce_target = \
                self.model(feats, len_feat, phones, len_phone, target_in, targets, paddings,
                           label_smooth=self.label_smooth)
            print(timer.toc())

            n_phone = len_phone.sum()
            n_token = torch.sum(1-paddings).float()
            tot_phone += n_phone
            tot_token += n_token
            n_sequence = len(utts)
            tot_sequence += n_sequence

            loss_ce_phone = loss_ce_phone.sum() / n_phone
            loss_ce_target = loss_ce_target.sum() / n_token
            loss_ctc = loss_ctc.sum() / n_sequence
            loss_qua = loss_qua.sum() / n_sequence
            loss = loss_ce_phone + loss_ce_target + \
                   self.lambda_qua * loss_qua + \
                   self.lambda_ctc * loss_ctc
            loss.backward()
            tot_loss += loss

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

            if niter % self.print_inteval == 0:
                print('''Epoch {} | Step {} | acoustic {}/{} {} | target {} | lr: {:.3e} | sent/sec: {:.1f}
acoustic cur_all_loss: {:.3f} loss_ce_phone: {:.3f} loss_ctc: {:.3f} loss_qua: {:.3f}
target   cur_all_loss: {:.3f} loss_ce_phone: {:.3f} loss_ctc: {:.3f} loss_qua: {:.3f} loss_ce_char: {:.3f}
                      '''.format(
                    self.epoch, self.step, niter, tot_iter_num, list(feats_acoustic.size()), list(feats.size()),
                    list(self.optimizer.param_groups)[0]["lr"], tot_sequence_acoustic/timer.toc(),
                    loss_acoustic, loss_ce_phone_acoustic, loss_ctc_acoustic, loss_qua_acoustic,
                    loss, loss_ce_phone, loss_ctc, loss_qua, loss_ce_target,
                ), flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_token).item()

    def iter_dev_epoch(self):
        loader = self.batchiter_dev
        self.model.eval()

        timer = utils.Timer()
        timer.tic()
        tot_loss = 0.
        tot_phone = 0
        tot_token = 0
        tot_sequence = 0

        for niter, (utts, data) in enumerate(loader):
            niter += 1
            feats, len_feat, phones, len_phone, target_in, target_out, paddings = \
                (i.to(self.device) for i in data)

            with torch.no_grad():
                ctc_loss, qua_loss, ce_phone_loss, ce_target_loss = \
                self.model(feats, len_feat, phones, len_phone, target_in, target_out, paddings)

            n_phone = len_phone.sum()
            n_token = torch.sum(1-paddings).float()
            tot_phone += n_phone
            tot_token += n_token
            n_sequence = len(utts)
            tot_sequence += n_sequence

            loss = ce_phone_loss.sum()/n_phone + \
                   ce_target_loss.sum()/n_token + \
                   self.lambda_qua * qua_loss.sum()/n_sequence + \
                   self.lambda_ctc * ctc_loss.sum()/n_sequence
            tot_loss += loss

            timer.toc()
            if niter % self.print_inteval == 0:
                print('Epoch {} | Step {} | Iter {} batch {} \ncur_all_loss: {:.3f} ce_loss: {:.3f} lr: {:.3e} sent/sec: {:.3f}\n'.format(
                    self.epoch, self.step, niter, list(feats.size()),
                    loss, tot_loss / tot_token, list(self.optimizer.param_groups)[0]["lr"], tot_sequence/timer.toc()
                ), flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_token).item()


class CPC_Solver(Solver):

    def iter_one_epoch(self, cross_valid=False):
        if cross_valid:
            loader = self.cv_loader
            self.model.eval()
        else:
            loader = self.tr_loader
            self.model.train()

        if self.config["multi_gpu"]:
            init_hidden = self.model.module.init_hidden
        else:
            init_hidden = self.model.init_hidden

        timer = utils.Timer()
        timer.tic()
        tot_loss = 0.
        tot_sequence = 0

        n_accu_batch = self.accumulate_grad_batch

        tot_iter_num = len(loader)
        for niter, (utts, data) in enumerate(loader):
            niter += 1
            feats, len_feat = (i.to(self.device) for i in data)
            if len(feats) < 2:
                continue

            if niter == 1 and self.epoch == 1:
                print('feats:\t{}\nlen_feat:\t{}'.format(
                    feats.size(), len_feat.size()))

            if cross_valid:
                with torch.no_grad():
                    hidden = init_hidden(len(feats), feats.device)
                    acc, loss, hidden = self.model(feats, len_feat, hidden)
            else:
                hidden = init_hidden(len(feats), feats.device)
                acc, loss, hidden = self.model(feats, len_feat, hidden)

            n_sequence = len(utts)
            tot_sequence += n_sequence
            tot_loss += loss

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
                print('Epoch {} | Step {} | Batch {}/{} {} \ncur_loss: {:.3f} avg_loss: {:.3f} cur_acc: {:.2f}% lr: {:.3e} sent/sec: {:.3f}\n'.format(
                    self.epoch, self.step, niter, tot_iter_num, list(feats.size()),
                    loss/n_sequence, tot_loss/tot_sequence, acc*100, list(self.optimizer.param_groups)[0]["lr"], tot_sequence/timer.toc()
                ), flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_sequence).item()
