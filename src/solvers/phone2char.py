"""
"""
import os
import time
import logging
import torch
import random
from torch.nn.utils import clip_grad_norm_

import utils
from solvers import Solver


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


class Phone2Char_CTC_Solver(Solver):

    def __init__(self, model, config, tr_loader, cv_loader, tokenizer):
        super().__init__(model, config, tr_loader, cv_loader)
        self.decode_fn = utils.ctc_decode_fn(
            list(tokenizer.unit2id.keys()),
            beam_width=1,
            blank_id=tokenizer.unit_num()-1)

    def iter_one_epoch(self, cross_valid=False):
        if cross_valid:
            loader = self.cv_loader
            self.model.eval()
            num_tokens = 0
            num_dist = 0
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
            xs_in, len_xs, _, targets, paddings = (i.to(self.device) for i in data)

            if cross_valid:
                with torch.no_grad():
                    ctc_loss = self.model(xs_in, len_xs, targets, paddings)
                    ctc_logits, len_logits = self.model.get_logits(xs_in, len_xs)
                pred_ids, len_decodeds, _ = self.model.batch_beam_decode(ctc_logits, len_logits, self.decode_fn)
                pred_ids = pred_ids[:, 0, :].cpu().numpy()
                len_decodeds = [i[0] for i in len_decodeds.cpu().tolist()]
                len_targets = (1-paddings).sum(-1)
                num_tokens += len_targets.sum()
                batch_dist = utils.batch_distance(pred_ids, len_decodeds, targets.cpu().numpy(), len_targets.cpu().tolist())

                num_dist += batch_dist
                random.randint(0, len(xs_in))
            else:
                ctc_loss = self.model(xs_in, len_xs, targets, paddings)

            n_token = torch.sum(1-paddings).float()
            tot_token += n_token
            n_sequence = len(utts)
            tot_sequence += n_sequence

            loss = ctc_loss.sum()/n_token
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
                print('Epoch {} | Step {} | Iter {} batch {} \ncur_all_loss: {:.3f} ctc_loss: {:.3f} lr: {:.3e} sent/sec: {:.3f}\n'.format(
                    self.epoch, self.step, niter, list(xs_in.size()),
                    loss, tot_loss / tot_token, list(self.optimizer.param_groups)[0]["lr"], tot_sequence/timer.toc()
                ), flush=True)

        torch.cuda.empty_cache()
        time.sleep(2)

        if cross_valid:
            return (tot_loss/tot_token).item(), num_tokens, num_dist * 100 / num_tokens
        else:
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
            cv_loss, ref_len, wer = self.iter_one_epoch(cross_valid=True)

            if self.best_cvloss > cv_loss:
                self.best_cvloss = cv_loss
            train_time = timer.toc()

            cv_msg = ("cv loss: {:.4f} | best cv loss {:.4f} | ref len: {} wer: {:.2f}%").format(cv_loss, self.best_cvloss, ref_len, wer)
            msg = "\n" + "-"*85 + "\n"
            msg += "Epoch {} Validation Summary:\n{}\n".format(self.epoch, cv_msg)
            msg += "Time cost: {:.4f} min".format(train_time/60.)
            msg += "\n" + "-"*85 + '\n'
            logging.info(msg)
            self.tr_loss.append(tr_loss)
            self.cv_loss.append(cv_loss)

            if self.num_last_ckpt_keep:
                utils.cleanup_ckpt(self.exp_dir, self.num_last_ckpt_keep)


class Phone2Char_CTC_GAN_Solver(Phone2Char_CTC_Solver):

    def iter_one_epoch(self, cross_valid=False):
        if cross_valid:
            loader = self.cv_loader
            self.model.eval()
            num_tokens = 0
            num_dist = 0
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
            xs_in, len_xs, _, targets, paddings = (i.to(self.device) for i in data)

            if cross_valid:
                with torch.no_grad():
                    ctc_loss = self.model(xs_in, len_xs, targets, paddings)
                    ctc_logits, len_logits = self.model.get_logits(xs_in, len_xs)
                pred_ids, len_decodeds, _ = self.model.batch_beam_decode(ctc_logits, len_logits, self.decode_fn)
                pred_ids = pred_ids[:, 0, :].cpu().numpy()
                len_decodeds = [i[0] for i in len_decodeds.cpu().tolist()]
                len_targets = (1-paddings).sum(-1)
                num_tokens += len_targets.sum()
                batch_dist = utils.batch_distance(pred_ids, len_decodeds, targets.cpu().numpy(), len_targets.cpu().tolist())
                num_dist += batch_dist
                random.randint(0, len(xs_in))
            else:
                ctc_loss = self.model(xs_in, len_xs, targets, paddings)

            n_token = torch.sum(1-paddings).float()
            tot_token += n_token
            n_sequence = len(utts)
            tot_sequence += n_sequence

            loss = ctc_loss.sum()/n_token
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
                print('Epoch {} | Step {} | Iter {} batch {} \ncur_all_loss: {:.3f} ctc_loss: {:.3f} lr: {:.3e} sent/sec: {:.3f}\n'.format(
                    self.epoch, self.step, niter, list(xs_in.size()),
                    loss, tot_loss / tot_token, list(self.optimizer.param_groups)[0]["lr"], tot_sequence/timer.toc()
                ), flush=True)

        if cross_valid:
            print('ref len: {} wer: {:.2f}%'.format(num_tokens, num_dist * 100 / num_tokens))
        torch.cuda.empty_cache()
        time.sleep(2)

        return (tot_loss/tot_token).item()
