import time
import torch
from torch.nn.utils import clip_grad_norm_

import utils
from solvers import Solver


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
