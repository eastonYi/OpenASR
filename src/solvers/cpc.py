import time
import torch
from torch.nn.utils import clip_grad_norm_

import utils
from solvers import Solver


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
