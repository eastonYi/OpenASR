import os
import logging
import torch

import utils
import schedule


class Solver(object):
    def __init__(self, model, config, tr_loader, cv_loader):
        self.config = config
        self.tr_loader = tr_loader
        self.cv_loader = cv_loader

        self.model = model
        if config['multi_gpu'] == True:
            self.model_to_pack = self.model.module
        else:
            self.model_to_pack = self.model

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.num_epoch = config['num_epoch']
        self.exp_dir = config['exp_dir']
        self.print_inteval = config['print_inteval']

        self.accumulate_grad_batch = config['accumulate_grad_batch']
        self.init_lr = config['init_lr']
        self.grad_max_norm = config['grad_max_norm']
        self.label_smooth = config['label_smooth']

        self.num_last_ckpt_keep = None
        if "num_last_ckpt_keep" in config:
            self.num_last_ckpt_keep = config['num_last_ckpt_keep']

        self.lr_scheduler = schedule.get_scheduler(config['lr_scheduler'])
        # Solver state
        self.epoch = 0
        self.step = 0
        self.tr_loss = []
        self.cv_loss = []
        self.lr = self.init_lr

        if config['optimtype'] == "sgd":
            self.optimizer = torch.optim.SGD(self.model_to_pack.parameters(), lr=self.lr, momentum=0.9)
        elif config['optimtype'] == "adam":
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
            "Solver_config": dict(self.config),
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
        self.lr_scheduler.restore_state(pkg['scheduler_state'])

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
