import torch as t
import csv
import time
from Utils import accuracy_topk
from LR_Scheduler import CyclicLR, find_learning_rate, CosineAnnealingWarmRestarts
import numpy as np
from torch.optim.lr_scheduler import StepLR,MultiStepLR
from collections import Iterable
import os
from tensorboardX import SummaryWriter
writer = SummaryWriter('./runs/exp1')  # tensorboard --logdir runs


class Trainer(object):
    '''
    Trainer implements the training process in a class.
    It has these variables:
        0. model: the nn architecture
        1. trainloader: the dataloader for training datasets
        2. valloader: the dataloader for validation dataset
        3. lr: the lr_range for training
        4. criterion: the loss for training, default CrossEntropy()
        5. optimizer: to specify the optim method for optimizing the loss, default SGD
        6. use_cuda: a bool variable to Specify whether or not use GPU, default True

    It has these methods:
        1. train(): train procedure
        2. eval(): eval procedure for testing and validation
        3. load(): load a saved model
        4. save(): save a model
        5. lr_find(): to estimate the best learning rate-> to go
    '''
    def __init__(self, model, train_loader,  lr, train_eval_loader=None, val_loader=None, epoches=10, lr_scheduler=None, criterion=None,
                 optimizer=None, use_cuda=True, log_interval=2000):
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._train_eval_loader = train_eval_loader
        if isinstance(lr, list) or isinstance(lr, tuple):
            if len(lr) != 2:
                raise ValueError("expected a list {} , got {}".format(
                    2, len(lr)))
            self._lr = lr
        if lr_scheduler is not None:
            if not isinstance(lr_scheduler, (StepLR, MultiStepLR, CosineAnnealingWarmRestarts, CyclicLR)):
                raise ValueError('The lr_scheduler is not a lr_scheduler object')
        else:
            self._lr_scheduler = None
        self._lr_scheduler = lr_scheduler
        if epoches >0:
            self._EPOCHS = epoches
        if criterion is None:
            self._criterion = t.nn.CrossEntropyLoss()
        else:
            self._criterion = criterion
        if optimizer is None:
            self._optimizer = t.optim.SGD(self._model.parameters(), lr=self._lr[0], momentum=0.9)
        else:
            self._optimizer = optimizer
        if use_cuda:
            self._device = t.device('cuda:0')
        else:
            self._device = t.device('cpu')
        self._logger = self.time_logger
        assert log_interval > 0
        self._log_interval = log_interval
        self._model = model.to(self._device)
        self._record_ind = []
        self._tr_loss = []
        self._val_loss = []
        self._val_acc = []

    def freeze_to(self, n: int) -> None:
        "Freeze layers up to layer group `n`."

        for g in list(self._model.children())[:n]:
            for l in g.parameters():
                l.requires_grad = False
        for g in list(self._model.children())[n:]:
            for l in g.parameters():
                l.requires_grad = True

    def freeze(self):
        self.freeze_to(-1)

    def unfreeze(self):
        self.freeze_to(0)

    def train(self):
        '''
        train procedure of the trainer
        '''
        loss = 0.0
        self._model.train()
        batch_size = self._train_loader.batch_size
        self._logger('Train BEGIN')
        self._logger(self.val())
        for epoch in range(self._EPOCHS):
            if self._lr_scheduler is not None and isinstance(self._lr_scheduler, (StepLR, CosineAnnealingWarmRestarts)):
                self._lr_scheduler.step()

            for i, (inputs, label) in enumerate(self._train_loader, 1):
                if self._lr_scheduler is not None and isinstance(self._lr_scheduler, CyclicLR):
                    self._lr_scheduler.batch_step()
                inputs, label = inputs.to(self._device), label.long().squeeze(1).to(self._device)
                self._optimizer.zero_grad()
                outputs = self._model(inputs)
                iter_loss = self._criterion(outputs, label)
                iter_loss.backward()
                self._optimizer.step()
                loss += iter_loss.item()

                if (epoch*len(self._train_loader)+i) % self._log_interval == 0:
                    total_num = self._log_interval * batch_size
                    _log_loss = loss/self._log_interval
                    val_acc, val_loss = self.val()
                    # record the loss of train and val
                    self._record_ind.append(epoch*len(self._train_loader)+i)
                    self._tr_loss.append(_log_loss)
                    self._val_loss.append(val_loss)
                    self._val_acc.append(val_acc)
                    # print to window for monitoring
                    self._logger('EPOCH[{}] ITER[{}] Train Loss:{:.6f}'.format(epoch, i, _log_loss))
                    self._logger('EPOCH[{}] ITER[{}] VAL Loss:{:.6f}, acc1:{:.6f}, acc3:{:.6f}, total sample:{}'
                                      .format(epoch, i, val_loss, val_acc[0], val_acc[2], total_num))
                    loss = 0.0
                    '''
                    for name, param in self._model.named_parameters():
                        if 'bn' not in name:
                            writer.add_histogram(name, param.clone().cpu().data.numpy(), 0)
                    '''
            self.save('./', epoch)

        return self._model

    def val(self):
        if self._val_loader is None:
            raise ValueError('The val_loader is None')
        val_acc, val_loss = self.eval(self._val_loader)
        return val_acc, val_loss

    def train_eval(self):
        if self._train_eval_loader is None:
            raise ValueError('The _train_eval_loader is None')
        train_acc, train_loss = self.eval(self._train_eval_loader)
        return train_acc, train_loss

    def eval(self, dataloader):
        '''
        for input type is iterable
        :return: val_loss, val_acc
        '''
        if dataloader is None or not isinstance(dataloader, Iterable):
            raise ValueError('The data_loader is None or is not Iterable')
        self._model.eval()
        losses = 0.0
        accs = np.array([0.0, 0.0, 0.0])
        nums = len(dataloader)
        batch_size = dataloader.batch_size

        with t.no_grad():
            for i, (inputs, label) in enumerate(dataloader, 1):
                inputs, label = inputs.to(self._device), label.long().squeeze(1).to(self._device)
                outputs = self._model(inputs)
                loss = self._criterion(outputs, label)
                losses += loss.item()
                res = accuracy_topk(outputs, label, (1, 2, 3))
                accs += res
        return (accs/(nums*batch_size)).tolist(), losses / nums

    def predict(self, inputs):
        if inputs is None:
            raise ValueError('The inputs is None')
        if not isinstance(inputs, t.Tensor):
            raise ValueError('The inputs should be tensor type')

        with t.no_grad():
            inputs, label = inputs.to(self._device)
            outputs = self._model(inputs)
            return outputs

    def save(self, output_path, epoch=0, i=0):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        file_name = 'Epoch{}_Iter{}.pkl'.format(epoch, i)
        file_path = os.path.join(output_path, file_name)
        t.save(self._model.state_dict(), file_path)

    def load(self, model_path):
        if not os.path.exists(model_path):
            raise ValueError('The given path is not exists.')
        state_dict = t.load(model_path)
        self._model.load_state_dict(state_dict)

    def time_logger(self,*args):
        ime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        ime_str += '  '
        for v in args:
            ime_str += str(v)
        with open('time_log.log', 'a+') as logger:
            logger.write(ime_str + '\n')
        print(ime_str)


