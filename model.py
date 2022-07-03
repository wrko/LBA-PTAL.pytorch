import numpy as np
import time
import os

import torch
import torch.nn as nn
from torch import optim
from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

from config import hp, path
from utils.kcluster import subaction_names


# encoder and decoder modules
@variational_estimator
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        # unidirectional lstm:
        self.lstm = BayesianLSTM(hp.user_pose_dim, hp.lstm_hidden_size)
        # create z from lstm's last output:
        self.fc_z = nn.Linear(hp.lstm_hidden_size, len(subaction_names))
        # active dropout:
        self.train()

    def forward(self, inputs, hidden_cell=None):
        batch_size = len(inputs)
        if hidden_cell is None:
            # then must init with zeros
            if hp.use_cuda:
                hidden = torch.zeros(batch_size, hp.lstm_hidden_size).cuda()
                cell = torch.zeros(batch_size, hp.lstm_hidden_size).cuda()
            else:
                hidden = torch.zeros(batch_size, hp.lstm_hidden_size)
                cell = torch.zeros(batch_size, hp.lstm_hidden_size)
            hidden_cell = (hidden, cell)
        _, (hidden, cell) = self.lstm(inputs.float(), hidden_cell)
        z = self.fc_z(hidden)
        return z


class PHAR():
    def __init__(self):
        if hp.use_cuda:
            self.model = LSTMModel().cuda()
        else:
            self.model = LSTMModel()
        self.optimizer = optim.Adam(self.model.parameters(), hp.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.time = time.time()

    def train(self, epoch, train_data_loader, test_data_loader):
        os.makedirs(path.lstm_model, exist_ok=True)
        file = open(os.path.join(path.lstm_model, f"{self.time}.log"), "a")
        # train:
        train_loss = 0.0
        for inps, outs in train_data_loader:
            loss = self.step(inps, outs, b_train=True, n_samples=100)
            train_loss += loss.item()
        train_loss = train_loss / len(train_data_loader)
        # test:
        with torch.no_grad():
            test_acc = 0
            for inps, outs in test_data_loader:
                _, acc, _ = self.step(inps, outs, b_train=False, n_samples=100)
                test_acc += acc
        test_acc = test_acc / len(test_data_loader)
        # save:
        line = f'epoch\t{epoch}\ttrain_loss\t{train_loss:0.4f}\ttest_acc\t{test_acc:0.4f}'
        file.write(line + '\n')
        if epoch % hp.save_epochs == 0:
            print(line)
            self.save(epoch)
        file.close()

    def step(self, inps, outs, b_train=False, n_samples=100):
        # make input and output as tensor
        b_numpy = type(inps) is (np.ndarray or list)
        if b_numpy:
            inps = torch.FloatTensor(inps)
            outs = torch.FloatTensor(outs)
        # dropout for training mode
        self.model.train(b_train)
        # prepare data
        if hp.use_cuda:
            inps = inps.cuda()
            outs = outs.cuda()
        # back-propagation
        if b_train:
            loss = self.model.sample_elbo(inputs=inps, labels=outs, criterion=self.loss_fn,
                                          sample_nbr=3, complexity_cost_weight=1/50000)
            loss.backward()
            self.optimizer.step()
            return loss
        # evaluate
        else:
            preds = [torch.argmax(self.model(inps), dim=1) for _ in range(n_samples)]
            preds = torch.stack(preds)
            correct = preds == outs
            acc = 100 * (float(correct.sum()) / outs.size(0) / n_samples)
            stds = correct.float().std(dim=1)
            pred = preds[0]
            if b_numpy:
                if hp.use_cuda:
                    pred = pred.cpu()
                pred = pred.detach().numpy()
            return pred, acc, stds.mean(dim=0)

    def save(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(path.lstm_model, 'LSTMModel_epoch_%d.pth' % epoch))

    def load(self, epoch):
        device = torch.device('cpu')
        saved_model = torch.load(os.path.join(path.lstm_model, 'LSTMModel_epoch_%d.pth' % epoch), map_location=device)
        self.model.load_state_dict(saved_model)

    def test(self, epoch, data_loader):
        # load trained model
        self.load(epoch)

        # calculate uncertainty
        acc_mean = 0.
        uncertainty_mean = 0.
        for inps, outs in data_loader:
            # predict
            _, acc, uncertainty = self.step(inps, outs, b_train=False, n_samples=100)
            acc_mean += acc
            uncertainty_mean += uncertainty
        return acc_mean / len(data_loader), uncertainty_mean / len(data_loader)
