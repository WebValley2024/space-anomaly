import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
import matplotlib.pyplot as plt
import shutil





def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

class Solver(object):

    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode="train",
                                               dataset=self.dataset,step=self.step, modality=self.mode)

        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode="test",
                                              dataset=self.dataset,step=self.step, modality=self.mode)
        
        print("train_loader: ", len(self.train_loader))
        print("test_loader: ", len(self.test_loader))
        
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.config = config
        self.loss_save_4 = []
        
        self.savedir = "test_"+str(self.ID)
        
        if self.mode == 'train':

            if not os.path.exists(self.savedir):
                os.makedirs(self.savedir)
            else:
                inpt = input("already exists, do you want to overwrite? y,n  ")

                if inpt == 'y':
                    shutil.rmtree(self.savedir)
                    os.makedirs(self.savedir)
                else:
                    raise Exception("folder already exists")
        
        logfile = os.path.join(self.savedir, "log.txt")
        self.log = open(logfile, "a")

    def build_model(self):

        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=self.e_layers, n_heads=self.n_heads, d_ff=self.d_ff, d_model=self.d_model, dropout=self.dropout) # e_layers=3, n_heads=8, d_ff=512, d_model=512, dropout=0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()


    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path

        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            
            for i, input_data in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)
                
                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()

                self.optimizer.step()
            
            self.loss_save_4.append(np.average(loss1_list))


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, i, train_loss))
            
            #adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
        
        torch.save(self.model.state_dict(), os.path.join(self.savedir, str(self.dataset) + '_checkpoint.pth'))
        
        plt.plot(self.loss_save_4)
        plt.savefig(os.path.join(self.savedir, 'loss.png'))

        self.log.write(str(self.model))
        self.log.write('\n\n\n')
        self.log.write('loss: ' + str(self.loss_save_4))
        self.log.write('\n\n')
        self.log.write('time: ' + str(time.time() - time_now))
        self.log.write('\n\n')

        for i in self.config.keys():
            self.log.write(str(i) + ' : ' + str(self.config[i]) + '\n')



    def test(self):
        
        if self.test_model is not None:
            
            self.model.load_state_dict(
                torch.load(
                    self.test_model))
        
        else:
            
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.savedir), str(self.dataset) + '_checkpoint.pth')))
        
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        
        for i, input_data in enumerate(self.train_loader):
            
            input = input_data.float().to(self.device)

            output, series, prior, _ = self.model(input)

            
            loss = torch.mean(criterion(input, output), dim=-1)
            
            series_loss = 0.0
            prior_loss = 0.0
            
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            
            attens_energy.append(cri)
            #print("cri: ", cri.shape,"i: ",i)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        
        #save to a csv
        np.savetxt(os.path.join(self.savedir,"train_energy.csv"), train_energy, delimiter=",")

        # (2) stastic on the train set
        attens_energy = []
        
        for i, input_data in enumerate(self.test_loader):
            
            input = input_data.float().to(self.device)
            print("input: ", input.shape)
            output, series, prior, _ = self.model(input)
            print("output: ", output.shape)
            
            loss = torch.mean(criterion(input, output), dim=-1)
            
            series_loss = 0.0
            prior_loss = 0.0
            
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            
            attens_energy.append(cri)
            print("cri: ", cri.shape,"i: ",i)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        
        #save to a csv
        np.savetxt(os.path.join(self.savedir,"train_energy_test.csv"), train_energy, delimiter=",")
