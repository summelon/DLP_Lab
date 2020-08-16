from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class VAE_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, rep_size,
                 num_layers, bidirectional=True,
                 anneal_mode='cyc', anneal_period=1e+3):
        super(VAE_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_direct = 2 if bidirectional else 1
        self.kl_anneal_step = 0
        self.mode = anneal_mode
        self.period = anneal_period

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.cond_embedding = nn.Embedding(4, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,
                            num_layers, bidirectional=bidirectional)
        self.fc_mean = nn.Linear(hidden_size*self.num_direct, rep_size)
        self.fc_var = nn.Linear(hidden_size*self.num_direct, rep_size)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, e_input, cond):
        embedded = self.embedding(e_input)
        cond_embedded = self.cond_embedding(cond.T)
        cond_embedded = cond_embedded.repeat(
                            self.lstm.num_layers*self.num_direct, 1, 1)
        hidden = (cond_embedded, cond_embedded)

        output, hidden = self.lstm(embedded, hidden)
        mu, logvar = self.fc_mean(output), self.fc_var(output)
        z = self._reparameterize(mu, logvar)
        # Loss
        annealing_ratio = torch.tensor(self.kl_cost_annealing())
        loss = self.cal_loss(mu, logvar) * annealing_ratio
        return z, hidden, loss

    def cal_loss(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def kl_cost_annealing(self):
        if self.mode == "mono":
            return (1 if self.kl_anneal_step > self.period
                    else self.kl_anneal_step / self.period)
        elif self.mode == "cyc":
            pd_point = self.kl_anneal_step % self.period
            return (1 if pd_point > (0.5 * self.period)
                    else pd_point / (0.5 * self.period))
        else:
            raise ValueError("KLD annealing mode error!")

    def aneal_step_update(self):
        self.kl_anneal_step += 1


class VAE_Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, rep_size,
                 num_layers, bidirectional=True):
        super(VAE_Decoder, self).__init__()
        self.rep_size = rep_size
        self.num_direct = 2 if bidirectional else 1
        self.cond_embedding = nn.Embedding(4, hidden_size)
        self.lstm = nn.LSTM(rep_size, hidden_size,
                            num_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*self.num_direct, output_size)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, d_input, cond, e_input=None, tf_flag=False):
        cond_embedded = self.cond_embedding(cond.T)
        cond_embedded = cond_embedded.repeat(
                            self.lstm.num_layers*self.num_direct, 1, 1)
        hidden = (cond_embedded, cond_embedded)

        # if self.training:
        #     d_input.add_(torch.randn(d_input.shape, device=d_input.device))

        if tf_flag:
            output, hidden = self.teacher_forcing(e_input, hidden)
        else:
            output, hidden = self.lstm(d_input, hidden)
        output = self.fc(output)
        # Loss
        loss = None if e_input is None else self.cal_loss(output, e_input)
        return output, hidden, loss

    def cal_loss(self, recon_inp, inp):
        inp = torch.flatten(inp)
        recon_inp = recon_inp.reshape(-1, recon_inp.shape[-1])
        return self.criterion(recon_inp, inp)

    def sps2cat(self, data, oh_dim):
        data = torch.nn.functional.one_hot(data, oh_dim)
        return data.type(torch.FloatTensor).to(data.device)

    def teacher_forcing(self, ground_truth, hid):
        assert ground_truth is not None, "Ground truth is None!"
        ground_truth = self.sps2cat(ground_truth, self.rep_size)
        otpts = list()
        for seq in ground_truth:
            seq = torch.unsqueeze(seq, 0)
            otpt, hid = self.lstm(seq, hid)
            otpts.append(otpt)
        return torch.cat(otpts), hid
