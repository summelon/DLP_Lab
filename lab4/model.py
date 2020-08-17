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
        self.word_embedding = nn.Embedding(input_size, hidden_size)
        self.cond_embedding = nn.Embedding(4, 8)
        self.lstm = nn.LSTM(hidden_size, hidden_size,
                            num_layers, bidirectional=bidirectional)
        self.hidden_fc = nn.Linear(8, hidden_size)
        self.fc_mean = nn.Linear(hidden_size*self.num_direct, rep_size)
        self.fc_var = nn.Linear(hidden_size*self.num_direct, rep_size)

    def forward(self, e_input, cond):
        embedded = self.word_embedding(e_input)
        hidden = self.init_hidden(cond)
        hidden = self.hidden_fc(hidden)

        output, hidden = self.lstm(embedded, (hidden, hidden))
        mu, logvar = self.fc_mean(output), self.fc_var(output)
        z = self._reparameterize(mu, logvar)
        # Loss
        annealing_ratio = torch.tensor(self.kl_cost_annealing())
        loss = self.cal_loss(mu, logvar) * annealing_ratio
        return z, hidden, loss

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def init_hidden(self, tense):
        repeat = self.num_direct * self.lstm.num_layers
        cond_embedded = self.cond_embedding(tense.T).repeat(repeat, 1, 1)
        return cond_embedded

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
    def __init__(self, hidden_size, output_size, dropout_rate,
                 num_layers, bidirectional=True):
        super(VAE_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_direct = 2 if bidirectional else 1
        self.word_embedding = nn.Embedding(output_size, hidden_size)
        self.cond_embedding = nn.Embedding(4, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            bidirectional=bidirectional, dropout=dropout_rate)
        self.hidden_fc = nn.Linear(hidden_size+17*output_size, hidden_size)
        self.fc = nn.Linear(hidden_size*self.num_direct, output_size)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, latent, cond, e_input=None, tf_flag=False):
        hidden = self.init_hidden(latent, cond)
        hidden = self.hidden_fc(hidden)

        if tf_flag:
            output, hidden = self.teacher_forcing(e_input, (hidden, hidden))
        else:
            output, hidden = self.normal_iterative(latent, (hidden, hidden))
        # Loss
        loss = None if e_input is None else self.cal_loss(output, e_input)
        return output, hidden, loss

    def init_hidden(self, latent, cond):
        repeat = self.num_direct*self.lstm.num_layers
        cond_embedded = self.cond_embedding(cond.T.repeat(repeat, 1))
        latent = latent.transpose(1, 0)
        latent = latent.reshape(1, latent.shape[0], -1).repeat(repeat, 1, 1)
        cat_hidden = torch.cat([latent, cond_embedded], dim=-1)
        return cat_hidden

    def cal_loss(self, recon_inp, inp):
        inp = torch.flatten(inp)
        recon_inp = recon_inp.reshape(-1, recon_inp.shape[-1])
        return self.criterion(recon_inp, inp)

    def sps2cat(self, data, oh_dim):
        data = torch.nn.functional.one_hot(data, oh_dim)
        return data.to(data.device)

    def teacher_forcing(self, ground_truth, hid):
        assert ground_truth is not None, "Ground truth is None!"
        word_embedded = self.word_embedding(ground_truth)
        otpts = list()
        for seq in word_embedded:
            seq = torch.unsqueeze(seq, 0)
            otpt, hid = self.lstm(seq, hid)
            otpt = self.fc(otpt)
            otpts.append(otpt)
        return torch.cat(otpts), hid

    def normal_iterative(self, latent, hid):
        sos = torch.tensor([[1]], device=latent.device)
        single_inp = sos.repeat(1, latent.shape[1])
        otpts = list()
        # Max length is 15, plus sos & eos
        for i in range(17):
            seq = self.word_embedding(single_inp)
            otpt, hid = self.lstm(seq, hid)
            otpt = self.fc(otpt)
            single_inp = torch.argmax(otpt, dim=-1).detach()
            single_inp = single_inp.type(torch.long).to(latent.device)
            otpts.append(otpt)
        return torch.cat(otpts), hid
