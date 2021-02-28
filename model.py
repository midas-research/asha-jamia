import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        # mask = torch.ones(attentions.size(), requires_grad=True)
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        if attentions.dim() == 1:
            attentions = attentions.unsqueeze(1)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class MyLSTM(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, lstm_layer=2, dropout=0.2):
        super(MyLSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.lstm1 = nn.LSTM(input_size=self.embedding_dim,
                             hidden_size=hidden_dim,
                             num_layers=lstm_layer,
                             bidirectional=True)
        self.atten1 = Attention(hidden_dim * 2, batch_first=True)  # 2 is bidrectional

    def forward(self, x, x_len):
        x = self.dropout(x)

        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.lstm1(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        #         print(x.shape)
        x, _ = self.atten1(x, lengths)  # skip connect

        return x, _


class AdvRedditModel(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, lstm_layer=2, dropout=0.2, eps=0.05):
        super(AdvRedditModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.eps = eps

        self.fc_1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc_2 = nn.Linear(hidden_dim * 2, 5)

        self.historic_model = MyLSTM(self.embedding_dim, self.hidden_dim, lstm_layer, dropout)

    def get_pred(self, feat):
        feat = self.fc_1(self.dropout(feat))
        return self.fc_2(feat)

    def forward(self, tweets, lengths, labels):
        h, _ = self.historic_model(tweets, lengths)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        e = self.get_pred(h)

        adv_ex = None
        if self.training:
            pred_loss = F.cross_entropy(e, labels, reduction='sum')
            h.retain_grad()
            pred_loss.backward(retain_graph=True)

            grad = h.grad.detach()
            grad = F.normalize(grad, dim=0)
            adv = h + self.eps * grad

            adv_ex = self.get_pred(adv)

        return e, adv_ex
