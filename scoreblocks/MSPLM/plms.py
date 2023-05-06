import torch
from torch import nn
from transformers import AutoModel
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(7)


class mainplm(nn.Module):
    def __init__(self, args):
        super(mainplm, self).__init__()
        self.args = args
        self.plm_batch_size = 1
        self.plm = AutoModel.from_pretrained(self.args['plm'])

        for param in self.plm.embeddings.parameters():
            param.requires_grad = False
        for i in range(11):
            for param in self.plm.encoder.layer[i].parameters():
                param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.plm.config.hidden_size, 1)
        )
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, device='cpu'):
        plm_output = torch.zeros(size=(document_batch.shape[0],
                                       min(document_batch.shape[1], self.plm_batch_size),
                                       self.plm.config.hidden_size),
                                 dtype=torch.float, device=device)
        for doc_id in range(document_batch.shape[0]):
            all_plm_output = self.plm(document_batch[doc_id][:self.plm_batch_size, 0],  # [1, 512]
                                      token_type_ids=document_batch[doc_id][:self.plm_batch_size, 1],
                                      attention_mask=document_batch[doc_id][:self.plm_batch_size, 2])
            plm_output[doc_id][:self.plm_batch_size] = all_plm_output.last_hidden_state[0][0].unsqueeze(0) # deberta:all_plm_output.last_hidden_state[0][0].unsqueeze(0) or bert:all_plm_output[1]
        prediction = self.mlp(plm_output.view(plm_output.shape[0], -1))
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class chunkplm(nn.Module):
    def __init__(self, args):
        super(chunkplm, self).__init__()
        self.args = args
        self.plm = AutoModel.from_pretrained(self.args['PLM'])

        for param in self.plm.embeddings.parameters():
            param.requires_grad = False
        for i in range(12):
            for param in self.plm.encoder.layer[i].parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(self.plm.config.hidden_size, self.plm.config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.plm.config.hidden_size, 1)
        )
        self.w_omega = nn.Parameter(torch.Tensor(self.plm.config.hidden_size, self.plm.config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, self.plm.config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.plm.config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, device='cpu', plm_batch_size = 0):
        # output的chunk数是wordpiece / chunk_size
        plm_output = torch.zeros(size=(document_batch.shape[0],
                                       min(document_batch.shape[1],
                                           plm_batch_size),
                                       self.plm.config.hidden_size), dtype=torch.float, device=device)
        for doc_id in range(document_batch.shape[0]): # document_batch torch.Size([2, 12, 3, 90])
            plm_output[doc_id][:plm_batch_size] = self.dropout(
                self.plm(document_batch[doc_id][:plm_batch_size, 0],
                         token_type_ids=document_batch[doc_id][
                                        :plm_batch_size, 1],
                         attention_mask=document_batch[doc_id][
                                        :plm_batch_size, 2])[1])
        output, (_, _) = self.lstm(plm_output.permute(1, 0, 2))
        output = output.permute(1, 0, 2)
        # (batch_size, seq_len, num_hiddens)
        attention_w = torch.tanh(torch.matmul(output, self.w_omega) + self.b_omega)
        attention_u = torch.matmul(attention_w, self.u_omega)  # (batch_size, seq_len, 1)
        attention_score = F.softmax(attention_u, dim=1)  # (batch_size, seq_len, 1)
        attention_hidden = output * attention_score  # (batch_size, seq_len, num_hiddens)
        attention_hidden = torch.sum(attention_hidden, dim=1)  # 加权求和 (batch_size, num_hiddens)
        prediction = self.mlp(attention_hidden)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction
