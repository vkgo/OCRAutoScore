from torch import nn
import torch


class multi_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weight = [args['w1'], args['w2'], args['w3']]
        self.MSE = nn.MSELoss().to(device=self.args['device'])
        self.CosineEmbeddingLoss = nn.CosineEmbeddingLoss().to(device=self.args['device'])
        self.MarginRankingLoss = nn.MarginRankingLoss().to(device=self.args['device'])

    def forward(self, y_trues, y_preds):
        """
        input must be [batchsize, 1]
        """
        m, n = y_trues.size()
        batchsize = y_preds.shape[0]
        mseloss = self.MSE(y_trues, y_preds)
        simloss = torch.max(torch.tensor(0., device=self.args['device']), self.CosineEmbeddingLoss(y_trues.resize(n, m), y_preds.resize(n, m), torch.ones(batchsize, dtype=torch.int, device=self.args['device'])))

        # count rankloss
        rankloss = torch.tensor(0., device=self.args['device'])
        for i in range(batchsize):
            for j in range(i + 1, batchsize):
                input1_pred = y_preds[i]
                input2_pred = y_preds[j]
                input1_true = y_trues[i]
                input2_true = y_trues[j]
                target = 0
                if input1_true > input2_true:
                    target = 1
                elif input1_true < input2_true:
                    target = -1
                else:
                    if input1_pred > input2_pred:
                        target = -1
                    elif input1_pred < input2_pred:
                        target = 1
                target = torch.tensor([target], device=self.args['device'])
                rankloss += self.MarginRankingLoss(input1_pred, input2_pred, target)

        print(f'mseloss{self.weight[0] * mseloss}\tsimloss{self.weight[1] * simloss}\trankloss{self.weight[2] * rankloss}')

        return self.weight[0] * mseloss + self.weight[1] * simloss + self.weight[2] * rankloss


if __name__ == '__main__':
    """
    used to debug
    """
    # x1 = torch.randn(8, 1, device='cuda')
    # x2 = torch.randn(8, 1, device='cuda')
    x1 = torch.tensor([2, 3, 4, 5, 6, 7], device='cuda').resize(6, 1)
    x2 = torch.tensor([1.1, 3.9, 4.36, 5.35, 6.1, 6.9], device='cuda').resize(6, 1)
    print(x1, x2)
    loss = multi_loss(args={'device': 'cuda', 'w1': 40, 'w2':100, 'w3':1})
    print(loss(x1, x2))