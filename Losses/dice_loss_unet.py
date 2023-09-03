import torch
from torch import nn
from torch.autograd import Variable

class DiceLoss(nn.Module):

    def __init__(self, device, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.device = device

    def forward(self, inputs, target):
        # eps = 0.0001
        # logits = torch.sigmoid(logits)
        # num = target.size(0)
        # m1 = logits.view(num, -1).type(torch.FloatTensor).to(self.device)
        # m2 = target.view(num, -1).type(torch.FloatTensor).to(self.device)
        # self.inter = torch.mul(m1, m2)
        # m1sqrt = torch.mul(m1, m1)
        # m2sqrt = torch.mul(m2, m2)
        # self.union = torch.sum(m1sqrt, dim=1).type(torch.FloatTensor).to(self.device) + \
        #              torch.sum(m2sqrt, dim=1).type(torch.FloatTensor).to(self.device) + eps
        # t = (2 * torch.sum(self.inter, 1) + eps) / self.union
        # t = (2 * torch.sum(self.inter) + eps) / self.union
        # loss = 1 - (torch.sum(t) / num)

        smooth = 1
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # inputs = torch.ge(inputs, 0.5).float()    #or
        # inputs = (inputs > 0.5).float()   #binary output (1 if > 0.5, 0 otherwise)
        # inputs = Variable(inputs, requires_grad=True)
        # print(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = target.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        loss = 1 - dice

        return loss