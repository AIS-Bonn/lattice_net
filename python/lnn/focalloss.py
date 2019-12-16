# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()



#better focaloss adapted from from here  https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b
#It works like  crossEntropy so the input should be raw output of the model, without passing through softmax or anything
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction='mean')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather( F.softmax(input_, 1) , 1, target.unsqueeze(1)) 
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            return loss
        # return torch.mean(loss) if self.reduction == 'mean'
            # else torch.sum(loss) if self.reduction == 'sum'
            # else loss