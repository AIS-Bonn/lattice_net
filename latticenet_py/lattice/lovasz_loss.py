import torch
import torch.nn as nn
from torch.autograd import Function


#from https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/LovaszSoftmax/lovasz_loss.py

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, ignore_index, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction
        self.ignore_index=ignore_index

    def prob_flatten(self, input, target):
        assert input.dim() in [2]
        num_class = input.size(1)
        # if input.dim() == 4:
            # input = input.permute(0, 2, 3, 1).contiguous()
        input_flatten = input.view(-1, num_class)
        # elif input.dim() == 5:
        #     input = input.permute(0, 2, 3, 4, 1).contiguous()
        #     input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            if c!=self.ignore_index:
                target_c = (targets == c).float()
                nr_pixels_gt_for_this_class=target_c.sum()
                if nr_pixels_gt_for_this_class==0:
                    continue #as described in the paper, we skip the penalty for the classes that are not present in this sample
                if num_classes == 1:
                    input_c = inputs[:, 0]
                else:
                    input_c = inputs[:, c]
                loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
                loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
                target_c_sorted = target_c[loss_index]
                losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        inputs=inputs.exp()
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses
