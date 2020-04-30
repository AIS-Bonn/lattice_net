# https://github.com/CoinCheung/pytorch-loss/blob/master/dice_loss.py
# https://github.com/pytorch/pytorch/issues/1249
# https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/5

import torch
import torch.nn as nn

class GeneralizedSoftDiceLoss(nn.Module):
    def __init__(self,
                 p=1,
                 smooth=1,
                 reduction='mean',
                 weight=1,
                 ignore_index=None):
        super(GeneralizedSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
        # self.weight = None if weight is None else torch.tensor(weight)
        self.weight = None 
        self.ignore_index = ignore_index

    def forward(self, output, target):
        # '''
        # args: logits: tensor of shape (N, C, H, W)
        # args: label: tensor of shape(N, H, W)
        # '''
        # # overcome ignored label
        # ignore = label.data.cpu() == self.ignore_lb
        # label[ignore] = 0
        # lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        # ignore = ignore.nonzero()
        # _, M = ignore.size()
        # a, *b = ignore.chunk(M, dim=1)
        # lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0

        # # compute loss
        # probs = torch.sigmoid(logits)
        # numer = torch.sum((probs*lb_one_hot), dim=(2, 3))
        # denom = torch.sum(probs.pow(self.p)+lb_one_hot.pow(self.p), dim=(2, 3))
        # if not self.weight is None:
        #     numer = numer * self.weight.view(1, -1)
        #     denom = denom * self.weight.view(1, -1)
        # numer = torch.sum(numer, dim=1)
        # denom = torch.sum(denom, dim=1)
        # loss = 1 - (2*numer+self.smooth)/(denom+self.smooth)

        # if self.reduction == 'mean':
        #     loss = loss.mean()
        # return loss

        # output = output.exp()
        # encoded_target = output.data.clone().zero_()
        # if self.ignore_index is not None:
        #     # mask of invalid label
        #     mask = target == self.ignore_index
        #     # clone target to not affect the variable ?
        #     filtered_target = target.clone()
        #     # replace invalid label with whatever legal index value
        #     filtered_target[mask] = 0
        #     # one hot encoding
        #     encoded_target.scatter_(1, filtered_target.unsqueeze(1), 1)
        #     # expand the mask for the encoded target array
        #     mask = mask.unsqueeze(1).expand(output.data.size())
        #     # apply 0 to masked pixels
        #     encoded_target[mask] = 0
        # else:
        #     encoded_target.scatter_(1, target.unsqueeze(1), 1)
        # # encoded_target = torch.Tensor(encoded_target)

        # assert output.size() == encoded_target.size(), "Input sizes must be equal."
        # assert output.dim() == 4, "Input must be a 4D Tensor."

        # numerator = (output * encoded_target).sum(dim=3).sum(dim=2)
        # denominator = output.pow(2) + encoded_target
        # if ignore_index is not None:
        #     # exclude masked values from den1
        #     denominator[mask] = 0

        # dice = 2 * (numerator / denominator.sum(dim=3).sum(dim=2)) * self.weight
        # return dice.sum() / dice.size(0)



        """
        output : NxCxHxW Variable
        target :  NxHxW LongTensor
        weights : C FloatTensor
        ignore_index : int index to ignore from loss
        """

        #recieve a vector of num_poins x classes

        # output=output.transpose(0,1).unsqueeze(0).unsqueeze(3) #1 x classes x num_points x 1
        # target=target.unsqueeze(0).unsqueeze(2) #1 x num_points x 1
        

        # # smooth=1.0
        # smooth=0.0
        # # smooth=0.01
        # eps = 0.000001

        # output = output.exp()
        # encoded_target = output.detach() * 0
        # if self.ignore_index is not None:
        #     mask = target == self.ignore_index
        #     target = target.clone()
        #     target[mask] = 0
        #     encoded_target.scatter_(1, target.unsqueeze(1), 1)
        #     mask = mask.unsqueeze(1).expand_as(encoded_target)
        #     encoded_target[mask] = 0
        # else:
        #     encoded_target.scatter_(1, target.unsqueeze(1), 1)

        # if self.weight is None:
        #     self.weight = torch.ones(output.shape[1]).to("cuda")
        #     self.weight[self.ignore_index]=0

        # # intersection = output * encoded_target
        # # numerator = 2 * intersection.sum(0).sum(1).sum(1)
        # # denominator = output + encoded_target

        # # if self.ignore_index is not None:
        # #     denominator[mask] = 0
        # # denominator = denominator.sum(0).sum(1).sum(1) + eps
        # # loss_per_channel = self.weight * (1 - (numerator / denominator))

        # # print("loss per channel is ", loss_per_channel)

        # # return loss_per_channel.sum() / output.size(1)


        # ##attempt 2 https://dev.to/andys0975/what-is-dice-loss-for-image-segmentation-3p85
        # product=output * encoded_target
        # intersection=product.sum(0).sum(1).sum(1)
        # denominator = output + encoded_target
        # if self.ignore_index is not None:
        #     denominator[mask] = 0
        # denominator = denominator.sum(0).sum(1).sum(1)
        # coefficient= (2 * intersection +smooth) / ( denominator + smooth+eps )
        # loss_per_channel = self.weight * (1 - coefficient)

        # print("loss per channel is ", loss_per_channel)
        
        # nr_classes=output.size(1)
        # if self.ignore_index is not None:
        #     nr_classes=nr_classes-1
        # return loss_per_channel.sum() / nr_classes







        #attempt 3 https://stackoverflow.com/questions/56508089/dice-loss-in-3d-pytorch
        #and also https://forums.fast.ai/t/understanding-the-dice-coefficient/5838
        nr_points=output.shape[0]
        nr_classes=output.shape[1]
        # smooth=100.0/nr_classes
        smooth=0.0
        eps = 0.000001


        output = output.exp()
        # if self.ignore_index is not None:
        #     mask = target.ne_(self.ignore_index)
        #     mask.requires_grad = False

        #     output = output * mask
        #     target = target * mask

        output = output.view(nr_points, -1)
        target = target.view(nr_points, -1)
        # print("output has shape ", output.shape)
        # print("target has shape ", target.shape)

        #the output is nr_points x nr_classes, it has for each point the probability that the point is of a certain class
        #the target is nr_pointx x 1, it has for every point, directly the label.
        # we make the target also be a nr_points x nr_classes where the column indicated by target will be set to 1
        encoded_target = torch.cuda.FloatTensor(nr_points, nr_classes).fill_(0)
        encoded_target.scatter_(1, target, 1) # arguments are dim, index, sec

        #target may have a lot of points that are background and therefore when they are one hot encoded, they will create a lots of vector of type [1,0,0,0..] ( if the backgorund idx is 0 for example)
        # if self.ignore_index is not None: 
            # mask=torch.cuda.FloatTensor(nr_points, nr_classes).fill_(1)
            # encoded_target[:,self.ignore_index]=0 
            #setting the whole column to zero will efectivelly make so that the intersection for that ignored class will always be zero
            #the union in that case will be whatever the networks votes for there, which sould also be set to zero but it doesnt matter because the loss incentivizes the network to vote for valid classes, it it start voting for background, the union starts to increase for all classe which means the loss will increase
            # output[:,self.ignore_index]=0
            #in that case also the union for that column will be zero

        if self.weight is None:
            self.weight = torch.ones(nr_classes).to("cuda")
        self.weight[self.ignore_index]=0

        #if we are ignoring a certain idx, then the intersection and union for it will be set to zero
        # mask_ignore_idx=torch.ones(nr_classes).to("cuda")
        # mask_ignore_idx[self.ignore_index]=0

        # here we see that the jacarrd index is better than the dice https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a

        target=target.float()
        intersection_per_class = (output * encoded_target).sum(0)  #summ over all the points, the points that have a 1 exactly where the gt is 1, will count as an intersection
        union_per_class= (output + encoded_target).sum(0)
        # union_per_class-=intersection_per_class #to acount for the fasct that the intersection gets counted twice when we sum the output and the target
        # print("intersection_per_class is ", intersection_per_class)
        # print("union_per_class is ", union_per_class)

        #you can also avoid the sum and get a dice coeff per class which you can then use some weighting upon
        # intersection_per_class=mask_ignore_idx*intersection_per_class
        # union_per_class=mask_ignore_idx*union_per_class
        dice_coeff_per_class= (2*intersection_per_class+smooth) / ( union_per_class + smooth +eps )
        loss_per_class=1-dice_coeff_per_class
        loss_per_class=self.weight*loss_per_class
        # print("loss per class is ", loss_per_class)
        loss= loss_per_class.sum()/nr_classes



        return loss









# def dice_loss(output, target, weights=1, ignore_index=None):
#     output = output.exp()
#     encoded_target = output.data.clone().zero_()
#     if ignore_index is not None:
#         # mask of invalid label
#         mask = target == ignore_index
#         # clone target to not affect the variable ?
#         filtered_target = target.clone()
#         # replace invalid label with whatever legal index value
#         filtered_target[mask] = 0
#         # one hot encoding
#         encoded_target.scatter_(1, filtered_target.unsqueeze(1), 1)
#         # expand the mask for the encoded target array
#         mask = mask.unsqueeze(1).expand(output.data.size())
#         # apply 0 to masked pixels
#         encoded_target[mask] = 0
#     else:
#         encoded_target.scatter_(1, target.unsqueeze(1), 1)
#     encoded_target = Variable(encoded_target)

#     assert output.size() == encoded_target.size(), "Input sizes must be equal."
#     assert output.dim() == 4, "Input must be a 4D Tensor."

#     numerator = (output * encoded_target).sum(dim=3).sum(dim=2)
#     denominator = output.pow(2) + encoded_target
#     if ignore_index is not None:
#         # exclude masked values from den1
#         denominator[mask] = 0

#     dice = 2 * (numerator / denominator.sum(dim=3).sum(dim=2)) * weights
#     return dice.sum() / dice.size(0)