

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
import re


class QAVectorLossFunction():
    def __init__(self, cross_entropy: CrossEntropyLoss, numb_of_indices=4):
        self.cross_entropy = cross_entropy
        self.numb_of_indices = numb_of_indices

    def __call__(self, input: Tensor, target: Tensor):
        #print(input.size())
        #print(target.size())
        loss_sum = torch.zeros(1, requires_grad=True).to(input.device)
        target = torch.permute(target, (1, 0)) # flip target dims <batch, indices> -> <indices, batch>
        for inp, targ in zip(input, target): # iterate over indices
            #print(inp.size())
            #print(targ.size())
            loss_sum = loss_sum.add(self.cross_entropy(inp, targ))
        #r = self.cross_entropy(input, target)
        #print(loss_sum.div_(4))
        #exit()
        return loss_sum.div(4)
        '''loss_sum = torch.zeros(1)
        for i in range(self.numb_of_indices):
            print(input[i].size(), target[i].size())
            loss_sum.add_(self.cross_entropy(input[i], target[i]))
        l = loss_sum.div_(4)
        print(l)
        exit()
        return l'''

# not used
class QALossFunction():
    def __init__(self, cross_entropy: CrossEntropyLoss, numb_of_indices=4):
        self.cross_entropy = cross_entropy
        self.numb_of_indices = numb_of_indices
        self.target_value_e1_s = 1
        self.target_value_e1_e = 2
        self.target_value_e2_s = 3
        self.target_value_e2_e = 4

    def __call__(self, input: Tensor, target: Tensor):
        # note - input has the size of <batch size, numb of tokens (256?)>
        # target has size <batch size, numb of indices (4?)>
        #input_pred = input.argmax(dim=1)
        #print(input_pred.size())
        #print(input.size())
        #print(target.size())
        #exit()

        #real_target = 
        #torch.set_printoptions(threshold=10_000)
        #print(input)
        #print(target)
        #exit()
        

        real_target = []
        for batch_idx, batch_t in enumerate(target):
            real_target.append([0.0] * len(input[batch_idx]))
            #print(real_target)
            real_target[batch_idx][batch_t[0]] = self.target_value_e1_s
            real_target[batch_idx][batch_t[1]] = self.target_value_e1_e
            real_target[batch_idx][batch_t[2]] = self.target_value_e2_s
            real_target[batch_idx][batch_t[3]] = self.target_value_e2_e
            print(real_target[batch_idx][batch_t[0]], batch_idx, batch_t, len(input[batch_idx]))
            print(real_target[batch_idx][batch_t[1]])
            print(real_target[batch_idx][batch_t[2]])
            print(real_target[batch_idx][batch_t[3]])
        #real_target = torch.tensor(real_target, dtype=torch.double)

        torch.set_printoptions(threshold=10_000)
        print(real_target)
        exit()

        loss_sum = torch.zeros(1)
        for i in range(self.numb_of_indices):
            #inp = input[i, :].double()
            #tar = target[i].double()
            #print(inp.type())
            #print(tar.type())
            #print('inp', inp.size())
            #print('tar', tar.size())
            #loss_sum.add_(self.cross_entropy(inp, tar))
            loss_sum.add_(self.cross_entropy(input[i].double(), target[i].double()))
        print(loss_sum)
        exit()
        return loss_sum.div_(4)


