import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch


def generate_grad(grad, adv_grad, layer_mask=None, lr=0.01):
    ret_grad = OrderedDict()
    for k,v in grad.items():
        if k in layer_mask:
            g1 = grad[k]
            g2 = adv_grad[k]

            diff = g1 - g2
            pos_mask = diff > 0
            pos_mask = pos_mask.int()
            neg_mask = diff < 0
            neg_mask = neg_mask.int()
            zero_mask = diff == 0
            zero_mask = zero_mask.int()

            sign_diff = g1.sign() + g2.sign()  # 0 represnets opposite search direction. 
            sign_diff[sign_diff>0] = 1
            sign_diff[sign_diff<0] = 1
            sign_diff[sign_diff == 0] = lr
            
            g1_part = pos_mask * g2 * sign_diff * (1 + lr)
            # g1_part = pos_mask * (g2 + diff * ratio)
            g2_part = neg_mask * g2 * sign_diff
            same_part = zero_mask * g2 

            ret_grad[k] = g1_part + g2_part + same_part
    
        else:
            ret_grad[k] = g2

    return ret_grad


def get_clear_grad(model):
    grad = OrderedDict()
    for t1,t2 in zip(model.parameters(), model.state_dict().items()):
        try:
            grad[t2[0]] = t1.grad.data.clone()
        except:
            # print(t2[0])
            grad[t2[0]] = None
    return grad

     
        
def get_grad(model, y, y_true, optimizer, criterion):
    loss = criterion(y, y_true)
    optimizer.zero_grad()
    loss.backward()
    grad = OrderedDict()
    for t1,t2 in zip(model.parameters(), model.state_dict().items()):
        grad[t2[0]] = t1.grad.data.clone()
    return grad


def set_grad(model, grad):
    for t1,t2 in zip(model.parameters(), model.state_dict().items()):
        if 'weight' in t2[0] and t2[0] in grad:
            t1.grad = grad[t2[0]]
        else:
            pass
            # print(t2[0])
            # print('something wrong.')

def get_grad_diff_layer_mask(grad, adv_grad, ratio=0.1):
    layer_mask = OrderedDict()
    avg_list = []

    def cal_mean_diff(g1, g2):
        diff = g1 - g2
        normalized_diff = (diff - torch.min(diff)) / (torch.max(diff) - torch.min(diff))
        return torch.mean(normalized_diff)

    for k,v in grad.items():
        if 'weight' not in k:
            continue
        layer_mask[k] = 0
        g1 = grad[k]
        g2 = adv_grad[k]
        avg_g = cal_mean_diff(g1, g2)
        avg_list.append(avg_g)
    
    # torch.kthvalue from smallest to largest.
    avg_list = torch.tensor(avg_list)
    threshold = torch.kthvalue(avg_list, int(avg_list.size(0) * (1 - ratio))).values
    for k,v in layer_mask.items():
        if v >= threshold:
            layer_mask = 1

    return layer_mask