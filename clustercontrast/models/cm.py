# from _typeshed import OpenTextModeUpdating
import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch import einsum


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets):

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss
    
class CM_Camera(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, cams, features, pids, camids, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, cams, pids, camids)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, cams, pids, camids = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y, z in zip(inputs, targets, cams):
            index_m = (pids == y) & (camids == z)
            ctx.features[index_m] = ctx.momentum * ctx.features[index_m] + (1. - ctx.momentum) * x
            ctx.features[index_m] /= ctx.features[index_m].norm()

        return grad_inputs, None, None, None, None, None, None


def cm_camera(inputs, targets, cams, features, pids, camids, momentum=0.5):
    return CM_Camera.apply(inputs, targets, cams, features, pids, camids, torch.Tensor([momentum]).to(inputs.device))

class CameraMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, pids, camids, temp=0.05, momentum=0.2, margin=0.0):
        super(CameraMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        # self.pids = torch.LongTensor(pids)
        # self.camids = torch.LongTensor(camids)
        self.margin = margin

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('pids', torch.LongTensor(pids))
        self.register_buffer('camids', torch.LongTensor(camids))
        
    def forward(self, inputs, targets, cams):

        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = cm_camera(inputs, targets, cams, self.features, 
                            self.pids, self.camids, self.momentum)
        outputs = (outputs + 1.0) * 0.5 
        loss_p = 0.0
        loss_n = 0.0
        b = inputs.shape[0]
        for idx in range(b):
            pos = outputs[idx][self.pids == targets[idx]]
            index_neg = ~(self.pids == targets[idx]) & (self.camids == cams[idx]) 
            neg = outputs[idx][index_neg]
            if len(neg) == 0:
                neg = torch.sort(outputs[idx][~(self.pids == targets[idx])], dim=1)[:len(pos)]
            alpha_p = torch.relu(-pos.detach() + 1)
            alpha_n = torch.relu(neg.detach())
            loss_p += torch.sum(torch.exp(-alpha_p*(pos-1)/self.temp))
            loss_n += torch.sum(torch.exp(alpha_n*neg/self.temp))
        loss = torch.log(1+loss_p*loss_n)
        return loss
# class CameraMemory(nn.Module, ABC):
#     def __init__(self, num_features, num_samples, pids, camids, temp=0.05, momentum=0.2, margin=0.0):
#         super(CameraMemory, self).__init__()
#         self.num_features = num_features
#         self.num_samples = num_samples

#         self.momentum = momentum
#         self.temp = temp
#         # self.pids = torch.LongTensor(pids)
#         # self.camids = torch.LongTensor(camids)
#         self.margin = margin

#         self.register_buffer('features', torch.zeros(num_samples, num_features))
#         self.register_buffer('pids', torch.LongTensor(pids))
#         self.register_buffer('camids', torch.LongTensor(camids))
        
#     def forward(self, inputs, targets, cams):

#         inputs = F.normalize(inputs, dim=1).cuda()
#         outputs = cm_camera(inputs, targets, cams, self.features, 
#                             self.pids, self.camids, self.momentum)
#         outputs = outputs / self.temp
#         loss = 0.0
#         b = inputs.shape[0]
#         for idx in range(b):
#             pos = outputs[idx][self.pids == targets[idx]]
#             index_neg = (~(self.pids == targets[idx])) & (self.camids == cams[idx])
#             neg = outputs[idx][index_neg]
#             tmp = torch.cat([pos, neg])
#             tmp = F.softmax(tmp,dim=0)[:len(pos)]
#             loss += (-torch.mean(torch.log(tmp)))
#         loss /= b
#         return loss

