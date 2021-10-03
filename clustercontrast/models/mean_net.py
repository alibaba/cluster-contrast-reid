import copy
import torch
from torch import nn

from torch.nn import init
import torch.nn.functional as F
from torch import einsum

class Embedding(nn.Module):
    def __init__(self, planes, embed_feat=0, dropout=0.0):
        super(Embedding, self).__init__()

        self.has_embedding = embed_feat > 0
        self.dropout = dropout

        if self.has_embedding:
            self.feat_reduction = nn.Linear(planes, embed_feat)
            init.kaiming_normal_(self.feat_reduction.weight, mode="fan_out")
            init.constant_(self.feat_reduction.bias, 0)
            planes = embed_feat

        self.feat_bn = nn.BatchNorm1d(planes)
        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.num_features = planes
        self.dropout = dropout

    def forward(self, x):

        if self.has_embedding:
            feat = self.feat_reduction(x)
            N, L = feat.size()
            # (N,L)->(N,C,L) to fit sync BN
            feat = self.feat_bn(feat.view(N, L, 1)).view(N, L)
            if self.training:
                feat = nn.functional.relu(feat)
        else:
            N, L = x.size()
            # (N,L)->(N,C,L) to fit sync BN
            feat = self.feat_bn(x.view(N, L, 1)).view(N, L)

        if self.dropout > 0:
            feat = nn.functional.dropout(feat, p=self.dropout, training=self.training)

        return feat

class TeacherStudentNetwork(nn.Module):
    """
    TeacherStudentNetwork.
    """

    def __init__(
        self, net=None, alpha=0.999, in_dim=2048, mid_dim=512
    ):
        super(TeacherStudentNetwork, self).__init__()
        self.net = net
        self.mean_net = copy.deepcopy(self.net)
        self.net_params = []
        self.mean_net_params = []
        self.embedding = nn.Sequential(
            Embedding(in_dim, mid_dim),
            Embedding(mid_dim, in_dim)
        )

        for param, param_m in zip(self.net.parameters(), self.mean_net.parameters()):
            self.net_params.append(param)
            self.mean_net_params.append(param_m)
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
            

        self.alpha = alpha

    def forward(self, x, mode=0):
        if mode == 0:
            return self.net(x)
        else:
            feat = self.net(x[0])
            feat = self.embedding(feat)
            feat = F.normalize(feat, dim=1)
            with torch.no_grad():
                self._update_mean_net()
                mean_feat = self.mean_net(x[1])
                
            mean_feat = F.normalize(mean_feat, dim=1)
            return feat, mean_feat

    @torch.no_grad()
    def _update_mean_net(self):
        for param, param_m in zip(self.net_params, self.mean_net_params):
            param_m.data.mul_(self.alpha).add_(param.data, alpha=1-self.alpha)

    @torch.no_grad()
    def _init_embedding(self):
        for child in self.embedding.children():
            init.kaiming_normal_(child.feat_reduction.weight, mode="fan_out")
            init.constant_(child.feat_reduction.bias, 0)
            init.constant_(child.feat_bn.weight, 1)
            init.constant_(child.feat_bn.bias, 0)
            
            
class SelfDisLoss(nn.Module):
    def __init__(self):
        super(SelfDisLoss, self).__init__()
    
    def forward(self,feat, mean_feat):
        
        sim = einsum("nc,nc->n", [feat, mean_feat])
        dis = torch.sqrt(2.0*(1-sim))
        loss = torch.mean(dis)
        return loss