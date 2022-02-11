import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

from utils import uniform

class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout, a, b1, b2, n_layer=2, *args):
        super(RGCN, self).__init__()
        self.a = a
        #self.b1 = b1
        #self.b2 = b2
        self.n_layer = n_layer
        self.entity_embedding = nn.Embedding(num_entities, 64)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 64))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(
            64, 64, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            64, 64, num_relations * 2, num_bases=num_bases)
        self.conv3 = RGCNConv(
            64, 64, num_relations * 2, num_bases=num_bases)
        self.conv4 = RGCNConv(
            64, 64, num_relations * 2, num_bases=num_bases)

        self.dropout_ratio = dropout
        self.sig = torch.nn.Sigmoid()

    def forward(self, entity, edge_index, edge_type, edge_norm):
        if self.n_layer == 1:
            x = self.entity_embedding(entity)
            x = self.conv1(x, edge_index, edge_type, edge_norm)
        elif self.n_layer == 2:
            x = self.entity_embedding(entity)
            x = self.conv1(x, edge_index, edge_type, edge_norm)
            x = self.conv2(x, edge_index, edge_type, edge_norm)
        else:
            x = self.entity_embedding(entity)
            x = self.conv1(x, edge_index, edge_type, edge_norm)
            x = self.conv2(x, edge_index, edge_type, edge_norm)
            x = self.conv3(x, edge_index, edge_type, edge_norm)

        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)

        return score

    def fair_norm(self, embedding, idx):
        job_list_idx = []
        gender_idx = []
        for i in idx: #.cpu().detach().numpy():
            if int(i[1]) == 2:
                job_id = int(i[2]) #.cpu().detach().numpy()
                job_list_idx.append(job_id)
            if int(i[1]) == 148:
                gender_idx.append(int(i[2]))
        uniq_gender_idx = list(set(gender_idx))

        if len(uniq_gender_idx) != 2:
            return 0

        gen_rel = torch.tensor([148], dtype=torch.long).contiguous()
        job_rel = torch.tensor([2], dtype=torch.long).contiguous()
        job2tensor = torch.tensor(job_list_idx, dtype=torch.long).contiguous()
        gen2tensor = torch.tensor(uniq_gender_idx, dtype=torch.long).contiguous()

        gen_rel_emb = self.relation_embedding[gen_rel]
        job_rel_emb = self.relation_embedding[job_rel]
        gen_emb0 = embedding[gen2tensor][0].view(len(embedding[gen2tensor][0]), 1)
        gen_emb1 = embedding[gen2tensor][1].view(len(embedding[gen2tensor][1]), 1)

        diff_gen = gen_emb0 - gen_emb1
        diff_gen = torch.norm(diff_gen, 2)

        job_emb = embedding[job2tensor]
        fair_sum0 = torch.sum(gen_emb0.t() * job_emb * job_rel_emb * gen_rel_emb)
        fair_sum1 = torch.sum(gen_emb1.t() * job_emb * job_rel_emb * gen_rel_emb)
        diff_fair = fair_sum0 - fair_sum1
        diff_fair = torch.norm(diff_fair, 2)

        return self.a*(self.sig((diff_fair))) / (self.sig(diff_gen))

    def score_loss(self, embedding, idx, triplets, target):
        score = self.distmult(embedding, triplets)
        fair_sum = self.fair_norm(embedding, idx)
        return F.binary_cross_entropy_with_logits(score, target) + fair_sum

    def score_loss_fair(self, embedding, idx, triplets, target):
        fair_sum = self.fair_norm(embedding, idx)
        b2 = self.b2
        return (1-b2)*fair_sum

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))

class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)


    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)


    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
