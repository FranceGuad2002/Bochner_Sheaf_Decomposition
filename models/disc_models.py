# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0
# Modified by the authors of the Bochner decomposition paper; see README.md for the changes.

import torch
import torch.nn.functional as F
import torch_sparse

#import Tuple

from torch import nn
from models.sheaf_base import SheafDiffusion
from models import laplacian_builders as lb
from lib import laplace as lap
from models.sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner, DiagEdgeWeightLearner, LocalNodeSheafLearner, LocalConcatSheafLearnerVariant
from models.orthogonal import Orthogonal


def _apply_gate(L, gate, final_d):
    """Scale cross-node entries of the sparse diffusion Laplacian L by the
    source node's outgoing-message gate; self-block entries (row and col
    belong to the same node) are left at 1.0, so a gated node's own
    self-retention is untouched — only what it sends to *other* nodes is
    suppressed/attenuated. No-op (returns L unchanged) if gate is None, so
    every non-gated forward call is completely unaffected."""
    if gate is None:
        return L
    row, col = L[0]
    node_row, node_col = row // final_d, col // final_d
    is_self_block = node_row == node_col
    entry_gate = torch.where(is_self_block, torch.ones_like(L[1]), gate[node_col])
    return L[0], L[1] * entry_gate


class DiscreteDiagSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteDiagSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        """Abbiamo un MLP per ogni layer, ma se non usiamo la non linearità allora ne usiamo solo uno e lo condividiamo tra i layer.
        Questo è perché se non usiamo la non linearità allora il modello è equivalente a un modello con un solo layer di apprendimento della sheaf, quindi non ha senso avere più di un MLP."""

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)

        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))

        
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                         normalised=self.normalised,
                                                         deg_normalised=self.deg_normalised,
                                                         add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        # self._last_maps = None
        # self._last_trans_maps = None
        # self._last_laplacian = None

    def forward(self, x, gate=None):

        #print(f"Input x size: {x.detach().cpu().numpy().shape}")

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)
        #print(f"After initial linear layers x size: {x.detach().cpu().numpy().shape}")

        x0 = x
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_node_norms = {}

        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)

                # print maps to see them
                #print(f"Layer {layer} maps: {maps.detach().cpu().numpy()}, maps length: {maps.size(0)}, edge_index: {self.edge_index}, len edge_index: {self.edge_index.size(1)}")
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

                self._last_maps[layer] = maps
                self._last_trans_maps[layer] = trans_maps
                self._last_laplacian[layer] = L

                # print(f"Layer {layer} maps: {self._last_maps.detach().cpu().numpy()}")
                # print(f"Layer {layer} Laplacian: {self._last_laplacian[0].detach().cpu().numpy()}, {self._last_laplacian[1].detach().cpu().numpy()}")

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            L_gated = _apply_gate(L, gate, self.final_d)
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        # save the norms of node features
        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class DiscreteBundleSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteBundleSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            
            if self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans, normalised=self.normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x, gate=None):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_node_norms = {}

        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, self.edge_index)
                edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)
                self._last_maps[layer] = maps
                self._last_trans_maps[layer] = trans_maps
                self._last_laplacian[layer] = L


            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            L_gated = _apply_gate(L, gate, self.final_d)
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0
        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class DiscreteBochnerSheafDiffusion(SheafDiffusion):
    """Bundle sheaf diffusion whose restriction maps are F_{v<=e} = Sigma_e @ O_{v<=e}: an
    orthogonal map individual to each direction, scaled by a diagonal Sigma_e shared symmetrically
    between both endpoints of edge e. Reduces to BundleSheaf when Sigma_e = I.

    Because F_{v<=e}^T F_{v<=e} = O_{v<=e}^T Sigma_e^2 O_{v<=e} is a genuine d x d matrix (not a
    multiple of the identity like in the scalar-weighted Bundle case), the Laplacian diagonal
    blocks can't use NormConnectionLaplacianBuilder's scalar-degree shortcut and are instead built
    with GeneralLaplacianBuilder, which accumulates them densely.
    """

    def __init__(self, edge_index, args):
        super(DiscreteBochnerSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.sigma_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            self.sigma_learners.append(DiagEdgeWeightLearner(self.hidden_dim, self.d, scale=1.0))

        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=self.orth_trans)
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x, gate=None):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_node_norms = {}

        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                map_params = self.sheaf_learners[layer](x_maps, self.edge_index)
                orth_maps = self.orth_transform(map_params)
                sigma = self.sigma_learners[layer](x_maps, self.edge_index)
                maps = orth_maps * sigma.unsqueeze(-1)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)
                self._last_maps[layer] = maps
                self._last_trans_maps[layer] = trans_maps
                self._last_laplacian[layer] = L

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            L_gated = _apply_gate(L, gate, self.final_d)
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD (used by GeneralLaplacianBuilder's normalisation).
        assert torch.all(torch.isfinite(x))
        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class DiscreteBochnerFlatSheafDiffusion(SheafDiffusion):
    """Flat variant of BochnerSheaf: each vertex v has a single orthogonal map O_v shared by all
    its incident edges (O_{v<=e} = O_v for every e touching v), so the restriction map for edge e
    from v's side is F_{v<=e} = Sigma_e @ O_v. Since O_v doesn't depend on the edge, the diagonal
    block collapses to O_v^T @ diag(sum_e Sigma_e^2) @ O_v -- a free spectral decomposition -- and
    normalisation reduces to pure per-channel vector arithmetic (see FlatBochnerLaplacianBuilder),
    with no SVD/Cholesky/matrix inversion anywhere. Equivalent to DiscreteDiagSheafDiffusion after
    a per-node change of basis by O_v -- a curvature-zero ("flat") connection.
    """

    def __init__(self, edge_index, args):
        super(DiscreteBochnerFlatSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.sigma_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            self.sheaf_learners.append(LocalNodeSheafLearner(
                self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            self.sigma_learners.append(DiagEdgeWeightLearner(self.hidden_dim, self.d, scale=1.0))

        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=self.orth_trans)
        self.laplacian_builder = lb.FlatBochnerLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x, gate=None):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_node_norms = {}

        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                map_params = self.sheaf_learners[layer](x_maps)
                O = self.orth_transform(map_params)
                sigma = self.sigma_learners[layer](x_maps, self.edge_index)
                L, trans_maps = self.laplacian_builder(O, sigma)
                self.sheaf_learners[layer].set_L(trans_maps)
                self._last_maps[layer] = O
                self._last_trans_maps[layer] = trans_maps
                self._last_laplacian[layer] = L

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            L_gated = _apply_gate(L, gate, self.final_d)
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class DiscreteGeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteGeneralSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)
        self.laplacian_builder_other = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=not self.normalised, deg_normalised=False)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x, gate=None):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_laplacian_other = {}
        self._last_node_norms = {}

        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

                self._last_maps[layer] = maps
                self._last_trans_maps[layer] = trans_maps
                self._last_laplacian[layer] = L

                if getattr(self, 'save_other_laplacian', False):
                    with torch.no_grad():
                        L_other, _ = self.laplacian_builder_other(maps)
                    self._last_laplacian_other[layer] = L_other

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            L_gated = _apply_gate(L, gate, self.final_d)
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))
        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class DiscreteIdentitySheafDiffusion(SheafDiffusion):
    """Control for the conviction analysis: DiscreteGeneralSheafDiffusion with every
    restriction map fixed to the d x d identity instead of learnt. Everything else
    (encoder, left/right weights, epsilon residual update, dropout, readout, message
    gate, saved node norms) is identical. At d=1 the operator is the GCN-normalised
    graph Laplacian D~^{-1/2}(D - A)D~^{-1/2}, with D~ = D + I."""

    def __init__(self, edge_index, args):
        super(DiscreteIdentitySheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] >= 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        # Nothing is learnt: no sheaf parameters, so the sheaf parameter group is empty.
        self.sheaf_learners = nn.ModuleList()
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)
        # A plain attribute, not a registered buffer: the checkpoint then holds only the
        # learnt weights, so the pruning intervention can reload them into a model
        # built on a smaller graph.
        self.L = self._identity_laplacian()

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def _identity_laplacian(self):
        # Eval mode: in training mode GeneralLaplacianBuilder perturbs the diagonal by
        # U(-1e-3, 1e-3), which only matters for SVD gradients through learnt maps.
        # Nothing is differentiated here, so build the exact operator.
        self.laplacian_builder.eval()
        maps = torch.eye(self.d, device=self.edge_index.device).unsqueeze(0).repeat(
            self.edge_index.size(1), 1, 1)
        with torch.no_grad():
            L, _ = self.laplacian_builder(maps)
        return L

    def update_edge_index(self, edge_index):
        super(DiscreteIdentitySheafDiffusion, self).update_edge_index(edge_index)
        self.L = self._identity_laplacian()

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x, gate=None):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}
        self._last_laplacian_other = {}
        self._last_node_norms = {}

        for layer in range(self.layers):
            self._last_node_norms[layer] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
            self._last_laplacian[layer] = self.L

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            L_gated = _apply_gate(self.L, gate, self.final_d)
            x = torch_sparse.spmm(L_gated[0], L_gated[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        assert torch.all(torch.isfinite(x))
        self._last_node_norms[self.layers] = x.detach().reshape(self.graph_size, -1).norm(dim=1).cpu()
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
