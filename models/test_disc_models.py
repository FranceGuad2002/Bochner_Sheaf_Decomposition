# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0
# Modified by the authors of the Bochner decomposition paper; see README.md for the changes.

import torch
import pytest
import numpy as np
import networkx as nx
import torch.nn.functional as F

from lib.perm_utils import generate_permutation_matrices, permute_graph
from models.disc_models import (DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion,
                                DiscreteGeneralSheafDiffusion, DiscreteIdentitySheafDiffusion)
from torch_geometric.utils import from_networkx
from utils.heterophilic import get_dataset


def get_test_config():
    return {
        'hidden_channels': 5,
        'device': torch.device('cpu'),
        'layers': 2,
        'normalised': False,
        'deg_normalised': False,
        'linear': False,
        'input_dropout': 0.5,
        'dropout': 0.5,
        'left_weights': True,
        'right_weights': True,
        'use_act': True,
        'second_linear': True,
        'add_lp': True,
        'add_hp': True,
        'max_t': 1.0,
        'sheaf_act': 'tanh',
        'tol_scale': 1.0,
        'int_method': 'euler',
        'step_size': 0.1,
        'max_iters': 100,
        'tol_scale_adjoint': 1.0,
        'adjoint_method': 'adaptive_heun',
        'adjoint_step_size': 0.1,
        'edge_weights': True,
        'orth': 'householder',
        'sparse_learner': False,
    }


@pytest.mark.parametrize("graph_id, d", [(256, 7), (70, 5), (379, 3), (1200, 5)])
def test_permutation_equivariance_of_diag_sheaf_diffusion_on_graph_atals(graph_id, d):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    graph = nx.graph_atlas(graph_id)
    data = from_networkx(graph)
    data.x = torch.FloatTensor(size=(len(graph), 13)).uniform_(-1.0, 1.0)

    args = get_test_config()
    args['graph_size'] = data.size(0)
    args['input_dim'] = 13
    args['output_dim'] = 3
    args['d'] = d
    args['normalised'] = True
    args['deg_normalised'] = False

    P = generate_permutation_matrices(size=len(graph), amount=1)[0]
    perm_data = permute_graph(data, P)

    # Construct the model
    with torch.no_grad():
        model = DiscreteDiagSheafDiffusion(data.edge_index, args)
        model.eval()
        out = model(data.x)
        out = torch.FloatTensor(P.astype(np.float64) @ out.numpy().astype(np.float64))

        model.update_edge_index(perm_data.edge_index)
        perm_out = model(perm_data.x)

    assert torch.allclose(out, perm_out, atol=1e-6)


@pytest.mark.parametrize("d, normalised", [(3, True), (5, False), (7, True)])
def test_permutation_equivariance_of_diag_sheaf_diffusion_on_texas(d, normalised):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    dataset = get_dataset('texas')
    data = dataset[0]

    perms = 3
    Ps = generate_permutation_matrices(size=data.x.size(0), amount=perms)

    args = get_test_config()
    args['graph_size'] = data.size(0)
    args['input_dim'] = dataset.num_features
    args['output_dim'] = dataset.num_classes
    args['d'] = d
    args['normalised'] = normalised
    args['deg_normalised'] = not normalised

    # Construct the model
    for i in range(perms):
        with torch.no_grad():
            model = DiscreteDiagSheafDiffusion(data.edge_index, args)
            model.eval()
            out = model(data.x)
            out = torch.FloatTensor(Ps[i] @ out.numpy())

            perm_data = permute_graph(data.clone(), Ps[i])
            model.update_edge_index(perm_data.edge_index)
            perm_out = model(perm_data.x)

        assert torch.allclose(out, perm_out, atol=1e-5)


@pytest.mark.parametrize("d, orth", [(2, "euler"), (3, "euler"), (5, "matrix_exp"), (6, "cayley"), (7, "householder")])
def test_permutation_equivariance_of_bundle_sheaf_diffusion_on_texas(d, orth):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    dataset = get_dataset('texas')
    data = dataset[0]

    perms = 3
    Ps = generate_permutation_matrices(size=data.x.size(0), amount=perms)

    args = get_test_config().copy()
    args['graph_size'] = data.size(0)
    args['input_dim'] = dataset.num_features
    args['output_dim'] = dataset.num_classes
    args['d'] = d
    args['normalised'] = True
    args['orth'] = orth
    args['edge_weights'] = True

    # Construct the model
    for i in range(perms):
        with torch.no_grad():
            model = DiscreteBundleSheafDiffusion(data.edge_index, args)
            model.eval()
            out = model(data.x)
            out = torch.FloatTensor(Ps[i] @ out.numpy())

            perm_data = permute_graph(data.clone(), Ps[i])
            model.update_edge_index(perm_data.edge_index)
            perm_out = model(perm_data.x)

        assert torch.allclose(out, perm_out, atol=1e-6)


@pytest.mark.parametrize("d, normalised", [(3, True), (5, True), (5, False),
                                           (7, True), (6, False), (4, True), (7, False)])
def test_permutation_equivariance_of_general_sheaf_diffusion_on_texas(d, normalised):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    dataset = get_dataset('texas')
    data = dataset[0]

    perms = 5
    Ps = generate_permutation_matrices(size=data.x.size(0), amount=perms)

    args = get_test_config().copy()
    args['graph_size'] = data.size(0)
    args['input_dim'] = dataset.num_features
    args['output_dim'] = dataset.num_classes
    args['d'] = d
    args['normalised'] = normalised
    args['deg_normalised'] = not normalised

    # Construct the model
    for i in range(perms):
        with torch.no_grad():
            model = DiscreteGeneralSheafDiffusion(data.edge_index, args)
            model.eval()
            assert not model.training
            assert not model.laplacian_builder.training
            out = model(data.x)
            out = torch.FloatTensor(Ps[i].astype(np.float64) @ out.numpy().astype(np.float64))

            perm_data = permute_graph(data.clone(), Ps[i])
            model.update_edge_index(perm_data.edge_index)
            perm_out = model(perm_data.x)

        assert torch.allclose(out, perm_out, atol=1e-4)


@pytest.mark.parametrize("d, orth", [(2, "euler"), (3, "euler"), (5, "matrix_exp"), (6, "cayley"), (7, "householder")])
def test_bundle_diffusion_backprop(d, orth):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)
    num_feat = 13

    # Build a test graph
    graph = nx.graph_atlas(546)
    data = from_networkx(graph)
    data.x = torch.FloatTensor(size=(len(graph), num_feat)).uniform_(-1.0, 1.0)
    data.y = torch.tensor([1]*len(graph), dtype=torch.long)

    # Construct the model
    args = get_test_config().copy()
    args['graph_size'] = data.size(0)
    args['input_dim'] = num_feat
    args['output_dim'] = 2
    args['d'] = d
    args['normalised'] = True
    args['orth'] = orth

    model = DiscreteBundleSheafDiffusion(data.edge_index, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

    optimizer.zero_grad()
    out = model(data.x)
    assert list(out.size()) == [len(graph), 2]
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()

    # Check that gradients are back-propagated through the Laplacian in particular
    for param in model.parameters():
        assert param.grad is not None


# ── Identity sheaf (conviction control) ───────────────────────────────────────

def _training_defaults(**overrides):
    """The defaults of exp/parser.py, i.e. exactly what a training run uses for any flag
    it does not pass (get_test_config above predates several required arguments)."""
    from exp.parser import get_parser
    args = vars(get_parser().parse_args([]))
    args.update({'device': torch.device('cpu'), 'normalised': True, 'deg_normalised': False})
    args.update(overrides)
    return args


def _identity_sheaf_setup(d, num_feat=13):
    """A small graph and the configuration of the paper's runs."""
    torch.random.manual_seed(0)
    np.random.seed(0)
    graph = nx.graph_atlas(546)
    data = from_networkx(graph)
    data.x = torch.FloatTensor(size=(len(graph), num_feat)).uniform_(-1.0, 1.0)
    data.y = torch.tensor([i % 2 for i in range(len(graph))], dtype=torch.long)

    args = _training_defaults(graph_size=data.size(0), input_dim=num_feat, output_dim=2, d=d)
    return data, args


def _dense(L, size):
    (idx, val) = L
    M = torch.zeros(size, size)
    M[idx[0], idx[1]] = val
    return M


@pytest.mark.parametrize("d", [1, 3])
def test_identity_sheaf_laplacian_is_normalised_graph_laplacian(d):
    data, args = _identity_sheaf_setup(d)
    n = data.size(0)
    model = DiscreteIdentitySheafDiffusion(data.edge_index, args)

    A = torch.zeros(n, n)
    A[data.edge_index[0], data.edge_index[1]] = 1.0
    deg = A.sum(1)
    Dt = torch.diag((deg + 1).pow(-0.5))
    L_ref = torch.kron(Dt @ (torch.diag(deg) - A) @ Dt, torch.eye(d))

    L = _dense(model.L, n * d)
    assert torch.allclose(L, L_ref, atol=1e-6)

    # The operator does not depend on the train/eval mode (no SVD perturbation baked in).
    model.train()
    model(data.x)
    assert torch.equal(_dense(model.L, n * d), L)
    model.eval()
    model(data.x)
    assert torch.equal(_dense(model.L, n * d), L)


@pytest.mark.parametrize("d", [1, 3])
def test_identity_sheaf_has_no_sheaf_parameters_and_trains(d):
    data, args = _identity_sheaf_setup(d)
    n = data.size(0)
    model = DiscreteIdentitySheafDiffusion(data.edge_index, args)
    L_before = _dense(model.L, n * d)

    sheaf_params, other_params = model.grouped_parameters()
    assert len(sheaf_params) == 0
    assert len(other_params) == len(list(model.parameters()))

    # Same optimizer construction as exp/run.py, with the (empty) sheaf group.
    optimizer = torch.optim.Adam([
        {'params': sheaf_params, 'weight_decay': 1e-3},
        {'params': other_params, 'weight_decay': 1e-3}], lr=0.01)
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model(data.x), data.y)
    loss.backward()
    optimizer.step()

    for param in model.parameters():
        assert param.grad is not None
    assert torch.equal(_dense(model.L, n * d), L_before)
    # The fixed Laplacian is not part of the checkpoint.
    assert not any(k.startswith('L') for k in model.state_dict().keys())


@pytest.mark.parametrize("d", [1, 3])
def test_identity_sheaf_node_norms_and_gate(d):
    data, args = _identity_sheaf_setup(d)
    n = data.size(0)
    model = DiscreteIdentitySheafDiffusion(data.edge_index, args)
    model.eval()

    with torch.no_grad():
        out = model(data.x)
        assert sorted(model._last_node_norms.keys()) == list(range(args['layers'] + 1))
        for norms in model._last_node_norms.values():
            assert norms.shape == (n,)

        ones = torch.ones(n)
        assert torch.allclose(model(data.x, gate=ones), out)

        gate = torch.ones(n)
        gate[:3] = 0.0
        assert not torch.allclose(model(data.x, gate=gate), out)


@pytest.mark.parametrize("d", [1, 3])
def test_permutation_equivariance_of_identity_sheaf_diffusion_on_texas(d):
    torch.random.manual_seed(0)
    np.random.seed(0)

    dataset = get_dataset('texas')
    data = dataset[0]

    perms = 3
    Ps = generate_permutation_matrices(size=data.x.size(0), amount=perms)

    args = _training_defaults(graph_size=data.size(0), input_dim=dataset.num_features,
                              output_dim=dataset.num_classes, d=d)

    for i in range(perms):
        with torch.no_grad():
            model = DiscreteIdentitySheafDiffusion(data.edge_index, args)
            model.eval()
            out = model(data.x)
            out = torch.FloatTensor(Ps[i].astype(np.float64) @ out.numpy().astype(np.float64))

            perm_data = permute_graph(data.clone(), Ps[i])
            model.update_edge_index(perm_data.edge_index)
            perm_out = model(perm_data.x)

        assert torch.allclose(out, perm_out, atol=1e-5)







@pytest.mark.parametrize("name", ['JointSheafParams', 'JointSheafParamsAlt', 'JointSheafVanilla'])
def test_joint_models_raise_a_clear_unavailable_error(name):
    from models.unavailable import JointModelUnavailable, unavailable_model
    with pytest.raises(JointModelUnavailable, match="not implemented in this public repository"):
        unavailable_model(name)(None, {})
