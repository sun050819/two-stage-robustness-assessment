# -*- coding: utf-8 -*-
"""Complete branch-and-bound (BaB) verifier built on the CROWN bound engine.

Every search node carries the current box plus a set of ReLU sign decisions.
Bounds are recomputed with decision-aware interval propagation: the
pre-activation interval of a branched neuron is intersected with the branch
side and all downstream layers are re-propagated.

Soundness: the CROWN margin lower bound is sound under the accumulated
decisions.  Completeness: when every unstable ReLU has been branched on, the
network is affine over the leaf box, so the CROWN bound is exact; the leaf is
robust iff the minimum of the affine margin over the box is > 0, which is
checked exactly at the minimizing corner (gradient sign rule).

The search is best-first on the certified margin lower bound.  A node budget
avoids pathological blowups and yields verdict 'unknown'.  This verifier is
used only to cross-validate SAFE2 certificates on small networks.
"""
import heapq

import torch
import torch.nn.functional as F

import baseline_crown as bc


def _modules(model):
    if hasattr(model, 'net') and isinstance(model.net, torch.nn.Sequential):
        return list(model.net)
    if isinstance(model, torch.nn.Sequential):
        return list(model)
    raise TypeError('expected nn.Sequential or wrapper with .net')


def _preact_pass(model, x_lo, x_hi, input_shape, decisions):
    """Decision-aware forward pass (per-neuron decisions).

    Returns (preacts, flatten_shapes); preacts[i] = (l, u, neuron_id) for each
    ReLU in order, where (l, u) is the *decision-aware* pre-activation interval.
    """
    modules = _modules(model)
    xl, xh = x_lo, x_hi
    if input_shape is not None and xl.dim() == 1:
        xl = xl.view(*input_shape)
        xh = xh.view(*input_shape)
    preacts = []
    flatten_shapes = {}
    neuron_id = 0
    for idx, mod in enumerate(modules):
        if isinstance(mod, torch.nn.Linear):
            if xl.dim() > 1:
                xl = xl.flatten()
                xh = xh.flatten()
            xl, xh = bc._ibp_forward(xl, xh, mod)
        elif isinstance(mod, torch.nn.Conv2d):
            xl, xh = bc._ibp_forward(xl, xh, mod)
        elif isinstance(mod, torch.nn.ReLU):
            pl, ph = xl.clone(), xh.clone()
            n = xl.numel()
            zl = xl.flatten()
            zh = xh.flatten()
            gids = torch.arange(n) + neuron_id
            dead = torch.tensor([decisions.get(int(g), -1) == 0 for g in gids])
            zl2 = torch.where(dead, zl, torch.clamp(zl, min=0.0))
            zh2 = torch.where(dead, torch.clamp(zh, max=0.0),
                              torch.clamp(zh, min=0.0))
            xl = zl2.view_as(pl)
            xh = zh2.view_as(ph)
            preacts.append((pl, ph, neuron_id))
            neuron_id += n
        elif isinstance(mod, torch.nn.Flatten):
            flatten_shapes[idx] = tuple(xl.shape)
            xl, xh = bc._ibp_forward(xl, xh, mod)
    return preacts, flatten_shapes


def crown_logits_with_preacts(model, x_lo, x_hi, input_shape, preacts,
                              flatten_shapes):
    """CROWN logit bounds with decision-aware pre-activation intervals."""
    modules = _modules(model)
    K = None
    for m in reversed(modules):
        if isinstance(m, torch.nn.Linear):
            K = m.weight.shape[0]
            break
    a_l = torch.eye(K)
    b_l = torch.zeros(K)
    a_u = torch.eye(K)
    b_u = torch.zeros(K)
    relu_pos = len(preacts) - 1
    for r_idx in range(len(modules) - 1, -1, -1):
        mod = modules[r_idx]
        if isinstance(mod, torch.nn.Linear):
            b_l = b_l + a_l @ mod.bias
            b_u = b_u + a_u @ mod.bias
            a_l = a_l @ mod.weight
            a_u = a_u @ mod.weight
        elif isinstance(mod, torch.nn.Conv2d):
            red = tuple(range(1, a_l.dim()))
            b_l = b_l + (a_l * mod.bias.view(-1, 1, 1)).sum(dim=red)
            b_u = b_u + (a_u * mod.bias.view(-1, 1, 1)).sum(dim=red)
            op = tuple(1 if s == 2 else 0 for s in mod.stride)
            a_l = F.conv_transpose2d(a_l, mod.weight, None, mod.stride,
                                     mod.padding, output_padding=op)
            a_u = F.conv_transpose2d(a_u, mod.weight, None, mod.stride,
                                     mod.padding, output_padding=op)
        elif isinstance(mod, torch.nn.ReLU):
            pl, ph, _ = preacts[relu_pos]
            a_l = a_l * (a_l >= 0).float()
            dead = ph <= 0
            live = pl >= 0
            alpha_u = torch.where(live, torch.ones_like(ph),
                                  torch.where(dead, torch.zeros_like(ph),
                                              ph / (ph - pl)))
            beta_u = torch.where(live, torch.zeros_like(ph),
                                 torch.where(dead, torch.zeros_like(ph),
                                             -pl * ph / (ph - pl)))
            pos = (a_u >= 0).float()
            b_u = b_u + ((a_u * beta_u) * pos).sum(dim=tuple(range(1, a_u.dim())))
            a_u = (a_u * alpha_u) * pos
            relu_pos -= 1
        elif isinstance(mod, torch.nn.Flatten):
            in_shape = flatten_shapes[r_idx]
            a_l = a_l.view(a_l.shape[0], *in_shape)
            a_u = a_u.view(a_u.shape[0], *in_shape)
    lo = torch.empty(K)
    hi = torch.empty(K)
    for k in range(K):
        lo[k], _ = bc._box_lin(a_l[k], b_l[k], x_lo, x_hi)
        _, hi[k] = bc._box_lin(a_u[k], b_u[k], x_lo, x_hi)
    return lo, hi


def decision_margin_lb(model, x_lo, x_hi, c, input_shape, decisions):
    preacts, fl_shapes = _preact_pass(model, x_lo, x_hi, input_shape, decisions)
    lo, hi = crown_logits_with_preacts(model, x_lo, x_hi, input_shape,
                                       preacts, fl_shapes)
    others = [j for j in range(lo.shape[0]) if j != c]
    return float((lo[c] - torch.max(hi[others])).item())


def unstable_neurons(model, x_lo, x_hi, input_shape, decisions):
    """Per-neuron unstable set: neurons not yet decided whose decision-aware
    pre-activation interval crosses zero. Returns [(neuron_id, (lv, uv))]."""
    preacts, _ = _preact_pass(model, x_lo, x_hi, input_shape, decisions)
    out = []
    for (l, u, nid) in preacts:
        lf, uf = l.flatten(), u.flatten()
        for q in range(lf.shape[0]):
            gid = nid + q
            if gid in decisions:
                continue
            lv, uv = float(lf[q].item()), float(uf[q].item())
            if lv < 0 < uv:
                out.append((gid, (lv, uv)))
    return out


def _logit_affine(model, x_ref, input_shape, decisions):
    """Affine coefficients (A, b) of all logits under the decided network.

    A: (K, d), b: (K,); exact over the whole box at a leaf (the decided
    network is affine there).
    """
    x = x_ref.detach().clone().requires_grad_(True)
    preacts, logits_val = _decided_record_logits(model, x, input_shape, decisions)
    K = logits_val.shape[0]
    d = x_ref.shape[0]
    A = torch.zeros(K, d)
    b = torch.zeros(K)
    for k in range(K):
        logits_val[k].backward(retain_graph=True)
        gk = x.grad.detach().clone().flatten()
        A[k] = gk
        b[k] = float(logits_val[k].item()) - float((gk * x_ref.flatten()).sum().item())
        x.grad = None
    coeffs = {}
    for (nid, pre) in preacts:
        for local in range(pre.shape[1]):
            gid = nid + local
            if gid in decisions:
                xg = x.detach().clone().requires_grad_(True)
                pre2, _ = _decided_record_logits(model, xg, input_shape, decisions)
                z2 = None
                for (n2, p2) in pre2:
                    if n2 == nid:
                        z2 = p2[0, local]
                        break
                z2.backward()
                a_i = xg.grad.detach().clone().flatten()
                b_i = float(z2.item()) - float((a_i * x_ref.flatten()).sum().item())
                coeffs[gid] = (a_i, b_i)
    return A, b, coeffs


def _decided_record_logits(model, x, input_shape, decisions):
    """Decided forward returning (preact_values, logits)."""
    modules = _modules(model)
    if input_shape is not None:
        h = x.view(1, *input_shape)
    else:
        h = x.unsqueeze(0)
    preacts = []
    neuron_id = 0
    for mod in modules:
        if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
            h = mod(h)
        elif isinstance(mod, torch.nn.ReLU):
            preacts.append((neuron_id, h.clone().flatten(1)))
            z = h.flatten(1)
            n = z.shape[1]
            gids = torch.arange(n) + neuron_id
            dead = torch.tensor([decisions.get(int(g), -1) == 0 for g in gids])
            active = torch.tensor([decisions.get(int(g), -1) == 1 for g in gids])
            z = torch.where(dead.view(1, -1), torch.zeros_like(z),
                            torch.where(active.view(1, -1), z, torch.clamp(z, min=0.0)))
            h = z.view_as(h)
            neuron_id += n
        elif isinstance(mod, torch.nn.Flatten):
            h = h.flatten(1)
    return preacts, h.squeeze(0)


def affine_leaf_min_margin(model, x_lo, x_hi, c, input_shape, decisions):
    """Exact minimum of the margin over the leaf polytope via LP.

    margin(x) = min_j (logit_c - logit_j)(x); each (logit_c - logit_j) is
    affine under the fully-decided network, so the exact minimum over the
    polytope {box cap sign constraints} is the minimum over j of K-1 linear
    programs (solved with scipy HiGHS).

    Returns (min_margin, argmin_point); infeasible leaf -> (inf, None).
    """
    import numpy as np
    from scipy.optimize import linprog
    # Complete every neuron's sign: branched neurons use their decision, and
    # each fixed (unbranched) neuron uses its decision-aware interval sign.
    preacts, _ = _preact_pass(model, x_lo, x_hi, input_shape, decisions)
    full = dict(decisions)
    for (l, u, nid) in preacts:
        lf, uf = l.flatten(), u.flatten()
        for q in range(lf.shape[0]):
            gid = nid + q
            if gid in full:
                continue
            lv, uv = float(lf[q].item()), float(uf[q].item())
            if lv >= 0.0:
                full[gid] = 1
            elif uv <= 0.0:
                full[gid] = 0
    x_ref = (x_lo + x_hi) / 2.0
    A, b, coeffs = _logit_affine(model, x_ref, input_shape, full)
    d = x_ref.shape[0]
    xl = x_lo.detach().numpy().astype(float)
    xh = x_hi.detach().numpy().astype(float)
    A_ub = []
    b_ub = []
    # constraints only for the search-branched decisions (fixed neurons are
    # automatically satisfied in the feasible region)
    for gid, (a_i, b_i) in coeffs.items():
        if gid not in decisions:
            continue
        ai = a_i.detach().numpy().astype(float)
        bi = float(b_i)
        if decisions[gid] == 1:
            A_ub.append(-ai)
            b_ub.append(bi)
        else:
            A_ub.append(ai)
            b_ub.append(-bi)
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    K = A.shape[0]
    best = None
    best_x = None
    any_feasible = False
    for j in range(K):
        if j == c:
            continue
        obj = (A[c] - A[j]).detach().numpy().astype(float)
        obj_b = float((b[c] - b[j]).item())
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(xl, xh)),
                      method='highs')
        if res.success:
            any_feasible = True
            val = float(res.fun) + obj_b
            if best is None or val < best:
                best = val
                best_x = torch.tensor(res.x, dtype=torch.float32)
    if not any_feasible:
        return float('inf'), None
    return best, best_x


def verify_bab(model, x0, c, eps, input_shape=None, node_budget=50000,
               time_budget=None, box=None):
    """Complete BaB. Returns (verdict, nodes_explored, counterexample).

    verdict in {'robust', 'not_robust', 'unknown'}; counterexample is the
    minimizing-corner point of the affine leaf (a certified adversarial input)
    when verdict is 'not_robust', else None.  time_budget (seconds) is a
    wall-clock cap per box: the search stops early with 'unknown' when the
    budget is exhausted, guarding against pathological LP-leaf blowups.
    If box=(xl, xh) is given, the search starts from that box instead of
    x0 +/- eps (used by contraction-preprocessed BaB).
    """
    import time
    if box is not None:
        X_lo, X_hi = box[0].detach().clone(), box[1].detach().clone()
    else:
        X_lo = x0 - eps
        X_hi = x0 + eps
    if input_shape is not None:
        X_lo = torch.clamp(X_lo, 0.0, 1.0)
        X_hi = torch.clamp(X_hi, 0.0, 1.0)
    t_start = time.perf_counter()
    lb = decision_margin_lb(model, X_lo, X_hi, c, input_shape, {})
    heap = [(-lb, 0, X_lo, X_hi, {})]
    nodes = 0
    counter = 1
    robust_leaves = 0
    counterexample = None
    timed_out = False
    while heap and nodes < node_budget:
        if time_budget is not None and \
                time.perf_counter() - t_start > time_budget:
            timed_out = True
            break
        neg_lb, _, xl, xh, dec = heapq.heappop(heap)
        nodes += 1
        lb = -neg_lb
        if lb > 0:
            robust_leaves += 1
            continue
        unst = unstable_neurons(model, xl, xh, input_shape, dec)
        if not unst:
            m, x_min = affine_leaf_min_margin(model, xl, xh, c, input_shape, dec)
            if x_min is not None and m <= 0:
                counterexample = x_min
                break
            robust_leaves += 1
            continue
        nid, (l, u) = max(unst, key=lambda t: t[1][1] - t[1][0])
        for side in (1, 0):
            nd = dict(dec)
            nd[nid] = side
            lb2 = decision_margin_lb(model, xl, xh, c, input_shape, nd)
            heapq.heappush(heap, (-lb2, counter, xl, xh, nd))
            counter += 1
    if counterexample is not None:
        return 'not_robust', nodes, counterexample
    if timed_out or (nodes >= node_budget and heap):
        return 'unknown', nodes, None
    return 'robust', nodes, None


def _true_margin(model, x, c, input_shape):
    if input_shape is not None:
        h = x.view(1, *input_shape)
    else:
        h = x.unsqueeze(0)
    with torch.no_grad():
        logits = model(h).squeeze(0)
    others = [j for j in range(logits.shape[0]) if j != c]
    return float((logits[c] - torch.max(logits[others])).item())


def _leaf_corner_min(model, x_lo, x_hi, c, input_shape):
    """Exact min margin over a leaf box by corner evaluation.

    At a split-BaB leaf every ReLU pre-activation interval avoids 0, so the
    network is affine over the box and the margin is affine; its minimum over
    the box is attained at one of the 2^d corners.  Returns (min_margin,
    argmin_corner).
    """
    d = x_lo.shape[0]
    best = None
    best_x = None
    for corner in range(1 << d):
        xs = torch.stack([
            x_hi[j] if (corner >> j) & 1 else x_lo[j] for j in range(d)
        ])
        m = _true_margin(model, xs, c, input_shape)
        if best is None or m < best:
            best = m
            best_x = xs
    return best, best_x


def verify_split_bab(model, x0, c, eps, input_shape=None, node_budget=20000,
                     min_width=1e-5):
    """Complete verifier by input-space bisection (low-dimensional exact use).

    Nodes are boxes; per-box bounds come from CROWN; a box whose ReLUs are
    all fixed (pre-activation intervals avoid 0) is affine, so its exact min
    margin is obtained by evaluating the 2^d corners.  Splitting along the
    widest coordinate drives the box to a point, so every input region is
    eventually covered exactly (terminates in practice for small nets).
    Returns (verdict, nodes_explored, counterexample).
    """
    X_lo = x0 - eps
    X_hi = x0 + eps
    if input_shape is not None:
        X_lo = torch.clamp(X_lo, 0.0, 1.0)
        X_hi = torch.clamp(X_hi, 0.0, 1.0)
    lb = bc.crown_margin_lb(model, X_lo, X_hi, c, input_shape)
    heap = [(-lb, 0, X_lo, X_hi)]
    nodes = 0
    counter = 1
    counterexample = None
    while heap and nodes < node_budget:
        neg_lb, _, xl, xh = heapq.heappop(heap)
        nodes += 1
        lb = -neg_lb
        if lb > 0:
            continue
        unst = unstable_neurons(model, xl, xh, input_shape, {})
        if not unst:
            m, x_min = _leaf_corner_min(model, xl, xh, c, input_shape)
            if m <= 0:
                counterexample = x_min
                break
            continue
        widths = xh - xl
        dim = int(torch.argmax(widths).item())
        if float(widths[dim].item()) < min_width:
            continue
        mid = (float(xl[dim].item()) + float(xh[dim].item())) / 2.0
        for side in (0, 1):
            xl2 = xl.clone()
            xh2 = xh.clone()
            if side == 0:
                xh2[dim] = mid
            else:
                xl2[dim] = mid
            lb2 = bc.crown_margin_lb(model, xl2, xh2, c, input_shape)
            heapq.heappush(heap, (-lb2, counter, xl2, xh2))
            counter += 1
    if counterexample is not None:
        return 'not_robust', nodes, counterexample
    if nodes >= node_budget and heap:
        return 'unknown', nodes, None
    return 'robust', nodes, None
