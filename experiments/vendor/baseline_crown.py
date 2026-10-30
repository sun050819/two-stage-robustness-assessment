# -*- coding: utf-8 -*-
"""
CROWN (Zhang et al., NeurIPS 2018) certified bounds for ReLU networks with
Linear/Conv2d/ReLU/Flatten layers, implemented directly on top of PyTorch.

For every output neuron k we propagate two linear forms in the input x:
    f_k(x) >= a_l^k x + b_l^k   and   f_k(x) <= a_u^k x + b_u^k.
Backward propagation:
  * affine layer z = W y + bias: a <- W^T a,  b <- b + a^T bias (with the
    pre-update a);  for Conv2d the transposed operator is conv_transpose2d;
  * ReLU with pre-activation bounds [l,u]:
      lower: alpha_l in {0,1} chosen by the sign of the aggregated
             coefficient (CROWN rule);
      upper: secant line alpha_u = u/(u-l), beta_u = -l*u/(u-l) for
             l<0<u, 0 for dead neurons, identity for live neurons.
At the input box the linear form is evaluated soundly with the
positive/negative split.  The certified margin lower bound of
g = f_c - max_{j!=c} f_j is min_j [ lb(f_c) - ub(f_j) ] over the box.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import verifier_lib as vl


def _modules(model):
    if hasattr(model, 'net') and isinstance(model.net, nn.Sequential):
        return list(model.net)
    if isinstance(model, nn.Sequential):
        return list(model)
    raise TypeError('expected nn.Sequential or wrapper with .net')


def _ibp_forward(xl, xh, mod):
    """One affine/ReLU/Flatten IBP step; returns (xl, xh)."""
    if isinstance(mod, nn.Linear):
        if xl.dim() > 1:
            xl = xl.flatten()
            xh = xh.flatten()
        return vl.linear_bounds(mod.weight, mod.bias, xl, xh)
    if isinstance(mod, nn.Conv2d):
        Wp = torch.clamp(mod.weight, min=0.0)
        Wn = torch.clamp(mod.weight, max=0.0)
        yl = (F.conv2d(xl.unsqueeze(0), Wp, mod.bias, mod.stride, mod.padding)
              + F.conv2d(xh.unsqueeze(0), Wn, None, mod.stride, mod.padding)).squeeze(0)
        yh = (F.conv2d(xh.unsqueeze(0), Wp, mod.bias, mod.stride, mod.padding)
              + F.conv2d(xl.unsqueeze(0), Wn, None, mod.stride, mod.padding)).squeeze(0)
        return yl, yh
    if isinstance(mod, nn.ReLU):
        return torch.clamp(xl, min=0.0), torch.clamp(xh, min=0.0)
    if isinstance(mod, nn.Flatten):
        return xl.flatten(), xh.flatten()
    raise TypeError(f'unsupported module: {type(mod)}')


def crown_bounds_logits(model, x_lo, x_hi, input_shape=None, refine=False):
    """Per-output lower and upper bounds of all logits on the box.

    refine=True enables RefineIntermediateBounds (the standard CROWN-IBP
    intermediate-bound refinement used by auto-LiRPA): for every ReLU, the
    pre-activation bounds produced by the IBP forward pass are tightened by
    a CROWN backward from the source affine output (identity coefficients)
    to the input, then merged by max/min with the IBP bounds. All
    tightening steps are sound (box evaluation with the positive/negative
    split and max/min of sound bounds), so the refined bounds never
    invalidate the certified margin.
    """
    modules = _modules(model)
    xl, xh = x_lo, x_hi
    if input_shape is not None:
        xl = xl.view(*input_shape)
        xh = xh.view(*input_shape)
    preacts = []       # per ReLU, in module order
    relu_aff_out = []  # per ReLU: (module idx of source affine, its out shape)
    aff_idx = None
    aff_ops = []       # affine (linear/conv) records: in/out shape
    flatten_shapes = {}  # module index -> pre-flatten input shape
    for idx, mod in enumerate(modules):
        if isinstance(mod, nn.Linear):
            in_shape = tuple(xl.shape)
            xl, xh = _ibp_forward(xl, xh, mod)
            aff_ops.append({'kind': 'linear', 'in_shape': in_shape,
                            'out_shape': tuple(xl.shape)})
            aff_idx = idx
        elif isinstance(mod, nn.Conv2d):
            in_shape = tuple(xl.shape)
            xl, xh = _ibp_forward(xl, xh, mod)
            aff_ops.append({'kind': 'conv', 'in_shape': in_shape,
                            'out_shape': tuple(xl.shape)})
            aff_idx = idx
        elif isinstance(mod, nn.ReLU):
            preacts.append((xl.clone(), xh.clone()))
            relu_aff_out.append((aff_idx, tuple(xl.shape)))
            # propagate the ReLU through the interval box (clamp to >= 0);
            # preacts keeps the *pre-activation* bounds used by the backward
            # ReLU relaxation, the box passed to the next layer must be the
            # post-activation enclosure.
            xl, xh = _ibp_forward(xl, xh, mod)
        elif isinstance(mod, nn.Flatten):
            flatten_shapes[idx] = tuple(xl.shape)
            xl, xh = _ibp_forward(xl, xh, mod)
    K = xl.shape[0]

    if refine:
        relu_mod_pos = {}
        rp = 0
        for idx, mod in enumerate(modules):
            if isinstance(mod, nn.ReLU):
                relu_mod_pos[idx] = rp
                rp += 1
        for r in range(len(preacts)):
            src_aff_idx, out_shape = relu_aff_out[r]
            if src_aff_idx is None:
                continue
            d = 1
            for s in out_shape:
                d *= s
            a_l = torch.eye(d).view(d, *out_shape)
            b_l = torch.zeros(d)
            a_u = a_l.clone()
            b_u = b_l.clone()
            for ridx in range(src_aff_idx, -1, -1):
                mod = modules[ridx]
                if isinstance(mod, nn.Linear):
                    b_l = b_l + (a_l @ mod.bias)
                    b_u = b_u + (a_u @ mod.bias)
                    a_l = a_l @ mod.weight
                    a_u = a_u @ mod.weight
                elif isinstance(mod, nn.Conv2d):
                    red = tuple(range(1, a_l.dim()))
                    b_l = b_l + (a_l * mod.bias.view(-1, 1, 1)).sum(dim=red)
                    b_u = b_u + (a_u * mod.bias.view(-1, 1, 1)).sum(dim=red)
                    op = tuple(1 if s == 2 else 0 for s in mod.stride)
                    a_l = F.conv_transpose2d(a_l, mod.weight, None, mod.stride,
                                             mod.padding, output_padding=op)
                    a_u = F.conv_transpose2d(a_u, mod.weight, None, mod.stride,
                                             mod.padding, output_padding=op)
                elif isinstance(mod, nn.ReLU):
                    pl, ph = preacts[relu_mod_pos[ridx]]
                    dead_l = ph <= 0
                    live_l = pl >= 0
                    denom = (ph - pl).clamp(min=1e-12)
                    alpha_neg = torch.where(
                        live_l, torch.ones_like(ph),
                        torch.where(dead_l, torch.zeros_like(ph), ph / denom))
                    beta_neg = torch.where(
                        live_l, torch.zeros_like(ph),
                        torch.where(dead_l, torch.zeros_like(ph),
                                    -pl * ph / denom))
                    neg = (a_l < 0).float()
                    alpha_pos = torch.where(live_l, torch.ones_like(ph),
                                            torch.zeros_like(ph))
                    b_l = b_l + ((a_l * beta_neg) * neg).sum(
                        dim=tuple(range(1, a_l.dim())))
                    a_l = a_l * (neg * alpha_neg + (1 - neg) * alpha_pos)
                    dead = ph <= 0
                    live = pl >= 0
                    alpha_u = torch.where(
                        live, torch.ones_like(ph),
                        torch.where(dead, torch.zeros_like(ph),
                                    ph / (ph - pl)))
                    beta_u = torch.where(
                        live, torch.zeros_like(ph),
                        torch.where(dead, torch.zeros_like(ph),
                                    -pl * ph / (ph - pl)))
                    pos = (a_u >= 0).float()
                    neg_u = (1 - pos) * live.float()
                    b_u = b_u + ((a_u * beta_u) * pos).sum(
                        dim=tuple(range(1, a_u.dim())))
                    a_u = a_u * (pos * alpha_u + neg_u)
                elif isinstance(mod, nn.Flatten):
                    in_shape = flatten_shapes[ridx]
                    a_l = a_l.view(d, *in_shape)
                    a_u = a_u.view(d, *in_shape)
            lo_r, _ = _box_lin_each(a_l, b_l, x_lo, x_hi)
            _, hi_r = _box_lin_each(a_u, b_u, x_lo, x_hi)
            pl0, ph0 = preacts[r]
            preacts[r] = (torch.maximum(pl0, lo_r.view(*out_shape)),
                          torch.minimum(ph0, hi_r.view(*out_shape)))

    # initialize per-output coefficients
    a_l = torch.eye(K)
    b_l = torch.zeros(K)
    a_u = torch.eye(K)
    b_u = torch.zeros(K)
    relu_pos = len(preacts) - 1
    for r_idx in range(len(modules) - 1, -1, -1):
        mod = modules[r_idx]
        if isinstance(mod, nn.Linear):
            bias_contrib_l = a_l @ mod.bias
            bias_contrib_u = a_u @ mod.bias
            a_l = a_l @ mod.weight
            a_u = a_u @ mod.weight
            b_l = b_l + bias_contrib_l
            b_u = b_u + bias_contrib_u
        elif isinstance(mod, nn.Conv2d):
            red_dims = tuple(range(1, a_l.dim()))
            b_l = b_l + (a_l * mod.bias.view(-1, 1, 1)).sum(dim=red_dims)
            b_u = b_u + (a_u * mod.bias.view(-1, 1, 1)).sum(dim=red_dims)
            op = tuple(1 if s == 2 else 0 for s in mod.stride)
            a_l = F.conv_transpose2d(a_l, mod.weight, None, mod.stride,
                                     mod.padding, output_padding=op)
            a_u = F.conv_transpose2d(a_u, mod.weight, None, mod.stride,
                                     mod.padding, output_padding=op)
        elif isinstance(mod, nn.ReLU):
            pl, ph = preacts[relu_pos]
            # ---- lower bound ----
            # a_l >= 0: use a lower envelope of relu (relu(z) >= L(z)):
            #   live (pl>=0): L = z              -> alpha=1, beta=0
            #   dead (ph<=0): L = 0              -> alpha=0, beta=0
            #   straddle     : L = 0 is tighter  -> alpha=0, beta=0
            # a_l < 0: use an UPPER envelope U(z) >= relu(z) (inequality flips):
            #   live (pl>=0): U = z              -> alpha=1, beta=0
            #   dead (ph<=0): U = 0              -> alpha=0, beta=0
            #   straddle     : U = (z-pl)*ph/(ph-pl)
            #                              -> alpha=ph/(ph-pl), beta=-pl*ph/(ph-pl)
            dead_l = ph <= 0
            live_l = pl >= 0
            strad_l = ~(live_l | dead_l)
            denom = (ph - pl).clamp(min=1e-12)
            # envelope for a_l < 0 (upper envelope; straddle uses the secant)
            alpha_neg = torch.where(
                live_l, torch.ones_like(ph),
                torch.where(dead_l, torch.zeros_like(ph), ph / denom))
            beta_neg = torch.where(
                live_l, torch.zeros_like(ph),
                torch.where(dead_l, torch.zeros_like(ph), -pl * ph / denom))
            neg = (a_l < 0).float()
            # for a_l >= 0 use L(z): live -> z (alpha 1), else 0
            alpha_pos = torch.where(live_l, torch.ones_like(ph),
                                    torch.zeros_like(ph))
            b_l = b_l + ((a_l * beta_neg) * neg).sum(
                dim=tuple(range(1, a_l.dim())))
            a_l = a_l * (neg * alpha_neg + (1 - neg) * alpha_pos)
            # ---- upper bound ----
            # a_u >= 0: use upper envelope U(z) (secant on straddle);
            # a_u < 0 : live -> relu(z)=z exactly (keep a), dead/straddle ->
            #           use lower envelope L(z)=0 (drop the term).
            dead = ph <= 0
            live = pl >= 0
            u = ph
            l = pl
            alpha_u = torch.where(live, torch.ones_like(u),
                                  torch.where(dead, torch.zeros_like(u),
                                              u / (u - l)))
            beta_u = torch.where(live, torch.zeros_like(u),
                                 torch.where(dead, torch.zeros_like(u),
                                             -l * u / (u - l)))
            pos = (a_u >= 0).float()
            neg_u = (1 - pos) * live.float()
            b_u = b_u + ((a_u * beta_u) * pos).sum(dim=tuple(range(1, a_u.dim())))
            a_u = a_u * (pos * alpha_u + neg_u)
            relu_pos -= 1
        elif isinstance(mod, nn.Flatten):
            # reshape coefficients back to the pre-flatten shape
            in_shape = flatten_shapes[r_idx]
            a_l = a_l.view(a_l.shape[0], *in_shape)
            a_u = a_u.view(a_u.shape[0], *in_shape)
    lo = torch.empty(K)
    hi = torch.empty(K)
    for k in range(K):
        lo[k], _ = _box_lin(a_l[k], b_l[k], x_lo, x_hi)
        _, hi[k] = _box_lin(a_u[k], b_u[k], x_lo, x_hi)
    return lo, hi


def _box_lin_each(a, b, x_lo, x_hi):
    """Sound interval evaluation of a x + b over [x_lo, x_hi], per-row of a.
    a: [m, *in_shape]; b: [m]; returns per-row (lo, hi)."""
    a = a.view(a.shape[0], -1)
    xl = x_lo.view(-1)
    xh = x_hi.view(-1)
    ap = torch.clamp(a, min=0.0)
    an = torch.clamp(a, max=0.0)
    lo = (ap * xl + an * xh).sum(dim=1) + b
    hi = (ap * xh + an * xl).sum(dim=1) + b
    return lo, hi


def _box_lin(a, b, x_lo, x_hi):
    """Sound interval evaluation of a x + b over [x_lo, x_hi] (a flattened)."""
    a = a.flatten()
    ap = torch.clamp(a, min=0.0)
    an = torch.clamp(a, max=0.0)
    lo = (ap * x_lo + an * x_hi).sum() + b
    hi = (ap * x_hi + an * x_lo).sum() + b
    return lo, hi


def crown_margin_lb(model, x_lo, x_hi, c, input_shape=None):
    """CROWN certified lower bound of the margin on the box."""
    lo, hi = crown_bounds_logits(model, x_lo, x_hi, input_shape)
    others = [j for j in range(lo.shape[0]) if j != c]
    return float((lo[c] - torch.max(hi[others])).item())


def verify_crown(model, x0, c, eps, input_shape=None):
    X_lo = x0 - eps
    X_hi = x0 + eps
    return 'robust' if crown_margin_lb(model, X_lo, X_hi, c, input_shape) > 0 \
        else 'undecided'
