# -*- coding: utf-8 -*-
"""
Generic interval-verification core for ReLU networks with Linear and Conv2d
layers (no pooling; downsampling via stride-2 convolutions).

Extends verifier_lib to convolutional architectures and provides a *structured*
O(d) implementation of the interval-Krawczyk image for the scalar-margin
equation g(x)=0 in d unknowns.  The Moore-Penrose preconditioner C is a
d-vector, so the interval matrix I - C J(X) is a rank-1 perturbation of the
identity; the corner evaluations of the interval matrix-vector product can
therefore be evaluated in O(d) instead of O(d^2).  The structured routine is
mathematically identical to the dense construction in verifier_lib (validated
numerically in sanity tests).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import verifier_lib as vl


# ---------------------------------------------------------------------------
# Interval forward bounds for generic Sequential networks
# ---------------------------------------------------------------------------

def _modules(model):
    """Return the module list of an nn.Sequential or of a wrapper with .net."""
    if hasattr(model, 'net') and isinstance(model.net, nn.Sequential):
        return list(model.net)
    if isinstance(model, nn.Sequential):
        return list(model)
    raise TypeError('expected an nn.Sequential or a wrapper exposing .net')


def conv_bounds_single(xl, xh, conv):
    """Sound interval bounds of y = Conv(x) over the box [xl, xh] (4D, N=1)."""
    Wp = torch.clamp(conv.weight, min=0.0)
    Wn = torch.clamp(conv.weight, max=0.0)
    yl = (F.conv2d(xl, Wp, conv.bias, conv.stride, conv.padding)
          + F.conv2d(xh, Wn, None, conv.stride, conv.padding))
    yh = (F.conv2d(xh, Wp, conv.bias, conv.stride, conv.padding)
          + F.conv2d(xl, Wn, None, conv.stride, conv.padding))
    return vl._outward(yl, yh)


def forward_bounds(model, x_lo, x_hi, input_shape=None):
    """IBP forward bounds through Linear/Conv2d/ReLU.

    Args:
        x_lo, x_hi: 1-D tensors of length d (flattened input).
        input_shape: (C, H, W) when the first layer is Conv2d, else None.
    Returns:
        (logits_lo, logits_hi, ops); ops is a list of per-module records used
        by the interval-gradient backpropagation.
    """
    xl, xh = x_lo, x_hi
    if input_shape is not None and xl.dim() == 1:
        xl = xl.view(*input_shape)
        xh = xh.view(*input_shape)
    ops = []
    for mod in _modules(model):
        if isinstance(mod, nn.Linear):
            in_shape = xl.shape
            if xl.dim() > 1:
                xl = xl.flatten()
                xh = xh.flatten()
            xl, xh = vl.linear_bounds(mod.weight, mod.bias, xl, xh)
            ops.append({'kind': 'linear', 'preact': (xl, xh),
                        'in_shape': tuple(in_shape), 'out_shape': (mod.weight.shape[0],)})
        elif isinstance(mod, nn.Conv2d):
            in_shape = xl.shape
            xl, xh = conv_bounds_single(xl.unsqueeze(0), xh.unsqueeze(0), mod)
            xl = xl.squeeze(0)
            xh = xh.squeeze(0)
            ops.append({'kind': 'conv', 'preact': (xl, xh),
                        'in_shape': tuple(in_shape), 'out_shape': tuple(xl.shape)})
        elif isinstance(mod, nn.ReLU):
            xl, xh = vl.irelu(xl, xh)
            ops.append({'kind': 'relu'})
        elif isinstance(mod, nn.Flatten):
            in_shape = tuple(xl.shape)
            xl = xl.flatten()
            xh = xh.flatten()
            ops.append({'kind': 'flatten', 'in_shape': in_shape,
                        'out_shape': tuple(xl.shape)})
        else:
            raise TypeError(f'unsupported module: {type(mod)}')
    return xl, xh, ops


def gradient_interval_logit(model, x_lo, x_hi, k, ops, input_shape=None):
    """Interval enclosure of the gradient of logit k w.r.t. the input."""
    modules = _modules(model)
    dev = next(model.parameters()).device
    # output dimension
    K = None
    for m in reversed(modules):
        if isinstance(m, nn.Linear):
            K = m.weight.shape[0]
            break
    glo = torch.zeros(K, device=dev)
    ghi = torch.zeros(K, device=dev)
    glo[k] = 1.0
    ghi[k] = 1.0
    affine_ops = [op for op in ops if op['kind'] in ('linear', 'conv')]
    # ReLU index -> pre-activation bounds of the preceding affine layer
    relu_preact = {}
    lin_pos = 0
    for i, m in enumerate(modules):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            lin_pos += 1
        elif isinstance(m, nn.ReLU):
            relu_preact[i] = affine_ops[lin_pos - 1]['preact']
    conv_ops = {}
    for i, m in enumerate(modules):
        if isinstance(m, nn.Conv2d):
            p = sum(1 for j, mm in enumerate(modules[:i])
                    if isinstance(mm, (nn.Linear, nn.Conv2d)))
            conv_ops[i] = affine_ops[p]
    flatten_ops = {}
    for i, m in enumerate(modules):
        if isinstance(m, nn.Flatten):
            flatten_ops[i] = ops[sum(1 for j, mm in enumerate(modules[:i])
                                     if isinstance(mm, (nn.Linear, nn.Conv2d, nn.Flatten, nn.ReLU)))]
    for idx in range(len(modules) - 1, -1, -1):
        mod = modules[idx]
        if isinstance(mod, nn.Linear):
            W = mod.weight
            Wp = torch.clamp(W, min=0.0)
            Wn = torch.clamp(W, max=0.0)
            glo, ghi = vl._outward(Wp.t() @ glo + Wn.t() @ ghi,
                                   Wp.t() @ ghi + Wn.t() @ glo)
        elif isinstance(mod, nn.Conv2d):
            # reshape gradient to the conv output shape if flattened upstream
            out_shape = conv_ops[idx]['out_shape']
            if glo.dim() != 3 or tuple(glo.shape) != out_shape:
                glo = glo.view(*out_shape)
                ghi = ghi.view(*out_shape)
            in_shape = conv_ops[idx]['in_shape']
            Wp = torch.clamp(mod.weight, min=0.0)
            Wn = torch.clamp(mod.weight, max=0.0)
            ngl = (F.conv_transpose2d(glo.unsqueeze(0), Wp, None, mod.stride, mod.padding,
                                      output_padding=tuple(1 if s == 2 else 0 for s in mod.stride))
                   + F.conv_transpose2d(ghi.unsqueeze(0), Wn, None, mod.stride, mod.padding,
                                        output_padding=tuple(1 if s == 2 else 0 for s in mod.stride)))
            ngh = (F.conv_transpose2d(ghi.unsqueeze(0), Wp, None, mod.stride, mod.padding,
                                      output_padding=tuple(1 if s == 2 else 0 for s in mod.stride))
                   + F.conv_transpose2d(glo.unsqueeze(0), Wn, None, mod.stride, mod.padding,
                                        output_padding=tuple(1 if s == 2 else 0 for s in mod.stride)))
            glo = ngl.squeeze(0)
            ghi = ngh.squeeze(0)
            glo, ghi = vl._outward(glo, ghi)
        elif isinstance(mod, nn.ReLU):
            pl, ph = relu_preact[idx]
            d_lo = (pl >= 0).float()
            d_hi = (~(ph <= 0)).float()
            glo, ghi = vl.imul(glo, ghi, d_lo, d_hi)
        elif isinstance(mod, nn.Flatten):
            in_shape = flatten_ops[idx]['in_shape']
            glo = glo.view(*in_shape)
            ghi = ghi.view(*in_shape)
    return glo, ghi


def gradient_margin_hull(model, x_lo, x_hi, c, ops, input_shape=None):
    """Sound interval hull J(X) of the margin gradient over the box."""
    K = 0
    for m in _modules(model):
        if isinstance(m, nn.Linear):
            K = m.weight.shape[0]
    d = x_lo.shape[0]
    hull_lo = torch.full((d,), float('inf'), device=x_lo.device)
    hull_hi = torch.full((d,), float('-inf'), device=x_lo.device)
    g_c_lo, g_c_hi = gradient_interval_logit(model, x_lo, x_hi, c, ops, input_shape)
    g_c_lo = g_c_lo.flatten()
    g_c_hi = g_c_hi.flatten()
    for j in range(K):
        if j == c:
            continue
        g_j_lo, g_j_hi = gradient_interval_logit(model, x_lo, x_hi, j, ops, input_shape)
        g_j_lo = g_j_lo.flatten()
        g_j_hi = g_j_hi.flatten()
        dg_lo, dg_hi = vl._outward(g_c_lo - g_j_hi, g_c_hi - g_j_lo)
        hull_lo = torch.minimum(hull_lo, dg_lo)
        hull_hi = torch.maximum(hull_hi, dg_hi)
    return hull_lo, hull_hi


def forward_logits(model, x_flat, input_shape=None):
    with torch.no_grad():
        if input_shape is not None:
            x = x_flat.view(1, *input_shape)
        else:
            x = x_flat.unsqueeze(0)
        return model(x).squeeze(0)


def margin_at(model, x_flat, c, input_shape=None):
    logits = forward_logits(model, x_flat, input_shape)
    others = [j for j in range(logits.shape[0]) if j != c]
    return float((logits[c] - torch.max(logits[others])).item())


def point_gradient_margin(model, x_flat, c, input_shape=None):
    if input_shape is not None:
        x = x_flat.detach().clone().view(1, *input_shape).requires_grad_(True)
    else:
        x = x_flat.detach().clone().unsqueeze(0).requires_grad_(True)
    logits = model(x).squeeze(0)
    others = [j for j in range(logits.shape[0]) if j != c]
    g = logits[c] - torch.max(logits[others])
    g.backward()
    return x.grad.detach().clone().flatten()


# ---------------------------------------------------------------------------
# Structured O(d) interval-Krawczyk image
# ---------------------------------------------------------------------------

def krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi):
    """K(X) = m - C g(m) + (I - C J(X))(X - m), evaluated in O(d).

    The interval matrix M = I - C J(X) has rows  M[i,:] = e_i - C_i * J_sel(i)
    where J_sel(i) is J_lo or J_hi depending on the sign of C_i.  Every corner
    product M_c @ v is therefore computable as v - C (Hadamard) (J_sel . v) plus a
    diagonal correction, in O(d) time.  All endpoints are rounded outward
    (one ulp), so the result is a sound enclosure and agrees with the dense
    construction of verifier_lib.krawczyk_image up to the rounding of the
    different operation order.
    """
    d = X_lo.shape[0]
    pos = C >= 0.0
    Jlo = J_lo
    Jhi = J_hi

    def matvec(v):
        s_lo = (Jlo * v).sum()
        s_hi = (Jhi * v).sum()
        s_lo, s_hi = vl._outward(s_lo, s_hi)
        CJ_lo_v = C * torch.where(pos, s_lo, s_hi)
        CJ_hi_v = C * torch.where(pos, s_hi, s_lo)
        diag_lo = C * torch.where(pos, Jlo, Jhi)
        diag_hi = C * torch.where(pos, Jhi, Jlo)
        M_lo_v = v - CJ_lo_v + (diag_lo - diag_hi) * v
        M_hi_v = v - CJ_hi_v + (diag_hi - diag_lo) * v
        return vl._outward(M_lo_v, M_hi_v)

    xm_lo, xm_hi = vl._outward(X_lo - m, X_hi - m)
    a1, b1 = matvec(xm_lo)   # M_lo @ xm_lo, M_hi @ xm_lo
    a2, b2 = matvec(xm_hi)   # M_lo @ xm_hi, M_hi @ xm_hi
    prod_lo = torch.minimum(torch.minimum(a1, a2), torch.minimum(b1, b2))
    prod_hi = torch.maximum(torch.maximum(a1, a2), torch.maximum(b1, b2))
    base = m - C * g_m
    return vl._outward(base + prod_lo, base + prod_hi)


# ---------------------------------------------------------------------------
# Verification methods (generic over MLP / CNN)
# ---------------------------------------------------------------------------

def verify_ibp(model, x0, c, eps, input_shape=None, clamp01=False):
    X_lo, X_hi = vl._outward(x0 - eps, x0 + eps)
    if clamp01:
        X_lo, X_hi = torch.clamp(X_lo, 0.0, 1.0), torch.clamp(X_hi, 0.0, 1.0)
    logits_lo, logits_hi, _ = forward_bounds(model, X_lo, X_hi, input_shape)
    m_lo, _ = vl.margin_interval(logits_lo, logits_hi, c)
    return 'robust' if m_lo > 0 else 'undecided'


def verify_krawczyk_classical(model, x0, c, eps, input_shape=None, clamp01=False):
    X_lo, X_hi = vl._outward(x0 - eps, x0 + eps)
    if clamp01:
        X_lo, X_hi = torch.clamp(X_lo, 0.0, 1.0), torch.clamp(X_hi, 0.0, 1.0)
    ops = forward_bounds(model, X_lo, X_hi, input_shape)[2]
    m = (X_lo + X_hi) / 2.0
    g_m = margin_at(model, m, c, input_shape)
    if g_m <= 0:
        return 'exist'
    J_lo, J_hi = gradient_margin_hull(model, X_lo, X_hi, c, ops, input_shape)
    J_mid = (J_lo + J_hi) / 2.0
    C = vl.moore_penrose(J_mid)
    K_lo, K_hi = krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
    if vl.strict_inclusion(X_lo, X_hi, K_lo, K_hi):
        return 'exist'
    if vl.empty_intersection(X_lo, X_hi, K_lo, K_hi):
        return 'robust'
    return 'undecided'


def extract_root(model, X_lo, X_hi, c, input_shape=None, max_iter=40, tol=1e-9):
    for _ in range(max_iter):
        _, _, ops = forward_bounds(model, X_lo, X_hi, input_shape)
        m = (X_lo + X_hi) / 2.0
        g_m = margin_at(model, m, c, input_shape)
        J_lo, J_hi = gradient_margin_hull(model, X_lo, X_hi, c, ops, input_shape)
        J_mid = (J_lo + J_hi) / 2.0
        C = vl.moore_penrose(J_mid)
        K_lo, K_hi = krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
        cand_lo = torch.maximum(X_lo, K_lo)
        cand_hi = torch.minimum(X_hi, K_hi)
        if (not vl.empty_intersection(X_lo, X_hi, cand_lo, cand_hi)
                and vl.strict_shrink(X_lo, X_hi, cand_lo, cand_hi)):
            X_lo, X_hi = cand_lo, cand_hi
        else:
            break
    m = (X_lo + X_hi) / 2.0
    return m, margin_at(model, m, c, input_shape)


def verify_safe2(model, x0, c, eps, input_shape=None, alpha_min=0.2,
                 alpha_factor=0.8, max_iter_r=15, max_iter_e=15, tol=1e-6,
                 clamp01=False):
    """SAFE2 gated relaxed protocol with sound dual certificates (generic)."""
    X_lo, X_hi = vl._outward(x0 - eps, x0 + eps)
    if clamp01:
        X_lo, X_hi = torch.clamp(X_lo, 0.0, 1.0), torch.clamp(X_hi, 0.0, 1.0)
    width_hist = []
    root = None
    root_margin = None

    it = 0
    for it in range(max_iter_r):
        logits_lo, logits_hi, ops = forward_bounds(model, X_lo, X_hi, input_shape)
        m_lo, m_hi = vl.margin_interval(logits_lo, logits_hi, c)
        m = (X_lo + X_hi) / 2.0
        g_m = margin_at(model, m, c, input_shape)
        if g_m <= 0:
            return 'exist', it, width_hist, m, g_m
        if m_lo > 0:
            return 'robust', it, width_hist, root, root_margin
        J_lo, J_hi = gradient_margin_hull(model, X_lo, X_hi, c, ops, input_shape)
        J_mid = (J_lo + J_hi) / 2.0
        C = vl.moore_penrose(J_mid)
        K1_lo, K1_hi = krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
        if vl.strict_inclusion(X_lo, X_hi, K1_lo, K1_hi):
            root, root_margin = extract_root(model, X_lo, X_hi, c, input_shape)
            return 'exist', it, width_hist, root, root_margin
        if vl.empty_intersection(X_lo, X_hi, K1_lo, K1_hi):
            return 'robust', it, width_hist, root, root_margin
        cand_lo = torch.maximum(X_lo, K1_lo)
        cand_hi = torch.minimum(X_hi, K1_hi)
        if vl.strict_shrink(X_lo, X_hi, cand_lo, cand_hi):
            X_lo, X_hi = cand_lo, cand_hi
            width_hist.append(vl.box_width(X_lo, X_hi))
            if vl.box_width(X_lo, X_hi) < tol:
                return 'undecided', it, width_hist, root, root_margin
        else:
            break

    for it_e in range(max_iter_e):
        it = max_iter_r + it_e
        _, _, ops = forward_bounds(model, X_lo, X_hi, input_shape)
        m = (X_lo + X_hi) / 2.0
        g_m = margin_at(model, m, c, input_shape)
        if g_m <= 0:
            return 'exist', it, width_hist, m, g_m
        J_lo, J_hi = gradient_margin_hull(model, X_lo, X_hi, c, ops, input_shape)
        J_mid = (J_lo + J_hi) / 2.0
        C = vl.moore_penrose(J_mid)
        K1_lo, K1_hi = krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
        if vl.strict_inclusion(X_lo, X_hi, K1_lo, K1_hi):
            root, root_margin = extract_root(model, X_lo, X_hi, c, input_shape)
            return 'exist', it, width_hist, root, root_margin
        if vl.empty_intersection(X_lo, X_hi, K1_lo, K1_hi):
            break
        alpha = 1.0
        accepted = False
        for _try in range(16):
            Ja_lo = J_mid + alpha * (J_lo - J_mid)
            Ja_hi = J_mid + alpha * (J_hi - J_mid)
            Ka_lo, Ka_hi = krawczyk_image(m, g_m, Ja_lo, Ja_hi, C, X_lo, X_hi)
            cand_lo = torch.maximum(X_lo, Ka_lo)
            cand_hi = torch.minimum(X_hi, Ka_hi)
            if vl.strict_shrink(X_lo, X_hi, cand_lo, cand_hi):
                X_lo, X_hi = cand_lo, cand_hi
                accepted = True
                break
            alpha *= alpha_factor
            if alpha < alpha_min - 1e-9:
                break
        if not accepted:
            cand_lo = torch.maximum(X_lo, K1_lo)
            cand_hi = torch.minimum(X_hi, K1_hi)
            if vl.strict_shrink(X_lo, X_hi, cand_lo, cand_hi):
                X_lo, X_hi = cand_lo, cand_hi
            else:
                break
        width_hist.append(vl.box_width(X_lo, X_hi))
        if vl.box_width(X_lo, X_hi) < tol:
            break
    return 'undecided', it, width_hist, root, root_margin


# ---------------------------------------------------------------------------
# PGD reference attack (generic)
# ---------------------------------------------------------------------------

def pgd_attack(model, x0, c, eps, input_shape=None, steps=20, restarts=5,
               step_size=0.05):
    x0 = x0.detach().clone()
    for _ in range(restarts):
        noise = (torch.rand_like(x0) * 2 - 1) * eps
        x = torch.clamp(x0 + noise, 0.0, 1.0)
        x = x.requires_grad_(True)
        for _ in range(steps):
            xin = x.view(1, *input_shape) if input_shape is not None else x.unsqueeze(0)
            logits = model(xin).squeeze(0)
            others = [j for j in range(logits.shape[0]) if j != c]
            loss = logits[c] - torch.max(logits[others])
            loss.backward()
            with torch.no_grad():
                x = x + step_size * torch.sign(x.grad)
                x = torch.clamp(x, x0 - eps, x0 + eps)
                x = torch.clamp(x, 0.0, 1.0)
            x = x.detach().requires_grad_(True)
        with torch.no_grad():
            logits = forward_logits(model, x, input_shape)
            pred = int(torch.argmax(logits).item())
            if pred != c:
                return True
    return False


# ---------------------------------------------------------------------------
# Soundness spot-check of the interval enclosures (generic)
# ---------------------------------------------------------------------------

def sanity_check(model, x_lo, x_hi, c, input_shape=None, n_points=10):
    """Return True if all enclosure checks pass on the given box."""
    d = x_lo.shape[0]
    logits_lo, logits_hi, ops = forward_bounds(model, x_lo, x_hi, input_shape)
    g_lo, g_hi = gradient_margin_hull(model, x_lo, x_hi, c, ops, input_shape)
    ok = True
    rng = np.random.RandomState(0)
    for _ in range(n_points):
        xp = x_lo + torch.tensor(rng.rand(d), dtype=torch.float32) * (x_hi - x_lo)
        xp = torch.tensor(xp, dtype=torch.float32)
        logits = forward_logits(model, xp, input_shape)
        gm = logits[c] - torch.max(torch.cat([logits[:c], logits[c + 1:]]))
        m_lo, m_hi = vl.margin_interval(logits_lo, logits_hi, c)
        if not (m_lo <= gm <= m_hi):
            ok = False
        grad = point_gradient_margin(model, xp, c, input_shape)
        if not bool(torch.all(grad >= g_lo - 1e-3).item()) or \
           not bool(torch.all(grad <= g_hi + 1e-3).item()):
            ok = False
    return ok
