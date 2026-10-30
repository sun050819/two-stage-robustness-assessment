# -*- coding: utf-8 -*-
"""
Constructive demonstration that the strict-inclusion branch of the
Krawczyk existence test (Lemma 3, "Conditional Brouwer inclusion") does
fire on real interval equations (merged manuscript, Section 7.4.2).

The paper's certification stage only ever triggered the midpoint-witness
route (g(m) <= 0) on the tested workloads. This script isolates the
*other* branch of the Krawczyk test: when K1(X) is strictly contained in
the interior of X, an interval-Newton theorem guarantees a root of g in X
without evaluating the midpoint. We exhibit two interval equations
(1-D and 2-D) where the strict inclusion K1(X) subset int(X) holds, so
the mechanism is not vacuous.

K1(X) = m - Y g(m) + (I - Y J_X) (X - m),
with J_X an interval extension of the Jacobian over X and Y a
preconditioner (identity in the 1-D demo, rank-one in 2-D).
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(BASE, '..', 'results'))
os.makedirs(OUT_DIR, exist_ok=True)


def iv_add(a, b):   return (a[0] + b[0], a[1] + b[1])
def iv_sub(a, b):   return (a[0] - b[1], a[1] - b[0])
def iv_mul(a, b):
    xs = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(xs), max(xs))
def iv_scale(s, a):
    return (s * a[0], s * a[1]) if s >= 0 else (s * a[1], s * a[0])
def iv_recip(a):
    if a[0] > 0 or a[1] < 0:
        return (1.0 / a[1], 1.0 / a[0])
    raise ValueError('interval crosses zero')


def strict_inside(k1, x):
    """K1(X) strictly contained in the interior of X (elementwise)."""
    return k1[0] > x[0] and k1[1] < x[1]


# ---------------------------------------------------------------------------
# 1-D demo: g(x) = x^2 - 4, X = [1, 3], root x = 2.
# J_X = [2, 6] (interval derivative over X); Y = 1 / g'(m) = 1/4 at m = 2.
# ---------------------------------------------------------------------------
X1 = (1.0, 3.0)
m1 = 2.0
JX1 = (2.0, 6.0)          # interval extension of g' over X
Y1 = 1.0 / 4.0            # 1 / g'(m)
g_m1 = m1 * m1 - 4.0      # 0
term_a = iv_scale(Y1, (g_m1, g_m1))                   # Y g(m) = 0
lhs = iv_sub((m1, m1), term_a)                      # m - Y g(m)
I_minus = iv_sub((1.0, 1.0), iv_scale(Y1, JX1))   # 1 - Y J_X
x_m = iv_sub(X1, (m1, m1))                        # X - m
K1_1d = iv_add(lhs, iv_mul(I_minus, x_m))

# ---------------------------------------------------------------------------
# 2-D demo: g(x,y) = (x^2 + y^2 - 4, x - y), root (sqrt(2), sqrt(2)) ~ (1.414, 1.414)
# X = [1.3, 1.6] x [1.3, 1.6]; m = (1.45, 1.45).
# J_X (interval Jacobian over X): dg1/dx = 2x in [2.6, 3.2]; dg1/dy = 2y in [2.6, 3.2];
# dg2/dx = 1; dg2/dy = -1.
# Rank-one preconditioner Y = J(m)^{-1} (exact inverse at the midpoint here).
# ---------------------------------------------------------------------------
X2 = ((1.3, 1.6), (1.3, 1.6))
m2 = (1.45, 1.45)
Jm = [[2 * m2[0], 2 * m2[1]], [1.0, -1.0]]        # J(m)
det = Jm[0][0] * Jm[1][1] - Jm[0][1] * Jm[1][0]
Y2 = [[Jm[1][1] / det, -Jm[0][1] / det],
      [-Jm[1][0] / det, Jm[0][0] / det]]          # J(m)^{-1}

def g2(p):
    x, y = p
    return (x * x + y * y - 4.0, x - y)

def jac2_interval():
    return ([[2.6, 3.2], [2.6, 3.2]], [[1.0, 1.0], [-1.0, -1.0]])

g_m2 = g2(m2)
JX2 = jac2_interval()

# K1 = m - Y g(m) + (I - Y J_X)(X - m), interval arithmetic on rows.
def mat_vec_iv(A, v):
    n = len(A)
    out = []
    for i in range(n):
        acc = (0.0, 0.0)
        for j in range(n):
            a_iv = (A[i][j], A[i][j]) if isinstance(A[i][j], float) else A[i][j]
            acc = iv_add(acc, iv_mul(a_iv, v[j]))
        out.append(acc)
    return out

def mat_mul_iv(A, B):
    n = len(A)
    out = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = (0.0, 0.0)
            for k in range(n):
                acc = iv_add(acc, iv_mul(A[i][k], B[k][j]))
            out[i][j] = acc
    return out

def identity_iv(n):
    return [[(1.0, 1.0) if i == j else (0.0, 0.0) for j in range(n)] for i in range(n)]

Yg = mat_vec_iv(Y2, ((g_m2[0], g_m2[0]), (g_m2[1], g_m2[1])))  # Y g(m)
m_vec = ((m2[0], m2[0]), (m2[1], m2[1]))
lhs2 = tuple(iv_sub(m_vec[i], Yg[i]) for i in range(2))
# Promote the scalar preconditioner matrix to interval form:
Y2iv = [[(Y2[i][j], Y2[i][j]) for j in range(2)] for i in range(2)]
IYJ = identity_iv(2)
tmp = mat_mul_iv(Y2iv, JX2)
for i in range(2):
    for j in range(2):
        IYJ[i][j] = iv_sub((1.0, 1.0) if i == j else (0.0, 0.0), tmp[i][j])
x_m2 = tuple(iv_sub(X2[i], (m2[i], m2[i])) for i in range(2))
prod = mat_vec_iv(IYJ, x_m2)
K1_2d = tuple(iv_add(lhs2[i], prod[i]) for i in range(2))

result = {
    'demo_1d': {
        'g': 'x^2 - 4', 'X': X1, 'midpoint': m1,
        'g_midpoint': g_m1, 'J_X_interval': JX1, 'preconditioner': Y1,
        'K1': K1_1d,
        'strict_inclusion_K1_in_int_X': strict_inside(K1_1d, X1),
        'root_in_K1': 2.0 in (K1_1d[0], K1_1d[1]) or (K1_1d[0] <= 2.0 <= K1_1d[1]),
        'root_of_g_in_X': True,
    },
    'demo_2d': {
        'g': '(x^2 + y^2 - 4, x - y)',
        'X': X2, 'midpoint': m2, 'g_midpoint': g_m2,
        'J_X_interval': JX2, 'preconditioner': 'J(m)^{-1}',
        'K1': K1_2d,
        'strict_inclusion_K1_in_int_X': all(strict_inside(K1_2d[i], X2[i]) for i in range(2)),
        'true_root': (2.0 ** 0.5, 2.0 ** 0.5),
        'root_in_K1': (X2[0][0] <= 2.0 ** 0.5 <= X2[0][1] and
                       K1_2d[0][0] <= 2.0 ** 0.5 <= K1_2d[0][1] and
                       K1_2d[1][0] <= 2.0 ** 0.5 <= K1_2d[1][1]),
        'root_of_g_in_X': True,
    },
}

fn = os.path.join(OUT_DIR, 'lemma3_demo.json')
with open(fn, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=1)
print(json.dumps(result, indent=1), flush=True)
print(f'saved {fn}', flush=True)
