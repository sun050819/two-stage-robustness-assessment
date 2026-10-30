import json, numpy as np, os

d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'budget_sweep_ext.json')))
for e, row in d.items():
    Ns = np.array(sorted([k for k in row.keys() if k.isdigit()], key=int), dtype=float)
    S = np.array([row[str(int(n))]['S'] for n in Ns], dtype=float)
    a, b = np.polyfit(Ns, S, 1)
    pred = a * Ns + b
    ss_res = ((S - pred) ** 2).sum()
    ss_tot = ((S - S.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"eps={e} lin: a={a:.6f} b={b:.2f} R2={r2:.4f}")
    print("  S  =", S.tolist())
    print("  pred=", pred.round(2).tolist())
