"""
Deep dive: the math inside scaled dot-product attention, verified numerically.

We implement the forward AND backward pass of attention manually, then check
our gradients against torch.autograd and finite differences.

The forward:
    S   = Q K^T / sqrt(d_k)      (scaled scores)
    P   = softmax(S)             (attention probabilities)
    Y   = P V                    (weighted sum of values)

The backward (chain rule):
    dL/dV = P^T (dL/dY)
    dL/dP = (dL/dY) V^T
    dL/dS = P * (dL/dP - sum_j dL/dP_j * P_j)     <-- softmax Jacobian trick
    dL/dQ = (1/sqrt(d_k)) (dL/dS) K
    dL/dK = (1/sqrt(d_k)) (dL/dS)^T Q

Why is the softmax row-wise trick so cheap?
    The full Jacobian of softmax is a (n x n) matrix per row:
    J_ij = P_i (delta_ij - P_j)
    But we never build it: (dL/dS)_i = P_i (dL/dP_i - sum_j P_j dL/dP_j)
    — that's O(n) instead of O(n^2) per row.

Why divide by sqrt(d_k)?
    Q, K entries ~ N(0, 1) → dot product of length d_k has variance d_k,
    std ~ sqrt(d_k). Without scaling, softmax saturates to one-hot for large
    d_k → gradients vanish (softmax derivative -> 0).
"""

import math

import torch
import torch.nn.functional as F


def attention_forward(Q, K, V):
    """Returns (Y, P, S) with stable softmax."""
    d_k = Q.size(-1)
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    S_stable = S - S.max(dim=-1, keepdim=True).values  # numerical stability
    P = torch.softmax(S_stable, dim=-1)               # == softmax(S)
    Y = torch.matmul(P, V)
    return Y, P, S


def attention_backward(dLdY, Q, K, V, P):
    """Manual backward for attention given upstream gradient dLdY."""
    d_k = Q.size(-1)
    scale = 1.0 / math.sqrt(d_k)

    dLdV = torch.matmul(P.transpose(-2, -1), dLdY)
    dLdP = torch.matmul(dLdY, V.transpose(-2, -1))

    # softmax backward (the O(n) trick):
    # dL/dS_i = P_i * (dL/dP_i - sum_j P_j * dL/dP_j)
    dLdS = P * (dLdP - (dLdP * P).sum(dim=-1, keepdim=True))

    dLdQ = scale * torch.matmul(dLdS, K)
    dLdK = scale * torch.matmul(dLdS.transpose(-2, -1), Q)
    return dLdQ, dLdK, dLdV


def check_gradients():
    torch.manual_seed(42)
    B, H, T, D = 2, 3, 5, 8  # batch, heads, seq_len, d_k

    Q = torch.randn(B, H, T, D, dtype=torch.double)
    K = torch.randn(B, H, T, D, dtype=torch.double)
    V = torch.randn(B, H, T, D, dtype=torch.double)
    upstream = torch.randn_like(Q)

    # --- Reference gradients from autograd ---
    Qg, Kg, Vg = [x.clone().requires_grad_(True) for x in (Q, K, V)]
    Y_auto = F.scaled_dot_product_attention(Qg, Kg, Vg)
    Y_auto.backward(upstream)

    # --- Our manual gradients ---
    Y, P, _ = attention_forward(Q, K, V)
    dQ, dK, dV = attention_backward(upstream, Q, K, V, P)

    def rel_err(a, b):
        return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)

    print(f"{'tensor':>6} | {'autograd':>12} {'manual':>12} {'rel_err':>10}")
    for name, a, b in [("dQ", dQ, Qg.grad), ("dK", dK, Kg.grad), ("dV", dV, Vg.grad)]:
        print(f"{name:>6} | {b.abs().max().item():12.6f} {a.abs().max().item():12.6f} {rel_err(a, b):10.2e}")

    # --- Finite differences spot-check (per-element perturbation) ---
    # The loss must match upstream: dL/dY = upstream, so L = sum(Y * upstream)
    def loss_fn(q, k, v):
        return (attention_forward(q, k, v)[0] * upstream).sum()

    eps = 1e-6
    Q_flat = Q.view(-1)
    manual_flat = dQ.view(-1)
    idxs = torch.randint(0, Q_flat.numel(), (10,))
    errs = []
    for i in idxs:
        Qp, Qm = Q.clone(), Q.clone()
        Qp.view(-1)[i] += eps
        Qm.view(-1)[i] -= eps
        fd_i = (loss_fn(Qp, K, V) - loss_fn(Qm, K, V)) / (2 * eps)
        errs.append(abs(fd_i - manual_flat[i]).item() / (abs(fd_i) + 1e-12))
    print(f"finite-diff dQ rel_err (10 random elements): {max(errs):.2e}")

    assert rel_err(dQ, Qg.grad) < 1e-8, "dQ mismatch!"
    assert rel_err(dK, Kg.grad) < 1e-8, "dK mismatch!"
    assert rel_err(dV, Vg.grad) < 1e-8, "dV mismatch!"
    print("\nAll gradients match autograd within 1e-8.")


def softmax_saturation_demo():
    """Show why scaling by sqrt(d_k) matters."""
    n_keys = 50  # sequence length
    print(
        "\nAttention over %d keys: avg probability on the winning key "
        "(1.0 = one-hot = saturated, gradients ~ P(1-P) -> 0):" % n_keys
    )
    for d_k in [8, 64, 512, 4096]:
        # dot product of Q,K vectors with entries ~ N(0,1) has std ~ sqrt(d_k)
        dots = torch.randn(20000, n_keys) * math.sqrt(d_k)
        scaled = dots / math.sqrt(d_k)

        p_unscaled = F.softmax(dots, -1).max(-1).values.mean().item()
        p_scaled = F.softmax(scaled, -1).max(-1).values.mean().item()
        print(
            f"  d_k={d_k:5d} | unscaled max-P {p_unscaled:.3f} | "
            f"scaled max-P {p_scaled:.3f}"
        )


if __name__ == "__main__":
    check_gradients()
    softmax_saturation_demo()
