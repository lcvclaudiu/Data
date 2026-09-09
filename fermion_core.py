import math
import numpy as np
from scipy.linalg import eigh
from stitch_fss import build_B2_sparse, omega_s2


def gram_dense(L, typ):
    B = build_B2_sparse(L, typ)
    return (B.T @ B).toarray()


def diagonalize_state(L, typ):
    G = gram_dense(L, typ)
    lam, Q = eigh(G, check_finite=False, driver="evd")
    return G, lam, Q


def omega_from_lam(lam, T, mu):
    return float(omega_s2(np.sqrt(np.clip(lam, 0.0, None)), T, mu))


def delta_decomp(D, tol=1e-11):
    mask = np.any(np.abs(D) > tol, axis=0) | np.any(np.abs(D) > tol, axis=1)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros((D.shape[0], 0)), np.zeros(0)
    Ds = D[np.ix_(idx, idx)]
    w, v = eigh(Ds, check_finite=False, driver="evd")
    keep = np.abs(w) > tol
    U = np.zeros((D.shape[0], int(keep.sum())))
    U[idx, :] = v[:, keep]
    return U, w[keep]


def delta_omega_matsubara(lam, Q, G, Gp, T, mu, nfreq=128):
    """Exact-target low-rank Matsubara free-energy change.

    nfreq controls only truncation of the Matsubara sum. 128 was previously
    validated for the present STITCH parameter window to ~1e-6 J per move.
    Revalidate on your own machine/configuration with validate_updates.py.
    """
    U, c = delta_decomp(Gp - G)
    if len(c) == 0:
        return 0.0
    W = Q.T @ U
    total = 0.0
    for n in range(nfreq):
        om = (2 * n + 1) * np.pi * T
        z = (om + 1j * mu) ** 2
        R = (W.T * (1.0 / (lam + z))) @ W
        M = np.eye(len(c), dtype=complex) + c[:, None] * R
        sign, logabs = np.linalg.slogdet(M)
        total += np.real(np.log(sign) + logabs)
    return -2.0 * T * total


def flip_proposal(typ, i, j):
    prop = typ.copy()
    prop[i, j] ^= 1
    return prop


def m_from_nh(nh, N):
    return 2.0 * nh / N - 1.0
