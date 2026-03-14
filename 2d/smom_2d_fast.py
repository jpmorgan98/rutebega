import numpy as np
import matplotlib.pyplot as plt

# =============================
# Numba acceleration
# =============================
from numba import njit

# -----------------------------
# 2D Discrete Ordinates (in-plane)
# -----------------------------
def make_circle_quadrature(N_dir: int):
    theta = 2.0 * np.pi * (np.arange(N_dir) + 0.5) / N_dir
    mu = np.cos(theta)
    eta = np.sin(theta)
    w = (2.0 * np.pi / N_dir) * np.ones(N_dir)
    omega_total = np.sum(w)  # = 2*pi
    return mu, eta, w, omega_total


def _bc_to_dir_arrays(bc, N_dir):
    """Convert bc dict into 4 float64 arrays (N_dir,) for numba kernels."""
    def as_dir_array(val):
        val = np.asarray(val, dtype=np.float64)
        if val.shape == ():
            out = np.empty(N_dir, dtype=np.float64)
            out[:] = float(val)
            return out
        if val.shape != (N_dir,):
            raise ValueError(f"BC must be scalar or shape (N_dir,), got {val.shape}")
        return np.ascontiguousarray(val, dtype=np.float64)

    bcL = as_dir_array(bc.get("left", 0.0))
    bcR = as_dir_array(bc.get("right", 0.0))
    bcB = as_dir_array(bc.get("bottom", 0.0))
    bcT = as_dir_array(bc.get("top", 0.0))
    return bcL, bcR, bcB, bcT


def _precompute_Ak_stream(mu, eta, dx, dy):
    """
    Precompute the corner-balance streaming+leakage operator (no sigma_t term),
    per direction, as 4x4 blocks. This is constant over the iteration loop.
    """
    N_dir = mu.shape[0]
    hx = dx / 2.0
    hy = dy / 2.0

    Ak = np.zeros((N_dir, 4, 4), dtype=np.float64)
    for k in range(N_dir):
        muk = float(mu[k])
        etak = float(eta[k])

        # Ax (2x2)
        Ax00 = hy * (abs(muk) / 2.0)
        Ax01 = hy * (muk / 2.0)
        Ax10 = hy * (-muk / 2.0)
        Ax11 = hy * (abs(muk) / 2.0)

        # Ay (2x2)
        Ay00 = hx * (abs(etak) / 2.0)
        Ay01 = hx * (etak / 2.0)
        Ay10 = hx * (-etak / 2.0)
        Ay11 = hx * (abs(etak) / 2.0)

        # A = kron(I2,Ax) + kron(Ay,I2)
        # kron(I2,Ax)
        # [Ax  0
        #  0   Ax]
        Ak[k, 0, 0] = Ax00
        Ak[k, 0, 1] = Ax01
        Ak[k, 1, 0] = Ax10
        Ak[k, 1, 1] = Ax11
        Ak[k, 2, 2] = Ax00
        Ak[k, 2, 3] = Ax01
        Ak[k, 3, 2] = Ax10
        Ak[k, 3, 3] = Ax11

        # + kron(Ay,I2)
        # [Ay00*I2  Ay01*I2
        #  Ay10*I2  Ay11*I2]
        Ak[k, 0, 0] += Ay00
        Ak[k, 1, 1] += Ay00
        Ak[k, 0, 2] += Ay01
        Ak[k, 1, 3] += Ay01
        Ak[k, 2, 0] += Ay10
        Ak[k, 3, 1] += Ay10
        Ak[k, 2, 2] += Ay11
        Ak[k, 3, 3] += Ay11

    return np.ascontiguousarray(Ak)


# -----------------------------
# Numba kernels: edges, sweep, norms
# -----------------------------
@njit(cache=True, fastmath=True)
def compute_edge_fluxes_inplace_nb(psi_cell, mu, eta, bcL, bcR, bcB, bcT, psi_xedge, psi_yedge):
    Nx, Ny, N_dir, _ = psi_cell.shape

    # Vertical edges (x-normal): shape (Nx+1, Ny, N_dir, 2)
    for i_edge in range(Nx + 1):
        for j in range(Ny):
            for k in range(N_dir):
                if i_edge == 0:
                    psi_xedge[i_edge, j, k, 0] = bcL[k]
                    psi_xedge[i_edge, j, k, 1] = bcL[k]
                elif i_edge == Nx:
                    psi_xedge[i_edge, j, k, 0] = bcR[k]
                    psi_xedge[i_edge, j, k, 1] = bcR[k]
                else:
                    if mu[k] > 0.0:
                        iL = i_edge - 1
                        psi_xedge[i_edge, j, k, 0] = psi_cell[iL, j, k, 1]  # SE
                        psi_xedge[i_edge, j, k, 1] = psi_cell[iL, j, k, 3]  # NE
                    else:
                        iR = i_edge
                        psi_xedge[i_edge, j, k, 0] = psi_cell[iR, j, k, 0]  # SW
                        psi_xedge[i_edge, j, k, 1] = psi_cell[iR, j, k, 2]  # NW

    # Horizontal edges (y-normal): shape (Nx, Ny+1, N_dir, 2)
    for i in range(Nx):
        for j_edge in range(Ny + 1):
            for k in range(N_dir):
                if j_edge == 0:
                    psi_yedge[i, j_edge, k, 0] = bcB[k]
                    psi_yedge[i, j_edge, k, 1] = bcB[k]
                elif j_edge == Ny:
                    psi_yedge[i, j_edge, k, 0] = bcT[k]
                    psi_yedge[i, j_edge, k, 1] = bcT[k]
                else:
                    if eta[k] > 0.0:
                        jB = j_edge - 1
                        psi_yedge[i, j_edge, k, 0] = psi_cell[i, jB, k, 2]  # NW
                        psi_yedge[i, j_edge, k, 1] = psi_cell[i, jB, k, 3]  # NE
                    else:
                        jT = j_edge
                        psi_yedge[i, j_edge, k, 0] = psi_cell[i, jT, k, 0]  # SW
                        psi_yedge[i, j_edge, k, 1] = psi_cell[i, jT, k, 1]  # SE


@njit(cache=True, fastmath=True)
def _build_cell_matrix_and_rhs_nb(
    A, b,  # preallocated
    i, j,
    dx, dy, area, hx, hy,
    sig_t_ij, sig_s_ij, q_ij,
    mu, eta, w, omega_total,
    Ak_stream,
    psi_xedge, psi_yedge
):
    """
    Fills A (n,n) and b (n) for cell (i,j).
    n = 4*N_dir
    """
    N_dir = w.shape[0]
    n = 4 * N_dir

    # Zero
    for r in range(n):
        b[r] = 0.0
        for c in range(n):
            A[r, c] = 0.0

    # Block-diagonal direction operators: Ak_stream + sigma_t * (area/4)*I4
    st_term = (sig_t_ij * area) / 4.0
    for k in range(N_dir):
        base = 4 * k
        for r4 in range(4):
            for c4 in range(4):
                A[base + r4, base + c4] = Ak_stream[k, r4, c4]
        A[base + 0, base + 0] += st_term
        A[base + 1, base + 1] += st_term
        A[base + 2, base + 2] += st_term
        A[base + 3, base + 3] += st_term

    # Scattering coupling (same structure as your build_cell_matrix)
    beta = (sig_s_ij * area) / (16.0 * omega_total)
    if beta != 0.0:
        for c in range(4):
            for cp in range(4):
                for k in range(N_dir):
                    row = 4 * k + c
                    for kp in range(N_dir):
                        col = 4 * kp + cp
                        A[row, col] -= beta * w[kp]

    # RHS: isotropic source + inflow edge terms
    q_iso = q_ij / omega_total
    base_src = (area / 4.0) * q_iso
    for k in range(N_dir):
        base = 4 * k
        b[base + 0] = base_src
        b[base + 1] = base_src
        b[base + 2] = base_src
        b[base + 3] = base_src

        muk = mu[k]
        etak = eta[k]

        if muk > 0.0:
            inflow_bot = psi_xedge[i, j, k, 0]
            inflow_top = psi_xedge[i, j, k, 1]
            b[base + 0] += muk * hy * inflow_bot
            b[base + 2] += muk * hy * inflow_top
        elif muk < 0.0:
            inflow_bot = psi_xedge[i + 1, j, k, 0]
            inflow_top = psi_xedge[i + 1, j, k, 1]
            b[base + 1] += (-muk) * hy * inflow_bot
            b[base + 3] += (-muk) * hy * inflow_top

        if etak > 0.0:
            inflow_left = psi_yedge[i, j, k, 0]
            inflow_right = psi_yedge[i, j, k, 1]
            b[base + 0] += etak * hx * inflow_left
            b[base + 1] += etak * hx * inflow_right
        elif etak < 0.0:
            inflow_left = psi_yedge[i, j + 1, k, 0]
            inflow_right = psi_yedge[i, j + 1, k, 1]
            b[base + 2] += (-etak) * hx * inflow_left
            b[base + 3] += (-etak) * hx * inflow_right


@njit(cache=True, fastmath=True)
def transport_sweep_nb(
    psi,
    sig_t, sig_s, q_cell,
    dx, dy,
    mu, eta, w, omega_total,
    Ak_stream,
    psi_xedge, psi_yedge
):
    Nx, Ny, N_dir, _ = psi.shape
    n = 4 * N_dir
    area = dx * dy
    hx = dx / 2.0
    hy = dy / 2.0

    A = np.empty((n, n), dtype=np.float64)
    b = np.empty(n, dtype=np.float64)

    for i in range(Nx):
        for j in range(Ny):
            _build_cell_matrix_and_rhs_nb(
                A, b,
                i, j,
                dx, dy, area, hx, hy,
                sig_t[i, j], sig_s[i, j], q_cell[i, j],
                mu, eta, w, omega_total,
                Ak_stream,
                psi_xedge, psi_yedge
            )
            xloc = np.linalg.solve(A, b)  # small dense solve (n=4*N_dir)

            # write back to psi[i,j,:,:]
            for k in range(N_dir):
                base = 4 * k
                psi[i, j, k, 0] = xloc[base + 0]
                psi[i, j, k, 1] = xloc[base + 1]
                psi[i, j, k, 2] = xloc[base + 2]
                psi[i, j, k, 3] = xloc[base + 3]


@njit(cache=True, fastmath=True)
def l2_norm_diff_nb(a, b):
    s = 0.0
    for idx in range(a.size):
        d = a.flat[idx] - b.flat[idx]
        s += d * d
    return np.sqrt(s)


# ============================================================
# Second-moment acceleration (2D in-plane) - Numba
# ============================================================
@njit(cache=True, fastmath=True)
def compute_incident_partial_currents_2d_inplane_nb(bcL, bcR, bcB, bcT, mu, eta, w):
    N_dir = w.shape[0]
    Jm_left = 0.0
    Jm_right = 0.0
    Jm_bottom = 0.0
    Jm_top = 0.0

    for k in range(N_dir):
        wk = w[k]
        muk = mu[k]
        etak = eta[k]

        # Left: outward n=(-1,0), inflow muk>0
        if muk > 0.0:
            Jm_left += wk * (-muk) * bcL[k]

        # Right: outward n=(+1,0), inflow muk<0 (muk is negative)
        if muk < 0.0:
            Jm_right += wk * (muk) * bcR[k]

        # Bottom: outward n=(0,-1), inflow etak>0
        if etak > 0.0:
            Jm_bottom += wk * (-etak) * bcB[k]

        # Top: outward n=(0,+1), inflow etak<0 (etak is negative)
        if etak < 0.0:
            Jm_top += wk * (etak) * bcT[k]

    return Jm_left, Jm_right, Jm_bottom, Jm_top


@njit(cache=True, fastmath=True)
def compute_cell_moments_nb(psi, mu, eta, w, phi, Jx, Jy):
    Nx, Ny, N_dir, _ = psi.shape
    for i in range(Nx):
        for j in range(Ny):
            ph = 0.0
            jx = 0.0
            jy = 0.0
            for k in range(N_dir):
                avg = 0.25 * (psi[i, j, k, 0] + psi[i, j, k, 1] + psi[i, j, k, 2] + psi[i, j, k, 3])
                wk = w[k]
                ph += wk * avg
                jx += wk * mu[k] * avg
                jy += wk * eta[k] * avg
            phi[i, j] = ph
            Jx[i, j] = jx
            Jy[i, j] = jy


@njit(cache=True, fastmath=True)
def compute_cell_moments_inplane_nb(psi, mu, eta, w, phi, Qxx, Qxy):
    Nx, Ny, N_dir, _ = psi.shape
    for i in range(Nx):
        for j in range(Ny):
            ph = 0.0
            qxx = 0.0
            qxy = 0.0
            for k in range(N_dir):
                avg = 0.25 * (psi[i, j, k, 0] + psi[i, j, k, 1] + psi[i, j, k, 2] + psi[i, j, k, 3])
                wk = w[k]
                muk = mu[k]
                etak = eta[k]
                ph += wk * avg
                qxx += wk * (muk * muk - 0.5) * avg
                qxy += wk * (muk * etak) * avg
            phi[i, j] = ph
            Qxx[i, j] = qxx
            Qxy[i, j] = qxy


@njit(cache=True, fastmath=True)
def div_Q_vector_inplane_nb(Qxx, Qxy, dx, dy, Ux, Uy):
    """
    Neumann padding via edge replication, then centered differences.
    """
    Nx, Ny = Qxx.shape
    inv2dx = 1.0 / (2.0 * dx)
    inv2dy = 1.0 / (2.0 * dy)

    for i in range(Nx):
        im1 = i - 1 if i > 0 else 0
        ip1 = i + 1 if i < Nx - 1 else Nx - 1
        for j in range(Ny):
            jm1 = j - 1 if j > 0 else 0
            jp1 = j + 1 if j < Ny - 1 else Ny - 1

            dQxx_dx = (Qxx[ip1, j] - Qxx[im1, j]) * inv2dx
            dQxx_dy = (Qxx[i, jp1] - Qxx[i, jm1]) * inv2dy

            dQxy_dx = (Qxy[ip1, j] - Qxy[im1, j]) * inv2dx
            dQxy_dy = (Qxy[i, jp1] - Qxy[i, jm1]) * inv2dy

            Ux[i, j] = dQxx_dx + dQxy_dy
            Uy[i, j] = dQxy_dx - dQxx_dy


@njit(cache=True, fastmath=True)
def div_div_Q_inplane_nb(Qxx, Qxy, dx, dy, ddQ):
    Nx, Ny = Qxx.shape
    Ux = np.empty((Nx, Ny), dtype=np.float64)
    Uy = np.empty((Nx, Ny), dtype=np.float64)
    div_Q_vector_inplane_nb(Qxx, Qxy, dx, dy, Ux, Uy)

    inv2dx = 1.0 / (2.0 * dx)
    inv2dy = 1.0 / (2.0 * dy)

    for i in range(Nx):
        im1 = i - 1 if i > 0 else 0
        ip1 = i + 1 if i < Nx - 1 else Nx - 1
        for j in range(Ny):
            jm1 = j - 1 if j > 0 else 0
            jp1 = j + 1 if j < Ny - 1 else Ny - 1

            dUx_dx = (Ux[ip1, j] - Ux[im1, j]) * inv2dx
            dUy_dy = (Uy[i, jp1] - Uy[i, jm1]) * inv2dy
            ddQ[i, j] = dUx_dx + dUy_dy


@njit(cache=True, fastmath=True)
def grad_center_nb(phi, dx, dy, gx, gy):
    Nx, Ny = phi.shape
    inv2dx = 1.0 / (2.0 * dx)
    inv2dy = 1.0 / (2.0 * dy)
    invdx = 1.0 / dx
    invdy = 1.0 / dy

    for j in range(Ny):
        gx[0, j] = (phi[1, j] - phi[0, j]) * invdx
        gx[Nx - 1, j] = (phi[Nx - 1, j] - phi[Nx - 2, j]) * invdx
    for i in range(1, Nx - 1):
        for j in range(Ny):
            gx[i, j] = (phi[i + 1, j] - phi[i - 1, j]) * inv2dx

    for i in range(Nx):
        gy[i, 0] = (phi[i, 1] - phi[i, 0]) * invdy
        gy[i, Ny - 1] = (phi[i, Ny - 1] - phi[i, Ny - 2]) * invdy
    for i in range(Nx):
        for j in range(1, Ny - 1):
            gy[i, j] = (phi[i, j + 1] - phi[i, j - 1]) * inv2dy


# ---- Diffusion solve (CG) with Robin BCs, matrix-free ----
@njit(cache=True, fastmath=True)
def _harmonic(a, b):
    den = a + b
    if den > 0.0:
        return 2.0 * a * b / den
    return 0.0


@njit(cache=True, fastmath=True)
def _robin_ghost_coeff(Df, alpha, g, h):
    # Df*(phi_g - phi_c)/h + alpha*(phi_c + phi_g)/2 = g
    denom = (Df / h + alpha / 2.0)
    a = (Df / h - alpha / 2.0) / denom
    c = g / denom
    return a, c


@njit(cache=True, fastmath=True)
def diffusion_build_b_nb(phi_rhs, sig_a, Dcell, dx, dy, alphaL, gL, alphaR, gR, alphaB, gB, alphaT, gT, b):
    Nx, Ny = phi_rhs.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    for i in range(Nx):
        for j in range(Ny):
            b[i, j] = phi_rhs[i, j]

    for i in range(Nx):
        for j in range(Ny):
            Dc = Dcell[i, j]

            # West/East faces
            Dw = Dc if i == 0 else _harmonic(Dc, Dcell[i - 1, j])
            De = Dc if i == Nx - 1 else _harmonic(Dc, Dcell[i + 1, j])

            # South/North faces
            Ds = Dc if j == 0 else _harmonic(Dc, Dcell[i, j - 1])
            Dn = Dc if j == Ny - 1 else _harmonic(Dc, Dcell[i, j + 1])

            # West boundary contribution to RHS
            if i == 0:
                a, c = _robin_ghost_coeff(Dw, alphaL, gL[j], dx)
                b[i, j] += (Dw * inv_dx2) * c

            # East
            if i == Nx - 1:
                a, c = _robin_ghost_coeff(De, alphaR, gR[j], dx)
                b[i, j] += (De * inv_dx2) * c

            # South
            if j == 0:
                a, c = _robin_ghost_coeff(Ds, alphaB, gB[i], dy)
                b[i, j] += (Ds * inv_dy2) * c

            # North
            if j == Ny - 1:
                a, c = _robin_ghost_coeff(Dn, alphaT, gT[i], dy)
                b[i, j] += (Dn * inv_dy2) * c


@njit(cache=True, fastmath=True)
def diffusion_apply_A_nb(phi, sig_a, Dcell, dx, dy, alphaL, gL, alphaR, gR, alphaB, gB, alphaT, gT, out):
    """
    Apply A(phi) where A is the same operator you assembled in solve_diffusion,
    including Robin ghost elimination (a affects diagonal; c is handled in RHS build).
    """
    Nx, Ny = phi.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    for i in range(Nx):
        for j in range(Ny):
            out[i, j] = 0.0

    for i in range(Nx):
        for j in range(Ny):
            Dc = Dcell[i, j]
            sa = sig_a[i, j]

            Dw = Dc if i == 0 else _harmonic(Dc, Dcell[i - 1, j])
            De = Dc if i == Nx - 1 else _harmonic(Dc, Dcell[i + 1, j])
            Ds = Dc if j == 0 else _harmonic(Dc, Dcell[i, j - 1])
            Dn = Dc if j == Ny - 1 else _harmonic(Dc, Dcell[i, j + 1])

            diag = sa + (Dw + De) * inv_dx2 + (Ds + Dn) * inv_dy2

            # West
            if i > 0:
                out[i, j] += (-Dw * inv_dx2) * phi[i - 1, j]
            else:
                a, c = _robin_ghost_coeff(Dw, alphaL, gL[j], dx)
                diag += (-Dw * inv_dx2) * a

            # East
            if i < Nx - 1:
                out[i, j] += (-De * inv_dx2) * phi[i + 1, j]
            else:
                a, c = _robin_ghost_coeff(De, alphaR, gR[j], dx)
                diag += (-De * inv_dx2) * a

            # South
            if j > 0:
                out[i, j] += (-Ds * inv_dy2) * phi[i, j - 1]
            else:
                a, c = _robin_ghost_coeff(Ds, alphaB, gB[i], dy)
                diag += (-Ds * inv_dy2) * a

            # North
            if j < Ny - 1:
                out[i, j] += (-Dn * inv_dy2) * phi[i, j + 1]
            else:
                a, c = _robin_ghost_coeff(Dn, alphaT, gT[i], dy)
                diag += (-Dn * inv_dy2) * a

            out[i, j] += diag * phi[i, j]


@njit(cache=True, fastmath=True)
def _dot2(a, b):
    s = 0.0
    for idx in range(a.size):
        s += a.flat[idx] * b.flat[idx]
    return s


@njit(cache=True, fastmath=True)
def diffusion_cg_solve_nb(
    phi_rhs, sig_a, Dcell, dx, dy,
    alphaL, gL, alphaR, gR, alphaB, gB, alphaT, gT,
    phi_out,
    maxit, tol_rel
):
    Nx, Ny = phi_rhs.shape
    b = np.empty((Nx, Ny), dtype=np.float64)
    diffusion_build_b_nb(phi_rhs, sig_a, Dcell, dx, dy, alphaL, gL, alphaR, gR, alphaB, gB, alphaT, gT, b)

    x = phi_out
    for idx in range(x.size):
        x.flat[idx] = 0.0

    r = b.copy()
    p = r.copy()
    Ap = np.empty((Nx, Ny), dtype=np.float64)

    bnorm2 = _dot2(b, b)
    if bnorm2 <= 0.0:
        return

    rsold = _dot2(r, r)
    tol2 = (tol_rel * tol_rel) * bnorm2

    for _it in range(maxit):
        diffusion_apply_A_nb(p, sig_a, Dcell, dx, dy, alphaL, gL, alphaR, gR, alphaB, gB, alphaT, gT, Ap)
        denom = _dot2(p, Ap)
        if denom == 0.0:
            break
        alpha = rsold / denom

        # x += alpha*p ; r -= alpha*Ap
        for idx in range(x.size):
            x.flat[idx] += alpha * p.flat[idx]
            r.flat[idx] -= alpha * Ap.flat[idx]

        rsnew = _dot2(r, r)
        if rsnew < tol2:
            break

        beta = rsnew / rsold
        for idx in range(p.size):
            p.flat[idx] = r.flat[idx] + beta * p.flat[idx]
        rsold = rsnew


@njit(cache=True, fastmath=True)
def yavuz_update_edges_2d_nb(
    psi_xedge, psi_yedge,
    mu, eta,
    bcL, bcR, bcB, bcT,
    phi_half, Jx_half, Jy_half,
    phi_acc,  Jx_acc,  Jy_acc
):
    Nx_edges, Ny, N_dir, _ = psi_xedge.shape
    Nx = Nx_edges - 1
    Ny_edges = psi_yedge.shape[1]

    # Interior vertical edges
    for i_edge in range(1, Nx):
        for j in range(Ny):
            dphi_e = 0.5 * ((phi_acc[i_edge - 1, j] - phi_half[i_edge - 1, j]) +
                            (phi_acc[i_edge, j]     - phi_half[i_edge, j]))
            dJx_e  = 0.5 * ((Jx_acc[i_edge - 1, j] - Jx_half[i_edge - 1, j]) +
                            (Jx_acc[i_edge, j]     - Jx_half[i_edge, j]))
            dJy_e  = 0.5 * ((Jy_acc[i_edge - 1, j] - Jy_half[i_edge - 1, j]) +
                            (Jy_acc[i_edge, j]     - Jy_half[i_edge, j]))

            for k in range(N_dir):
                corr = (dphi_e / (2.0 * np.pi)) + ((mu[k] * dJx_e + eta[k] * dJy_e) / np.pi)
                psi_xedge[i_edge, j, k, 0] += corr
                psi_xedge[i_edge, j, k, 1] += corr

    # Interior horizontal edges
    for i in range(Nx):
        for j_edge in range(1, Ny_edges - 1):
            dphi_e = 0.5 * ((phi_acc[i, j_edge - 1] - phi_half[i, j_edge - 1]) +
                            (phi_acc[i, j_edge]     - phi_half[i, j_edge]))
            dJx_e  = 0.5 * ((Jx_acc[i, j_edge - 1] - Jx_half[i, j_edge - 1]) +
                            (Jx_acc[i, j_edge]     - Jx_half[i, j_edge]))
            dJy_e  = 0.5 * ((Jy_acc[i, j_edge - 1] - Jy_half[i, j_edge - 1]) +
                            (Jy_acc[i, j_edge]     - Jy_half[i, j_edge]))

            for k in range(N_dir):
                corr = (dphi_e / (2.0 * np.pi)) + ((mu[k] * dJx_e + eta[k] * dJy_e) / np.pi)
                psi_yedge[i, j_edge, k, 0] += corr
                psi_yedge[i, j_edge, k, 1] += corr

    # Re-enforce boundary values (like your python version)
    for j in range(Ny):
        for k in range(N_dir):
            psi_xedge[0, j, k, 0] = bcL[k]
            psi_xedge[0, j, k, 1] = bcL[k]
            psi_xedge[Nx, j, k, 0] = bcR[k]
            psi_xedge[Nx, j, k, 1] = bcR[k]

    for i in range(Nx):
        for k in range(N_dir):
            psi_yedge[i, 0, k, 0] = bcB[k]
            psi_yedge[i, 0, k, 1] = bcB[k]
            psi_yedge[i, Ny_edges - 1, k, 0] = bcT[k]
            psi_yedge[i, Ny_edges - 1, k, 1] = bcT[k]

    # Prevent negative edge fluxes
    for idx in range(psi_xedge.size):
        if psi_xedge.flat[idx] < 0.0:
            psi_xedge.flat[idx] = 0.0
    for idx in range(psi_yedge.size):
        if psi_yedge.flat[idx] < 0.0:
            psi_yedge.flat[idx] = 0.0


@njit(cache=True, fastmath=True)
def smm_nb(
    psi, psi_xedge, psi_yedge,
    q_cell, sig_t, sig_s, dx, dy,
    mu, eta, w,
    bcL, bcR, bcB, bcT,
    update_mode,  # 0=rescale, 1=yavuz
    phi_half_out, phi_acc_out
):
    Nx, Ny, N_dir, _ = psi.shape

    # Half moments (in-plane second moment)
    Qxx = np.empty((Nx, Ny), dtype=np.float64)
    Qxy = np.empty((Nx, Ny), dtype=np.float64)
    compute_cell_moments_inplane_nb(psi, mu, eta, w, phi_half_out, Qxx, Qxy)

    sig_a = sig_t - sig_s

    ddQ = np.empty((Nx, Ny), dtype=np.float64)
    div_div_Q_inplane_nb(Qxx, Qxy, dx, dy, ddQ)

    # s_sm = ddQ / sig_t
    rhs = np.empty((Nx, Ny), dtype=np.float64)
    for i in range(Nx):
        for j in range(Ny):
            rhs[i, j] = q_cell[i, j] + ddQ[i, j] / (sig_t[i, j] + 1e-300)

    # Ux, Uy
    Ux = np.empty((Nx, Ny), dtype=np.float64)
    Uy = np.empty((Nx, Ny), dtype=np.float64)
    div_Q_vector_inplane_nb(Qxx, Qxy, dx, dy, Ux, Uy)

    # Incident partial currents (scalars)
    Jm_left, Jm_right, Jm_bottom, Jm_top = compute_incident_partial_currents_2d_inplane_nb(
        bcL, bcR, bcB, bcT, mu, eta, w
    )

    omega_total = 0.0
    for k in range(N_dir):
        omega_total += w[k]

    M1x = 0.0
    M1y = 0.0
    for k in range(N_dir):
        if mu[k] > 0.0:
            M1x += w[k] * mu[k]
        if eta[k] > 0.0:
            M1y += w[k] * eta[k]

    alpha_x = 2.0 * M1x / omega_total
    alpha_y = 2.0 * M1y / omega_total

    # Robin g arrays
    g_left = np.empty(Ny, dtype=np.float64)
    g_right = np.empty(Ny, dtype=np.float64)
    g_bottom = np.empty(Nx, dtype=np.float64)
    g_top = np.empty(Nx, dtype=np.float64)

    for j in range(Ny):
        g_left[j] = (-2.0 * Jm_left) + (Ux[0, j] / (sig_t[0, j] + 1e-300))
        g_right[j] = (-2.0 * Jm_right) - (Ux[Nx - 1, j] / (sig_t[Nx - 1, j] + 1e-300))

    for i in range(Nx):
        g_bottom[i] = (-2.0 * Jm_bottom) + (Uy[i, 0] / (sig_t[i, 0] + 1e-300))
        g_top[i] = (-2.0 * Jm_top) - (Uy[i, Ny - 1] / (sig_t[i, Ny - 1] + 1e-300))

    # D0 = 1/(2*sig_t)
    D0 = np.empty((Nx, Ny), dtype=np.float64)
    for i in range(Nx):
        for j in range(Ny):
            D0[i, j] = 1.0 / (2.0 * (sig_t[i, j] + 1e-300))

    # Diffusion solve (CG)
    diffusion_cg_solve_nb(
        rhs, sig_a, D0, dx, dy,
        alpha_x, g_left,
        alpha_x, g_right,
        alpha_y, g_bottom,
        alpha_y, g_top,
        phi_acc_out,
        maxit=2000, tol_rel=1e-10
    )

    if update_mode == 0:
        # rescale
        eps = 1e-14
        for i in range(Nx):
            for j in range(Ny):
                scale = phi_acc_out[i, j] / (phi_half_out[i, j] + eps)
                if scale < 0.0:
                    scale = 0.0
                for k in range(N_dir):
                    psi[i, j, k, 0] *= scale
                    psi[i, j, k, 1] *= scale
                    psi[i, j, k, 2] *= scale
                    psi[i, j, k, 3] *= scale

        # Recompute edges after acceleration
        compute_edge_fluxes_inplace_nb(psi, mu, eta, bcL, bcR, bcB, bcT, psi_xedge, psi_yedge)

    else:
        # yavuz update on edges
        phi_half2 = np.empty((Nx, Ny), dtype=np.float64)
        Jx_half = np.empty((Nx, Ny), dtype=np.float64)
        Jy_half = np.empty((Nx, Ny), dtype=np.float64)
        compute_cell_moments_nb(psi, mu, eta, w, phi_half2, Jx_half, Jy_half)

        gx = np.empty((Nx, Ny), dtype=np.float64)
        gy = np.empty((Nx, Ny), dtype=np.float64)
        grad_center_nb(phi_acc_out, dx, dy, gx, gy)

        Jx_acc = np.empty((Nx, Ny), dtype=np.float64)
        Jy_acc = np.empty((Nx, Ny), dtype=np.float64)
        for i in range(Nx):
            for j in range(Ny):
                st = sig_t[i, j] + 1e-300
                Jx_acc[i, j] = -(0.5 * gx[i, j] + Ux[i, j]) / st
                Jy_acc[i, j] = -(0.5 * gy[i, j] + Uy[i, j]) / st

        yavuz_update_edges_2d_nb(
            psi_xedge, psi_yedge,
            mu, eta,
            bcL, bcR, bcB, bcT,
            phi_half2, Jx_half, Jy_half,
            phi_acc_out, Jx_acc, Jy_acc
        )


# -----------------------------
# OCI transport solve (2D) + SMA
# Signature kept IDENTICAL
# -----------------------------
def transport_2d_oci(
    Nx=30, Ny=30,
    Lx=3.0, Ly=3.0,
    N_dir=8,
    sig_t=None, sig_s=None, q=None,
    bc=None,
    tol=1e-4, max_it=2000, printer=True,
    smm_acc=True, update="rescale",
    sig_t_val=1.0, sig_s_val=0.0, q_val=1.0,
):
    if bc is None:
        bc = dict(left=0.0, right=0.0, bottom=0.0, top=0.0)

    dx = Lx / Nx
    dy = Ly / Ny

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Material/source fields
    if sig_t is None:
        sig_t = sig_t_val * np.ones((Nx, Ny), dtype=np.float64)
    else:
        sig_t = np.asarray(sig_t, dtype=np.float64)

    if sig_s is None:
        if np.any(sig_s_val > sig_t):
            raise ValueError("sig_s_val must be <= sig_t everywhere.")
        sig_s = sig_s_val * np.ones((Nx, Ny), dtype=np.float64)
    else:
        sig_s = np.asarray(sig_s, dtype=np.float64)

    if q is None:
        q_cell = q_val * np.ones((Nx, Ny), dtype=np.float64)
    else:
        q_cell = np.asarray(q, dtype=np.float64)

    # Quadrature
    mu, eta, w, omega_total = make_circle_quadrature(N_dir)
    mu = np.ascontiguousarray(mu, dtype=np.float64)
    eta = np.ascontiguousarray(eta, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)

    # BC arrays for numba
    bcL, bcR, bcB, bcT = _bc_to_dir_arrays(bc, N_dir)

    # Precompute per-direction streaming operator blocks (constant)
    Ak_stream = _precompute_Ak_stream(mu, eta, dx, dy)

    psi = np.zeros((Nx, Ny, N_dir, 4), dtype=np.float64)
    psi_last = psi.copy()

    # Edge arrays (allocated once, filled in-place)
    psi_xedge = np.zeros((Nx + 1, Ny, N_dir, 2), dtype=np.float64)
    psi_yedge = np.zeros((Nx, Ny + 1, N_dir, 2), dtype=np.float64)

    compute_edge_fluxes_inplace_nb(psi, mu, eta, bcL, bcR, bcB, bcT, psi_xedge, psi_yedge)

    # Update mode normalization (case-insensitive)
    upd = (update or "").strip().lower()
    update_mode = 0 if upd == "rescale" else 1  # treat anything else as "yavuz"

    # Decide whether to run SMM (respect old sig_s_val behavior, but also handle array sig_s)
    do_smm = smm_acc and (sig_s_val > 0.0 or np.any(sig_s > 0.0))

    # Preallocate moment arrays for SMM outputs (reused)
    phi_half = np.empty((Nx, Ny), dtype=np.float64)
    phi_acc = np.empty((Nx, Ny), dtype=np.float64)

    err_last = 1.0
    rho = np.nan
    it = 0

    for it in range(max_it):
        # --- transport half-step (OCI) ---
        transport_sweep_nb(
            psi,
            sig_t, sig_s, q_cell,
            dx, dy,
            mu, eta, w, omega_total,
            Ak_stream,
            psi_xedge, psi_yedge
        )

        # Update edges by upwind closure
        compute_edge_fluxes_inplace_nb(psi, mu, eta, bcL, bcR, bcB, bcT, psi_xedge, psi_yedge)

        # --- second-moment acceleration step (optional) ---
        if do_smm:
            smm_nb(
                psi, psi_xedge, psi_yedge,
                q_cell, sig_t, sig_s, dx, dy,
                mu, eta, w,
                bcL, bcR, bcB, bcT,
                update_mode,
                phi_half, phi_acc
            )

        # Convergence check
        err = l2_norm_diff_nb(psi, psi_last)
        rho = err / err_last if it > 0 else np.nan

        if printer:
            print(f"it {it:4d}  err {err:.3e}  ρ {rho:.5f}")

        if it > 2:
            rclip = min(max(rho if np.isfinite(rho) else 0.0, 0.0), 0.999999)
            if err < tol * (1.0 - rclip):
                break

        psi_last[:] = psi
        err_last = max(err, 1e-300)

    # Scalar flux (cell-average)
    psi_avg = np.mean(psi, axis=3)                 # (Nx,Ny,N_dir)
    phi = np.tensordot(psi_avg, w, axes=([2], [0])) # (Nx,Ny)

    angles = dict(mu=mu, eta=eta, w=w)
    mesh = dict(x=x, y=y, X=X, Y=Y, dx=dx, dy=dy)

    return phi, psi, angles, mesh, rho, it


# -----------------------------
# Demo (unchanged)
# -----------------------------
if __name__ == "__main__":
    q = 2.0
    sigma = 5.0
    c = 0.7
    sigma_s = sigma * c
    inf_homo = q / (sigma * (1 - c))
    inf_homo /= (2 * np.pi)

    bc = dict(left=inf_homo, right=inf_homo, bottom=inf_homo, top=inf_homo)

    phi, psi, ang, mesh, rho, it = transport_2d_oci(
        Nx=100, Ny=10, Lx=1.0, Ly=1.0,
        N_dir=8,
        sig_t_val=sigma, sig_s_val=sigma_s,
        q_val=q,
        bc=bc,
        tol=1e-6, max_it=500, printer=True,
        smm_acc=True, update="Yavuz"
    )

    x, y, X, Y = mesh["x"], mesh["y"], mesh["X"], mesh["Y"]
    plt.figure()
    plt.imshow(phi.T, origin="lower",
               extent=[x.min(), x.max(), y.min(), y.max()],
               aspect="auto")
    plt.colorbar(label=r"$\phi(x,y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Corner-Balance Sn (OCI) + Second-Moment Acceleration: Scalar Flux")
    plt.tight_layout()
    plt.show()

    print("phi min/max:", phi.min(), phi.max())
