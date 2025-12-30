import numpy as np
import numba as nb
import matplotlib.pyplot as plt

# ============================================================
# Quadrature (can stay in python; called once)
# ============================================================
def make_circle_quadrature(N_dir: int):
    theta = 2.0 * np.pi * (np.arange(N_dir) + 0.5) / N_dir
    mu = np.cos(theta)
    eta = np.sin(theta)
    w = (2.0 * np.pi / N_dir) * np.ones(N_dir)
    omega_total = np.sum(w)  # = 2*pi
    return mu, eta, w, omega_total


# ============================================================
# Numba helpers
# ============================================================

@nb.njit(cache=True, fastmath=True)
def _as_dir_array(val, N_dir):
    """Convert scalar or (N_dir,) array into (N_dir,) array (Numba-friendly)."""
    out = np.empty(N_dir, dtype=np.float64)
    if np.ndim(val) == 0:
        for k in range(N_dir):
            out[k] = float(val)
    else:
        # assume correct length
        for k in range(N_dir):
            out[k] = val[k]
    return out


@nb.njit(cache=True, fastmath=True)
def precompute_stream_mats(mu, eta, dx, dy):
    """
    Precompute per-direction 4x4 streaming matrix:
      S_k = kron(I2,Ax) + kron(Ay,I2)
    (no absorption term; that gets added per-cell via sig_t).
    """
    N_dir = mu.shape[0]
    S = np.zeros((N_dir, 4, 4), dtype=np.float64)

    hx = dx * 0.5
    hy = dy * 0.5

    for k in range(N_dir):
        muk = mu[k]
        etak = eta[k]

        # Ax = hy * [[|mu|/2, mu/2],[-mu/2, |mu|/2]]
        a = hy * (abs(muk) * 0.5)
        b = hy * (muk * 0.5)
        c = hy * (-muk * 0.5)
        d = hy * (abs(muk) * 0.5)

        # Ay = hx * [[|eta|/2, eta/2],[-eta/2, |eta|/2]]
        e = hx * (abs(etak) * 0.5)
        f = hx * (etak * 0.5)
        g = hx * (-etak * 0.5)
        h = hx * (abs(etak) * 0.5)

        # kron(I2,Ax) + kron(Ay,I2) in the explicit 4x4 form:
        # [[a+e, b,   f,   0],
        #  [c,   d+e, 0,   f],
        #  [g,   0,   a+h, b],
        #  [0,   g,   c,   d+h]]
        S[k, 0, 0] = a + e
        S[k, 0, 1] = b
        S[k, 0, 2] = f
        S[k, 0, 3] = 0.0

        S[k, 1, 0] = c
        S[k, 1, 1] = d + e
        S[k, 1, 2] = 0.0
        S[k, 1, 3] = f

        S[k, 2, 0] = g
        S[k, 2, 1] = 0.0
        S[k, 2, 2] = a + h
        S[k, 2, 3] = b

        S[k, 3, 0] = 0.0
        S[k, 3, 1] = g
        S[k, 3, 2] = c
        S[k, 3, 3] = d + h

    return S


@nb.njit(cache=True, fastmath=True)
def build_local_A_b(
    i, j,
    dx, dy,
    q_ij,
    sig_t_ij, sig_s_ij,
    mu, eta, w, omega_total,
    S_stream,              # (N_dir,4,4)
    psi_xedge, psi_yedge,  # edge arrays
    A, b                   # preallocated (nloc,nloc), (nloc,)
):
    """
    Fill A and b for cell (i,j).
    """
    N_dir = w.shape[0]
    nloc = 4 * N_dir

    # zero A,b
    for r in range(nloc):
        b[r] = 0.0
        for c in range(nloc):
            A[r, c] = 0.0

    area = dx * dy
    hy = dy * 0.5
    hx = dx * 0.5

    # block diagonal Ak = S_stream[k] + (sig_t*area/4) I4
    add_abs = (sig_t_ij * area * 0.25)
    for k in range(N_dir):
        base = 4 * k
        # copy 4x4
        for r4 in range(4):
            for c4 in range(4):
                A[base + r4, base + c4] = S_stream[k, r4, c4]
        # add absorption diag
        A[base + 0, base + 0] += add_abs
        A[base + 1, base + 1] += add_abs
        A[base + 2, base + 2] += add_abs
        A[base + 3, base + 3] += add_abs

    # scattering coupling term
    beta = sig_s_ij * area / (16.0 * omega_total)
    if beta != 0.0:
        for ccorner in range(4):
            for k in range(N_dir):
                r = 4 * k + ccorner
                for cpcorner in range(4):
                    for kp in range(N_dir):
                        col = 4 * kp + cpcorner
                        A[r, col] -= beta * w[kp]

    # RHS b
    q_iso = q_ij / omega_total
    base_src = (area * 0.25) * q_iso

    for k in range(N_dir):
        muk = mu[k]
        etak = eta[k]
        base = 4 * k

        b[base + 0] = base_src
        b[base + 1] = base_src
        b[base + 2] = base_src
        b[base + 3] = base_src

        # x-inflow
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

        # y-inflow
        if etak > 0.0:
            inflow_left  = psi_yedge[i, j, k, 0]
            inflow_right = psi_yedge[i, j, k, 1]
            b[base + 0] += etak * hx * inflow_left
            b[base + 1] += etak * hx * inflow_right
        elif etak < 0.0:
            inflow_left  = psi_yedge[i, j + 1, k, 0]
            inflow_right = psi_yedge[i, j + 1, k, 1]
            b[base + 2] += (-etak) * hx * inflow_left
            b[base + 3] += (-etak) * hx * inflow_right


@nb.njit(cache=True, fastmath=True)
def compute_edge_fluxes_nb(psi_cell, mu, eta, bcL, bcR, bcB, bcT):
    Nx, Ny, N_dir, _ = psi_cell.shape

    psi_xedge = np.zeros((Nx + 1, Ny, N_dir, 2), dtype=np.float64)
    psi_yedge = np.zeros((Nx, Ny + 1, N_dir, 2), dtype=np.float64)

    # vertical edges
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

    # horizontal edges
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

    return psi_xedge, psi_yedge


@nb.njit(cache=True, fastmath=True, parallel=True)
def compute_cell_moments_inplane_nb(psi, mu, eta, w):
    Nx, Ny, N_dir, _ = psi.shape
    phi = np.zeros((Nx, Ny), dtype=np.float64)
    Qxx = np.zeros((Nx, Ny), dtype=np.float64)
    Qxy = np.zeros((Nx, Ny), dtype=np.float64)

    for i in nb.prange(Nx):
        for j in range(Ny):
            for k in range(N_dir):
                # corner-average
                avg = 0.25 * (psi[i, j, k, 0] + psi[i, j, k, 1] + psi[i, j, k, 2] + psi[i, j, k, 3])
                wk = w[k]
                phi[i, j] += wk * avg
                qxx_basis = (mu[k] * mu[k] - 0.5)
                qxy_basis = (mu[k] * eta[k])
                Qxx[i, j] += wk * qxx_basis * avg
                Qxy[i, j] += wk * qxy_basis * avg

    return phi, Qxx, Qxy


@nb.njit(cache=True, fastmath=True)
def compute_cell_moments_nb(psi, mu, eta, w):
    Nx, Ny, N_dir, _ = psi.shape
    phi = np.zeros((Nx, Ny), dtype=np.float64)
    Jx  = np.zeros((Nx, Ny), dtype=np.float64)
    Jy  = np.zeros((Nx, Ny), dtype=np.float64)

    for i in range(Nx):
        for j in range(Ny):
            for k in range(N_dir):
                avg = 0.25 * (psi[i, j, k, 0] + psi[i, j, k, 1] + psi[i, j, k, 2] + psi[i, j, k, 3])
                wk = w[k]
                phi[i, j] += wk * avg
                Jx[i, j]  += wk * mu[k] * avg
                Jy[i, j]  += wk * eta[k] * avg

    return phi, Jx, Jy


@nb.njit(cache=True, fastmath=True)
def grad_center_nb(phi, dx, dy):
    Nx, Ny = phi.shape
    gx = np.zeros((Nx, Ny), dtype=np.float64)
    gy = np.zeros((Nx, Ny), dtype=np.float64)

    for j in range(Ny):
        gx[0, j] = (phi[1, j] - phi[0, j]) / dx
        for i in range(1, Nx - 1):
            gx[i, j] = (phi[i + 1, j] - phi[i - 1, j]) / (2.0 * dx)
        gx[Nx - 1, j] = (phi[Nx - 1, j] - phi[Nx - 2, j]) / dx

    for i in range(Nx):
        gy[i, 0] = (phi[i, 1] - phi[i, 0]) / dy
        for j in range(1, Ny - 1):
            gy[i, j] = (phi[i, j + 1] - phi[i, j - 1]) / (2.0 * dy)
        gy[i, Ny - 1] = (phi[i, Ny - 1] - phi[i, Ny - 2]) / dy

    return gx, gy


@nb.njit(cache=True, fastmath=True)
def div_Q_vector_inplane_nb(Qxx, Qxy, dx, dy):
    Nx, Ny = Qxx.shape
    Ux = np.zeros((Nx, Ny), dtype=np.float64)
    Uy = np.zeros((Nx, Ny), dtype=np.float64)

    inv2dx = 1.0 / (2.0 * dx)
    inv2dy = 1.0 / (2.0 * dy)

    for i in range(Nx):
        im1 = i - 1 if i > 0 else i
        ip1 = i + 1 if i < Nx - 1 else i
        for j in range(Ny):
            jm1 = j - 1 if j > 0 else j
            jp1 = j + 1 if j < Ny - 1 else j

            dQxx_dx = (Qxx[ip1, j] - Qxx[im1, j]) * inv2dx
            dQxx_dy = (Qxx[i, jp1] - Qxx[i, jm1]) * inv2dy

            dQxy_dx = (Qxy[ip1, j] - Qxy[im1, j]) * inv2dx
            dQxy_dy = (Qxy[i, jp1] - Qxy[i, jm1]) * inv2dy

            Ux[i, j] = dQxx_dx + dQxy_dy
            Uy[i, j] = dQxy_dx - dQxx_dy

    return Ux, Uy


@nb.njit(cache=True, fastmath=True)
def div_div_Q_inplane_nb(Qxx, Qxy, dx, dy):
    Ux, Uy = div_Q_vector_inplane_nb(Qxx, Qxy, dx, dy)
    Nx, Ny = Qxx.shape
    out = np.zeros((Nx, Ny), dtype=np.float64)

    inv2dx = 1.0 / (2.0 * dx)
    inv2dy = 1.0 / (2.0 * dy)

    for i in range(Nx):
        im1 = i - 1 if i > 0 else i
        ip1 = i + 1 if i < Nx - 1 else i
        for j in range(Ny):
            jm1 = j - 1 if j > 0 else j
            jp1 = j + 1 if j < Ny - 1 else j

            dUx_dx = (Ux[ip1, j] - Ux[im1, j]) * inv2dx
            dUy_dy = (Uy[i, jp1] - Uy[i, jm1]) * inv2dy
            out[i, j] = dUx_dx + dUy_dy

    return out


@nb.njit(cache=True, fastmath=True)
def incident_partial_currents_nb(bcL, bcR, bcB, bcT, mu, eta, w):
    N_dir = w.shape[0]
    Jm_left = 0.0
    Jm_right = 0.0
    Jm_bottom = 0.0
    Jm_top = 0.0

    for k in range(N_dir):
        if mu[k] > 0.0:
            Jm_left += w[k] * (-mu[k]) * bcL[k]
        elif mu[k] < 0.0:
            Jm_right += w[k] * (mu[k]) * bcR[k]  # mu is negative

        if eta[k] > 0.0:
            Jm_bottom += w[k] * (-eta[k]) * bcB[k]
        elif eta[k] < 0.0:
            Jm_top += w[k] * (eta[k]) * bcT[k]  # eta is negative

    return Jm_left, Jm_right, Jm_bottom, Jm_top


@nb.njit(cache=True, fastmath=True)
def yavuz_update_edges_2d_nb(
    psi_xedge, psi_yedge,
    mu, eta,
    bcL, bcR, bcB, bcT,
    phi_half, Jx_half, Jy_half,
    phi_acc,  Jx_acc,  Jy_acc,
):
    Nx_edges, Ny, N_dir, _ = psi_xedge.shape
    Nx = Nx_edges - 1
    Ny_edges = psi_yedge.shape[1]

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

    # re-enforce boundary edges
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
            psi_yedge[i, Ny, k, 0] = bcT[k]
            psi_yedge[i, Ny, k, 1] = bcT[k]

    # clamp negatives for robustness
    for i_edge in range(Nx + 1):
        for j in range(Ny):
            for k in range(N_dir):
                if psi_xedge[i_edge, j, k, 0] < 0.0:
                    psi_xedge[i_edge, j, k, 0] = 0.0
                if psi_xedge[i_edge, j, k, 1] < 0.0:
                    psi_xedge[i_edge, j, k, 1] = 0.0

    for i in range(Nx):
        for j_edge in range(Ny + 1):
            for k in range(N_dir):
                if psi_yedge[i, j_edge, k, 0] < 0.0:
                    psi_yedge[i, j_edge, k, 0] = 0.0
                if psi_yedge[i, j_edge, k, 1] < 0.0:
                    psi_yedge[i, j_edge, k, 1] = 0.0

    return psi_xedge, psi_yedge


# -------------------------
# Diffusion operator + CG
# -------------------------

@nb.njit(cache=True, fastmath=True)
def _harmonic(a, b):
    den = a + b
    if den > 0.0:
        return 2.0 * a * b / den
    return 0.0


@nb.njit(cache=True, fastmath=True)
def _robin_ghost_coeff(Df, alpha, g, h):
    # Df*(phi_g - phi_c)/h + alpha*(phi_c + phi_g)/2 = g
    denom = (Df / h + alpha * 0.5)
    a = (Df / h - alpha * 0.5) / denom   # phi_g = a*phi_c + c
    c = g / denom
    return a, c


@nb.njit(cache=True, fastmath=True, parallel=True)
def diffusion_matvec_nb(phi, out, sig_a, Dcell, dx, dy,
                        alphaL, alphaR, alphaB, alphaT,
                        aL, aR, aB, aT):
    """
    out = A(phi) where A is the diffusion+reaction operator with Robin BC
    using ghost elimination coeffs a*(phi_c) (no g contribution here).
    """
    Nx, Ny = phi.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    for i in nb.prange(Nx):
        for j in range(Ny):
            Dc = Dcell[i, j]
            sa = sig_a[i, j]

            # west/east face D
            if i > 0:
                Dw = _harmonic(Dc, Dcell[i - 1, j])
            else:
                Dw = Dc
            if i < Nx - 1:
                De = _harmonic(Dc, Dcell[i + 1, j])
            else:
                De = Dc

            # south/north face D
            if j > 0:
                Ds = _harmonic(Dc, Dcell[i, j - 1])
            else:
                Ds = Dc
            if j < Ny - 1:
                Dn = _harmonic(Dc, Dcell[i, j + 1])
            else:
                Dn = Dc

            diag = sa + (Dw + De) * inv_dx2 + (Ds + Dn) * inv_dy2
            acc = diag * phi[i, j]

            # west
            if i > 0:
                acc += (-Dw * inv_dx2) * phi[i - 1, j]
            else:
                acc += (-Dw * inv_dx2) * (aL[j] * phi[i, j])

            # east
            if i < Nx - 1:
                acc += (-De * inv_dx2) * phi[i + 1, j]
            else:
                acc += (-De * inv_dx2) * (aR[j] * phi[i, j])

            # south
            if j > 0:
                acc += (-Ds * inv_dy2) * phi[i, j - 1]
            else:
                acc += (-Ds * inv_dy2) * (aB[i] * phi[i, j])

            # north
            if j < Ny - 1:
                acc += (-Dn * inv_dy2) * phi[i, j + 1]
            else:
                acc += (-Dn * inv_dy2) * (aT[i] * phi[i, j])

            out[i, j] = acc


@nb.njit(cache=True, fastmath=True)
def build_diffusion_rhs_nb(phi_rhs, Dcell, dx, dy,
                           alphaL, gL, alphaR, gR, alphaB, gB, alphaT, gT,
                           aL, cL, aR, cR, aB, cB, aT, cT):
    """
    Build rhs = phi_rhs + boundary ghost contributions from 'c' terms.
    Also fills ghost coeff arrays (a,c) for each side based on D and g.
    """
    Nx, Ny = phi_rhs.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    rhs = phi_rhs.copy()

    # left/right depend on j; bottom/top depend on i
    for j in range(Ny):
        # left boundary cell i=0
        Dc = Dcell[0, j]
        a, c = _robin_ghost_coeff(Dc, alphaL, gL[j], dx)
        aL[j] = a
        cL[j] = c
        rhs[0, j] += (Dc * inv_dx2) * c

        # right boundary cell i=Nx-1
        Dc = Dcell[Nx - 1, j]
        a, c = _robin_ghost_coeff(Dc, alphaR, gR[j], dx)
        aR[j] = a
        cR[j] = c
        rhs[Nx - 1, j] += (Dc * inv_dx2) * c

    for i in range(Nx):
        # bottom boundary cell j=0
        Dc = Dcell[i, 0]
        a, c = _robin_ghost_coeff(Dc, alphaB, gB[i], dy)
        aB[i] = a
        cB[i] = c
        rhs[i, 0] += (Dc * inv_dy2) * c

        # top boundary cell j=Ny-1
        Dc = Dcell[i, Ny - 1]
        a, c = _robin_ghost_coeff(Dc, alphaT, gT[i], dy)
        aT[i] = a
        cT[i] = c
        rhs[i, Ny - 1] += (Dc * inv_dy2) * c

    return rhs


@nb.njit(cache=True, fastmath=True)
def cg_solve_diffusion_nb(phi0, b, sig_a, Dcell, dx, dy,
                          alphaL, alphaR, alphaB, alphaT,
                          aL, aR, aB, aT,
                          max_it, tol):
    """
    Matrix-free CG solve for A phi = b.
    """
    Nx, Ny = b.shape
    phi = phi0.copy()

    r = np.empty((Nx, Ny), dtype=np.float64)
    Ap = np.empty((Nx, Ny), dtype=np.float64)
    p = np.empty((Nx, Ny), dtype=np.float64)

    diffusion_matvec_nb(phi, Ap, sig_a, Dcell, dx, dy, alphaL, alphaR, alphaB, alphaT, aL, aR, aB, aT)

    # r = b - A phi
    rr = 0.0
    for i in range(Nx):
        for j in range(Ny):
            r[i, j] = b[i, j] - Ap[i, j]
            p[i, j] = r[i, j]
            rr += r[i, j] * r[i, j]

    if rr <= tol * tol:
        return phi

    for it in range(max_it):
        diffusion_matvec_nb(p, Ap, sig_a, Dcell, dx, dy, alphaL, alphaR, alphaB, alphaT, aL, aR, aB, aT)

        pAp = 0.0
        for i in range(Nx):
            for j in range(Ny):
                pAp += p[i, j] * Ap[i, j]

        if pAp == 0.0:
            break

        alpha = rr / pAp

        rr_new = 0.0
        for i in range(Nx):
            for j in range(Ny):
                phi[i, j] += alpha * p[i, j]
                r[i, j]   -= alpha * Ap[i, j]
                rr_new += r[i, j] * r[i, j]

        if rr_new <= tol * tol:
            break

        beta = rr_new / rr
        for i in range(Nx):
            for j in range(Ny):
                p[i, j] = r[i, j] + beta * p[i, j]

        rr = rr_new

    return phi


@nb.njit(cache=True, fastmath=True)
def smm_nb(psi, psi_xedge, psi_yedge,
           q_cell, sig_t, sig_s, dx, dy, mu, eta, w,
           bcL, bcR, bcB, bcT,
           update_yavuz):
    """
    Returns: psi, psi_xedge, psi_yedge, phi_half, phi_acc
    """
    phi_half, Qxx, Qxy = compute_cell_moments_inplane_nb(psi, mu, eta, w)
    Nx, Ny = phi_half.shape

    # sig_a
    sig_a = sig_t - sig_s  # array

    ddQ = div_div_Q_inplane_nb(Qxx, Qxy, dx, dy)
    s_sm = np.empty((Nx, Ny), dtype=np.float64)
    for i in range(Nx):
        for j in range(Ny):
            s_sm[i, j] = ddQ[i, j] / (sig_t[i, j] + 1e-300)

    Ux, Uy = div_Q_vector_inplane_nb(Qxx, Qxy, dx, dy)

    Jm_left, Jm_right, Jm_bottom, Jm_top = incident_partial_currents_nb(bcL, bcR, bcB, bcT, mu, eta, w)

    omega_total = 0.0
    for k in range(w.shape[0]):
        omega_total += w[k]

    # M1x, M1y
    M1x = 0.0
    M1y = 0.0
    for k in range(w.shape[0]):
        if mu[k] > 0.0:
            M1x += w[k] * mu[k]
        if eta[k] > 0.0:
            M1y += w[k] * eta[k]

    alpha_x = 2.0 * M1x / omega_total
    alpha_y = 2.0 * M1y / omega_total

    # boundary g arrays
    g_left  = np.empty(Ny, dtype=np.float64)
    g_right = np.empty(Ny, dtype=np.float64)
    g_bottom = np.empty(Nx, dtype=np.float64)
    g_top    = np.empty(Nx, dtype=np.float64)

    for j in range(Ny):
        g_left[j]  = (-2.0 * Jm_left)  + (Ux[0, j] / (sig_t[0, j] + 1e-300))
        g_right[j] = (-2.0 * Jm_right) - (Ux[Nx - 1, j] / (sig_t[Nx - 1, j] + 1e-300))

    for i in range(Nx):
        g_bottom[i] = (-2.0 * Jm_bottom) + (Uy[i, 0] / (sig_t[i, 0] + 1e-300))
        g_top[i]    = (-2.0 * Jm_top)    - (Uy[i, Ny - 1] / (sig_t[i, Ny - 1] + 1e-300))

    # D0
    D0 = np.empty((Nx, Ny), dtype=np.float64)
    for i in range(Nx):
        for j in range(Ny):
            D0[i, j] = 1.0 / (2.0 * (sig_t[i, j] + 1e-300))

    # rhs = q_cell + s_sm
    rhs = np.empty((Nx, Ny), dtype=np.float64)
    for i in range(Nx):
        for j in range(Ny):
            rhs[i, j] = q_cell[i, j] + s_sm[i, j]

    # ghost coeff arrays
    aL = np.empty(Ny, dtype=np.float64); cL = np.empty(Ny, dtype=np.float64)
    aR = np.empty(Ny, dtype=np.float64); cR = np.empty(Ny, dtype=np.float64)
    aB = np.empty(Nx, dtype=np.float64); cB = np.empty(Nx, dtype=np.float64)
    aT = np.empty(Nx, dtype=np.float64); cT = np.empty(Nx, dtype=np.float64)

    b = build_diffusion_rhs_nb(rhs, D0, dx, dy,
                               alpha_x, g_left, alpha_x, g_right, alpha_y, g_bottom, alpha_y, g_top,
                               aL, cL, aR, cR, aB, cB, aT, cT)

    # CG solve
    phi0 = phi_half  # good initial guess
    phi_acc = cg_solve_diffusion_nb(phi0, b, sig_a, D0, dx, dy,
                                    alpha_x, alpha_x, alpha_y, alpha_y,
                                    aL, aR, aB, aT,
                                    max_it=400, tol=1e-10)

    if update_yavuz:
        phi_half2, Jx_half, Jy_half = compute_cell_moments_nb(psi, mu, eta, w)
        gx, gy = grad_center_nb(phi_acc, dx, dy)

        Jx_acc = np.empty((Nx, Ny), dtype=np.float64)
        Jy_acc = np.empty((Nx, Ny), dtype=np.float64)
        for i in range(Nx):
            for j in range(Ny):
                invst = 1.0 / (sig_t[i, j] + 1e-300)
                Jx_acc[i, j] = -(0.5 * gx[i, j] + Ux[i, j]) * invst
                Jy_acc[i, j] = -(0.5 * gy[i, j] + Uy[i, j]) * invst

        psi_xedge, psi_yedge = yavuz_update_edges_2d_nb(
            psi_xedge, psi_yedge,
            mu, eta,
            bcL, bcR, bcB, bcT,
            phi_half2, Jx_half, Jy_half,
            phi_acc,  Jx_acc,  Jy_acc
        )
    else:
        # "rescale" mode (simple; if you want it, we can add fully numba version too)
        eps = 1e-14
        for i in range(Nx):
            for j in range(Ny):
                scale = phi_acc[i, j] / (phi_half[i, j] + eps)
                if scale < 0.0:
                    scale = 0.0
                for k in range(w.shape[0]):
                    for c in range(4):
                        psi[i, j, k, c] *= scale
        psi_xedge, psi_yedge = compute_edge_fluxes_nb(psi, mu, eta, bcL, bcR, bcB, bcT)

    return psi, psi_xedge, psi_yedge, phi_half, phi_acc


# ============================================================
# Compiled core loop
# ============================================================
@nb.njit(cache=True, fastmath=True, parallel=True)
def transport_2d_oci_core_nb(
    Nx, Ny, dx, dy,
    mu, eta, w, omega_total,
    sig_t, sig_s, q_cell,
    bcL, bcR, bcB, bcT,
    tol, max_it, printer_int,
    smm_do, update_yavuz
):
    N_dir = w.shape[0]
    psi = np.zeros((Nx, Ny, N_dir, 4), dtype=np.float64)
    psi_last = np.zeros((Nx, Ny, N_dir, 4), dtype=np.float64)

    psi_xedge, psi_yedge = compute_edge_fluxes_nb(psi, mu, eta, bcL, bcR, bcB, bcT)
    S_stream = precompute_stream_mats(mu, eta, dx, dy)

    nloc = 4 * N_dir

    err_last = 1.0
    rho = 0.0
    it = 0

    for it in range(max_it):

        # --- transport half-step (OCI) ---
        for i in nb.prange(Nx):
            # thread-private scratch (safe)
            A = np.empty((nloc, nloc), dtype=np.float64)
            b = np.empty(nloc, dtype=np.float64)

            for j in range(Ny):
                build_local_A_b(
                    i, j, dx, dy, q_cell[i, j],
                    sig_t[i, j], sig_s[i, j],
                    mu, eta, w, omega_total,
                    S_stream,
                    psi_xedge, psi_yedge,
                    A, b
                )

                xloc = np.linalg.solve(A, b)

                for k in range(N_dir):
                    base = 4 * k
                    psi[i, j, k, 0] = xloc[base + 0]
                    psi[i, j, k, 1] = xloc[base + 1]
                    psi[i, j, k, 2] = xloc[base + 2]
                    psi[i, j, k, 3] = xloc[base + 3]

        psi_xedge, psi_yedge = compute_edge_fluxes_nb(psi, mu, eta, bcL, bcR, bcB, bcT)

        if smm_do:
            psi, psi_xedge, psi_yedge, phi_half, phi_acc = smm_nb(
                psi, psi_xedge, psi_yedge,
                q_cell, sig_t, sig_s, dx, dy, mu, eta, w,
                bcL, bcR, bcB, bcT,
                update_yavuz
            )

        # convergence check (can also be parallelized, but keep as-is for now)
        err = 0.0
        for i2 in range(Nx):
            for j2 in range(Ny):
                for k2 in range(N_dir):
                    for c2 in range(4):
                        d = psi[i2, j2, k2, c2] - psi_last[i2, j2, k2, c2]
                        err += d * d
        err = np.sqrt(err)

        if it > 0:
            rho = err / err_last
        else:
            rho = 0.0

        # end the calculation early if divergence
        if it > 45 and rho > 1.0:
            break
            #return phi, psi, rho, it

        if printer_int != 0:
            print("it", it, "err", err, "rho", rho)

        if it > 2:
            rr = rho
            if rr < 0.0:
                rr = 0.0
            if rr > 0.999999:
                rr = 0.999999
            if err < tol * (1.0 - rr):
                break

        for i2 in range(Nx):
            for j2 in range(Ny):
                for k2 in range(N_dir):
                    for c2 in range(4):
                        psi_last[i2, j2, k2, c2] = psi[i2, j2, k2, c2]

        err_last = err
        if err_last < 1e-300:
            err_last = 1e-300

    phi = np.zeros((Nx, Ny), dtype=np.float64)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(N_dir):
                avg = 0.25 * (psi[i, j, k, 0] + psi[i, j, k, 1] + psi[i, j, k, 2] + psi[i, j, k, 3])
                phi[i, j] += w[k] * avg

    return phi, psi, rho, it



# ============================================================
# Public API (same signature as your original)
# ============================================================
def transport_2d_oci(
    Nx=30, Ny=30,
    Lx=3.0, Ly=3.0,
    N_dir=4,
    sig_t=None, sig_s=None, q=None,
    bc=None,
    tol=1e-4, max_it=2000, printer=True,
    smm_acc=True, update="yavuz",
    sig_t_val=1.0, sig_s_val=0.0, q_val=1.0,
):
    # keep your defaults
    if bc is None:
        bc = dict(left=0.0, right=0.0, bottom=0.0, top=0.0)

    dx = Lx / Nx
    dy = Ly / Ny

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")

    if sig_t is None:
        sig_t = sig_t_val * np.ones((Nx, Ny), dtype=np.float64)
    else:
        sig_t = np.asarray(sig_t, dtype=np.float64)

    if sig_s is None:
        sig_s = sig_s_val * np.ones((Nx, Ny), dtype=np.float64)
    else:
        sig_s = np.asarray(sig_s, dtype=np.float64)

    if q is None:
        q_cell = q_val * np.ones((Nx, Ny), dtype=np.float64)
    else:
        q_cell = np.asarray(q, dtype=np.float64)

    mu, eta, w, omega_total = make_circle_quadrature(N_dir)

    # convert bc dict -> per-direction arrays (Numba-friendly)
    bcL = np.asarray(bc.get("left", 0.0), dtype=np.float64)
    bcR = np.asarray(bc.get("right", 0.0), dtype=np.float64)
    bcB = np.asarray(bc.get("bottom", 0.0), dtype=np.float64)
    bcT = np.asarray(bc.get("top", 0.0), dtype=np.float64)

    # ensure (N_dir,)
    if bcL.ndim == 0:
        bcL = np.full(N_dir, float(bcL))
    if bcR.ndim == 0:
        bcR = np.full(N_dir, float(bcR))
    if bcB.ndim == 0:
        bcB = np.full(N_dir, float(bcB))
    if bcT.ndim == 0:
        bcT = np.full(N_dir, float(bcT))

    # keep your original gating behavior (uses sig_s_val)
    smm_do = bool(smm_acc and (sig_s_val > 0.0))
    update_yavuz = (update == "yavuz")

    phi, psi, rho, it = transport_2d_oci_core_nb(
        Nx, Ny, dx, dy,
        mu, eta, w, omega_total,
        sig_t, sig_s, q_cell,
        bcL, bcR, bcB, bcT,
        float(tol), int(max_it), int(1 if printer else 0),
        smm_do, update_yavuz
    )


    return phi, psi, rho, it


if __name__ == "__main__":

    import os
    #os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    q = 2.0
    sigma = 5.0
    c = .9
    sigma_s = sigma*c
    inf_homo = q / ( sigma*(1-c) )
    inf_homo /= (2*np.pi)

    L = 10
    mfp = 0.1
    dx = mfp/sigma
    N = int(L/dx)

    bc = dict(left=inf_homo, right=inf_homo, bottom=inf_homo, top=inf_homo)

    phi, psi, ang, it = transport_2d_oci(
        Nx=N, Ny=N, Lx=L, Ly=L,
        N_dir=8,
        sig_t_val=sigma, sig_s_val=sigma_s,
        q_val=q,
        bc=bc,
        tol=1e-6, max_it=500, printer=True,
        smm_acc=True, update="rescale"
    )

    dx = L / N
    dy = L / N

    x = (np.arange(N) + 0.5) * dx
    y = (np.arange(N) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")
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
