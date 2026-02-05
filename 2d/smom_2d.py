import numpy as np
import matplotlib.pyplot as plt

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


# -----------------------------
# Corner-balance local operator
# -----------------------------
def A_dir_corner_balance(mu, eta, dx, dy, sig_t):
    hx = dx / 2.0
    hy = dy / 2.0
    area = dx * dy

    Ax = hy * np.array([[abs(mu)/2.0,  mu/2.0],
                        [-mu/2.0,     abs(mu)/2.0]])

    Ay = hx * np.array([[abs(eta)/2.0,  eta/2.0],
                        [-eta/2.0,     abs(eta)/2.0]])

    A = np.kron(np.eye(2), Ax) + np.kron(Ay, np.eye(2))
    A += (sig_t * area / 4.0) * np.eye(4)
    return A


def build_cell_matrix(dx, dy, sig_t, sig_s, mu, eta, w, omega_total):
    N_dir = len(w)
    n = 4 * N_dir
    A = np.zeros((n, n))

    for k in range(N_dir):
        Ak = A_dir_corner_balance(mu[k], eta[k], dx, dy, sig_t)
        A[4*k:4*k+4, 4*k:4*k+4] = Ak

    area = dx * dy
    beta = sig_s * area / (16.0 * omega_total)   # NOTE the 16 = 4 (quarter area) * 4 (corner average)

    for c in range(4):
        row_idx = np.arange(N_dir) * 4 + c
        for cp in range(4):
            col_idx = np.arange(N_dir) * 4 + cp
            A[np.ix_(row_idx, col_idx)] -= beta * w[None, :]

    return A


# -----------------------------
# Upwind edge closure
# -----------------------------
def compute_edge_fluxes(psi_cell, mu, eta, bc):
    Nx, Ny, N_dir, _ = psi_cell.shape

    def as_dir_array(val):
        val = np.asarray(val)
        if val.shape == ():
            return val * np.ones(N_dir)
        assert val.shape == (N_dir,)
        return val

    bcL = as_dir_array(bc.get("left", 0.0))
    bcR = as_dir_array(bc.get("right", 0.0))
    bcB = as_dir_array(bc.get("bottom", 0.0))
    bcT = as_dir_array(bc.get("top", 0.0))

    psi_xedge = np.zeros((Nx+1, Ny, N_dir, 2))
    psi_yedge = np.zeros((Nx, Ny+1, N_dir, 2))

    for i_edge in range(Nx+1):
        for j in range(Ny):
            for k in range(N_dir):
                if i_edge == 0:
                    psi_xedge[i_edge, j, k, :] = bcL[k]
                elif i_edge == Nx:
                    psi_xedge[i_edge, j, k, :] = bcR[k]
                else:
                    if mu[k] > 0:
                        iL = i_edge - 1
                        psi_xedge[i_edge, j, k, 0] = psi_cell[iL, j, k, 1]  # SE
                        psi_xedge[i_edge, j, k, 1] = psi_cell[iL, j, k, 3]  # NE
                    else:
                        iR = i_edge
                        psi_xedge[i_edge, j, k, 0] = psi_cell[iR, j, k, 0]  # SW
                        psi_xedge[i_edge, j, k, 1] = psi_cell[iR, j, k, 2]  # NW

    for i in range(Nx):
        for j_edge in range(Ny+1):
            for k in range(N_dir):
                if j_edge == 0:
                    psi_yedge[i, j_edge, k, :] = bcB[k]
                elif j_edge == Ny:
                    psi_yedge[i, j_edge, k, :] = bcT[k]
                else:
                    if eta[k] > 0:
                        jB = j_edge - 1
                        psi_yedge[i, j_edge, k, 0] = psi_cell[i, jB, k, 2]  # NW
                        psi_yedge[i, j_edge, k, 1] = psi_cell[i, jB, k, 3]  # NE
                    else:
                        jT = j_edge
                        psi_yedge[i, j_edge, k, 0] = psi_cell[i, jT, k, 0]  # SW
                        psi_yedge[i, j_edge, k, 1] = psi_cell[i, jT, k, 1]  # SE

    return psi_xedge, psi_yedge


# -----------------------------
# RHS builder
# -----------------------------
def build_cell_rhs(i, j, dx, dy, q_cell, mu, eta, w, omega_total,
                   psi_xedge, psi_yedge):
    N_dir = len(w)

    hx = dx / 2.0
    hy = dy / 2.0
    area = dx * dy
    q_iso = q_cell / omega_total

    b = np.zeros(4 * N_dir)

    for k in range(N_dir):
        base = (area / 4.0) * q_iso
        b[4*k + 0] = base
        b[4*k + 1] = base
        b[4*k + 2] = base
        b[4*k + 3] = base

        if mu[k] > 0:
            inflow_bot = psi_xedge[i, j, k, 0]
            inflow_top = psi_xedge[i, j, k, 1]
            b[4*k + 0] += mu[k] * hy * inflow_bot
            b[4*k + 2] += mu[k] * hy * inflow_top
        elif mu[k] < 0:
            inflow_bot = psi_xedge[i+1, j, k, 0]
            inflow_top = psi_xedge[i+1, j, k, 1]
            b[4*k + 1] += (-mu[k]) * hy * inflow_bot
            b[4*k + 3] += (-mu[k]) * hy * inflow_top

        if eta[k] > 0:
            inflow_left  = psi_yedge[i, j,   k, 0]
            inflow_right = psi_yedge[i, j,   k, 1]
            b[4*k + 0] += eta[k] * hx * inflow_left
            b[4*k + 1] += eta[k] * hx * inflow_right
        elif eta[k] < 0:
            inflow_left  = psi_yedge[i, j+1, k, 0]
            inflow_right = psi_yedge[i, j+1, k, 1]
            b[4*k + 2] += (-eta[k]) * hx * inflow_left
            b[4*k + 3] += (-eta[k]) * hx * inflow_right

    return b


# ============================================================
# Second-moment acceleration (2D in-plane)
# ============================================================
def compute_incident_partial_currents_2d_inplane(bc, mu, eta, w):
    """
    Returns Jminus_inc arrays on each side:
      left/right: shape (Ny,) but constant in y for uniform bc -> we return scalar
      bottom/top: shape (Nx,) similarly -> we return scalar
    We'll return scalars (one per side) since bc is constant here; extendable.
    Convention: J^- = ∫_{Omega·n<0} (Omega·n) psi_inc dOmega  (negative value).
    """
    N_dir = len(w)

    def as_dir_array(val):
        val = np.asarray(val)
        if val.shape == ():
            return val * np.ones(N_dir)
        assert val.shape == (N_dir,)
        return val

    bcL = as_dir_array(bc.get("left", 0.0))
    bcR = as_dir_array(bc.get("right", 0.0))
    bcB = as_dir_array(bc.get("bottom", 0.0))
    bcT = as_dir_array(bc.get("top", 0.0))

    # Left boundary: outward n = (-1,0) -> mu_n = -mu, inflow mu>0
    Jm_left  = np.sum(w[(mu > 0)] * (-mu[(mu > 0)]) * bcL[(mu > 0)])

    # Right boundary: outward n = (+1,0) -> mu_n = +mu, inflow mu<0
    Jm_right = np.sum(w[(mu < 0)] * ( mu[(mu < 0)]) * bcR[(mu < 0)])  # mu is negative

    # Bottom boundary: outward n = (0,-1) -> mu_n = -eta, inflow eta>0
    Jm_bottom = np.sum(w[(eta > 0)] * (-eta[(eta > 0)]) * bcB[(eta > 0)])

    # Top boundary: outward n = (0,+1) -> mu_n = +eta, inflow eta<0
    Jm_top    = np.sum(w[(eta < 0)] * ( eta[(eta < 0)]) * bcT[(eta < 0)])  # eta is negative

    return dict(left=Jm_left, right=Jm_right, bottom=Jm_bottom, top=Jm_top)


def _pad_neumann(A):
    return np.pad(A, pad_width=((1, 1), (1, 1)), mode="edge")


def div_Q_vector_inplane(Qxx, Qxy, dx, dy):
    """
    Returns U = div(Q) as (Ux, Uy), where:
      Ux = dQxx/dx + dQxy/dy
      Uy = dQxy/dx + dQyy/dy, but Qyy = -Qxx (traceless in-plane) -> dQyy/dy = -dQxx/dy
      => Uy = dQxy/dx - dQxx/dy
    """
    Qxxp = _pad_neumann(Qxx)
    Qxyp = _pad_neumann(Qxy)

    dQxx_dx = (Qxxp[2:, 1:-1] - Qxxp[:-2, 1:-1]) / (2.0 * dx)
    dQxx_dy = (Qxxp[1:-1, 2:] - Qxxp[1:-1, :-2]) / (2.0 * dy)

    dQxy_dx = (Qxyp[2:, 1:-1] - Qxyp[:-2, 1:-1]) / (2.0 * dx)
    dQxy_dy = (Qxyp[1:-1, 2:] - Qxyp[1:-1, :-2]) / (2.0 * dy)

    Ux = dQxx_dx + dQxy_dy
    Uy = dQxy_dx - dQxx_dy
    return Ux, Uy


def div_div_Q_inplane(Qxx, Qxy, dx, dy):
    Ux, Uy = div_Q_vector_inplane(Qxx, Qxy, dx, dy)
    Uxp = _pad_neumann(Ux)
    Uyp = _pad_neumann(Uy)

    dUx_dx = (Uxp[2:, 1:-1] - Uxp[:-2, 1:-1]) / (2.0 * dx)
    dUy_dy = (Uyp[1:-1, 2:] - Uyp[1:-1, :-2]) / (2.0 * dy)
    return dUx_dx + dUy_dy


def solve_diffusion(phi_rhs, sig_a, Dcell, dx, dy, robin):
    """
    Solve:  -div(D grad phi) + sig_a * phi = phi_rhs
    on a uniform grid with cell-centered unknowns.

    Dcell, sig_a, phi_rhs are (Nx,Ny).
    Robin BC on each side: D * dphi/dn + alpha * phi = g
    implemented via ghost-cell elimination using:
      D*(phi_g - phi_c)/h + alpha*(phi_c + phi_g)/2 = g
    with h = dx on left/right and h = dy on bottom/top, consistent with your original.

    robin sides:
      robin["left"]   = {"alpha": alpha, "g": g_left[j]}
      robin["right"]  = {"alpha": alpha, "g": g_right[j]}
      robin["bottom"] = {"alpha": alpha, "g": g_bottom[i]}
      robin["top"]    = {"alpha": alpha, "g": g_top[i]}
    """
    Nx, Ny = phi_rhs.shape
    N = Nx * Ny

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    def idx(i, j):
        return i * Ny + j

    def harmonic(a, b):
        den = a + b
        return np.where(den > 0.0, 2.0 * a * b / den, 0.0)

    def robin_ghost_coeff(Df, alpha, g, h):
        # Df*(phi_g - phi_c)/h + alpha*(phi_c + phi_g)/2 = g
        denom = (Df / h + alpha / 2.0)
        a = (Df / h - alpha / 2.0) / denom   # phi_g = a*phi_c + c
        c = g / denom
        return a, c

    alphaL, gL = robin["left"]["alpha"], robin["left"]["g"]      # gL shape (Ny,)
    alphaR, gR = robin["right"]["alpha"], robin["right"]["g"]    # gR shape (Ny,)
    alphaB, gB = robin["bottom"]["alpha"], robin["bottom"]["g"]  # gB shape (Nx,)
    alphaT, gT = robin["top"]["alpha"], robin["top"]["g"]        # gT shape (Nx,)

    A = np.zeros((N, N))
    b = phi_rhs.reshape(N).copy()

    for i in range(Nx):
        for j in range(Ny):
            p = idx(i, j)

            Dc = Dcell[i, j]
            sa = sig_a[i, j]

            # Face diffusion coefficients
            # West/East faces:
            if i > 0:
                Dw = harmonic(Dc, Dcell[i - 1, j])
            else:
                Dw = Dc
            if i < Nx - 1:
                De = harmonic(Dc, Dcell[i + 1, j])
            else:
                De = Dc

            # South/North faces:
            if j > 0:
                Ds = harmonic(Dc, Dcell[i, j - 1])
            else:
                Ds = Dc
            if j < Ny - 1:
                Dn = harmonic(Dc, Dcell[i, j + 1])
            else:
                Dn = Dc

            diag = sa + (Dw + De) * inv_dx2 + (Ds + Dn) * inv_dy2

            # West
            if i > 0:
                A[p, idx(i - 1, j)] = -Dw * inv_dx2
            else:
                a, c = robin_ghost_coeff(Dw, alphaL, gL[j], dx)
                diag += (-Dw * inv_dx2) * a
                b[p] += (Dw * inv_dx2) * c

            # East
            if i < Nx - 1:
                A[p, idx(i + 1, j)] = -De * inv_dx2
            else:
                a, c = robin_ghost_coeff(De, alphaR, gR[j], dx)
                diag += (-De * inv_dx2) * a
                b[p] += (De * inv_dx2) * c

            # South
            if j > 0:
                A[p, idx(i, j - 1)] = -Ds * inv_dy2
            else:
                a, c = robin_ghost_coeff(Ds, alphaB, gB[i], dy)
                diag += (-Ds * inv_dy2) * a
                b[p] += (Ds * inv_dy2) * c

            # North
            if j < Ny - 1:
                A[p, idx(i, j + 1)] = -Dn * inv_dy2
            else:
                a, c = robin_ghost_coeff(Dn, alphaT, gT[i], dy)
                diag += (-Dn * inv_dy2) * a
                b[p] += (Dn * inv_dy2) * c

            A[p, p] = diag

    phi = np.linalg.solve(A, b).reshape(Nx, Ny)
    return phi


def compute_cell_moments(psi, mu, eta, w):
    """
    psi: (Nx,Ny,N_dir,4)
    Returns cell-centered moments:
      phi (Nx,Ny)
      Jx  (Nx,Ny)
      Jy  (Nx,Ny)
    using corner-average angular flux.
    """
    psi_avg = np.mean(psi, axis=3)  # (Nx,Ny,N_dir)
    phi = np.tensordot(psi_avg, w, axes=([2],[0]))  # (Nx,Ny)
    Jx  = np.tensordot(psi_avg, w * mu, axes=([2],[0]))
    Jy  = np.tensordot(psi_avg, w * eta, axes=([2],[0]))
    return phi, Jx, Jy


def grad_center(phi, dx, dy):
    """Centered gradient on cell centers with one-sided at boundaries."""
    Nx, Ny = phi.shape
    gx = np.zeros_like(phi)
    gy = np.zeros_like(phi)

    gx[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2*dx)
    gx[0, :]    = (phi[1, :] - phi[0, :]) / dx
    gx[-1, :]   = (phi[-1, :] - phi[-2, :]) / dx

    gy[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2*dy)
    gy[:, 0]    = (phi[:, 1] - phi[:, 0]) / dy
    gy[:, -1]   = (phi[:, -1] - phi[:, -2]) / dy

    return gx, gy

def compute_cell_moments_inplane(psi, mu, eta, w):
    psi_avg = np.mean(psi, axis=3)  # (Nx,Ny,N_dir)
    phi = np.tensordot(psi_avg, w, axes=([2], [0]))

    qxx_basis = (mu**2 - 0.5)
    qxy_basis = (mu * eta)

    Qxx = np.tensordot(psi_avg, w * qxx_basis, axes=([2], [0]))
    Qxy = np.tensordot(psi_avg, w * qxy_basis, axes=([2], [0]))
    return phi, Qxx, Qxy


def diffusion_face_currents(phi, Dcell, dx, dy, robin):
    """
    Compute face currents Jx_face (Nx+1,Ny), Jy_face (Nx,Ny+1)
    with Jx positive in +x, Jy positive in +y.

    Uses harmonic face D in the interior.
    Uses the same Robin ghost relation as solve_diffusion at boundaries.
    """
    Nx, Ny = phi.shape

    def harmonic(a, b):
        den = a + b
        return np.where(den > 0.0, 2.0 * a * b / den, 0.0)

    def robin_ghost_coeff(Df, alpha, g, h):
        denom = (Df / h + alpha / 2.0)
        a = (Df / h - alpha / 2.0) / denom   # phi_g = a*phi_c + c
        c = g / denom
        return a, c

    Jx = np.zeros((Nx+1, Ny))
    Jy = np.zeros((Nx, Ny+1))

    # --- x-faces ---
    for j in range(Ny):
        # left boundary face (between ghost and cell 0)
        Dc = Dcell[0, j]
        Dw = Dc
        a, c = robin_ghost_coeff(Dw, robin["left"]["alpha"], robin["left"]["g"][j], dx)
        phi_g = a * phi[0, j] + c
        # gradient in +x at boundary face: (phi_c - phi_g)/dx
        Jx[0, j] = -Dw * (phi[0, j] - phi_g) / dx

        # interior faces
        for i in range(1, Nx):
            Df = harmonic(Dcell[i-1, j], Dcell[i, j])
            Jx[i, j] = -Df * (phi[i, j] - phi[i-1, j]) / dx

        # right boundary face (between cell Nx-1 and ghost)
        Dc = Dcell[Nx-1, j]
        De = Dc
        a, c = robin_ghost_coeff(De, robin["right"]["alpha"], robin["right"]["g"][j], dx)
        phi_g = a * phi[Nx-1, j] + c
        Jx[Nx, j] = -De * (phi_g - phi[Nx-1, j]) / dx  # gradient in +x: (phi_g - phi_c)/dx

    # --- y-faces ---
    for i in range(Nx):
        # bottom boundary face
        Dc = Dcell[i, 0]
        Ds = Dc
        a, c = robin_ghost_coeff(Ds, robin["bottom"]["alpha"], robin["bottom"]["g"][i], dy)
        phi_g = a * phi[i, 0] + c
        Jy[i, 0] = -Ds * (phi[i, 0] - phi_g) / dy

        # interior faces
        for j in range(1, Ny):
            Df = harmonic(Dcell[i, j-1], Dcell[i, j])
            Jy[i, j] = -Df * (phi[i, j] - phi[i, j-1]) / dy

        # top boundary face
        Dc = Dcell[i, Ny-1]
        Dn = Dc
        a, c = robin_ghost_coeff(Dn, robin["top"]["alpha"], robin["top"]["g"][i], dy)
        phi_g = a * phi[i, Ny-1] + c
        Jy[i, Ny] = -Dn * (phi_g - phi[i, Ny-1]) / dy

    return Jx, Jy

def holo_set_interior_inflow_from_LO(
    psi_xedge, psi_yedge, mu, eta, w, bc,
    phi_acc, Jx_face, Jy_face,
):
    """
    Overwrite *interior* inflow on faces using LO face moments (phi_acc, J_face).
    Keeps boundary inflow = bc.
    """
    Nx_edges, Ny, N_dir, _ = psi_xedge.shape
    Nx = Nx_edges - 1
    Ny_edges = psi_yedge.shape[1]

    mu = np.asarray(mu); eta = np.asarray(eta); w = np.asarray(w)
    omega_total = np.sum(w)

    # Precompute M2 for x-normal and y-normal (full-range)
    M2x = np.sum(w * mu**2)
    M2y = np.sum(w * eta**2)

    # --- vertical faces i_edge = 1..Nx-1 ---
    for i_edge in range(1, Nx):
        iL = i_edge - 1
        iR = i_edge
        for j in range(Ny):
            # face scalar flux: average of adjacent cells (simple, consistent enough here)
            phi_f = 0.5 * (phi_acc[iL, j] + phi_acc[iR, j])

            # Jx_face is the net current in +x through this face
            Jf = Jx_face[i_edge, j]

            # Inflow to RIGHT cell uses mu>0 with mu_n = mu
            a = phi_f / omega_total
            b = Jf / (M2x + 1e-300)
            for k in np.where(mu > 0)[0]:
                psi_xedge[i_edge, j, k, :] = a + b * mu[k]

            # Inflow to LEFT cell uses mu<0 with mu_n = -mu, and outward normal is -x so Jn_out = -Jf
            bL = (-Jf) / (M2x + 1e-300)
            for k in np.where(mu < 0)[0]:
                psi_xedge[i_edge, j, k, :] = a + bL * (-mu[k])

    # --- horizontal faces j_edge = 1..Ny-1 ---
    for i in range(Nx):
        for j_edge in range(1, Ny_edges-1):
            jB = j_edge - 1
            jT = j_edge
            phi_f = 0.5 * (phi_acc[i, jB] + phi_acc[i, jT])
            Jf = Jy_face[i, j_edge]  # net current in +y

            # inflow to TOP (eta>0), mu_n=eta
            a = phi_f / omega_total
            b = Jf / (M2y + 1e-300)
            for k in np.where(eta > 0)[0]:
                psi_yedge[i, j_edge, k, :] = a + b * eta[k]

            # inflow to BOTTOM (eta<0), outward normal -y so Jn_out = -Jf, mu_n = -eta
            bB = (-Jf) / (M2y + 1e-300)
            for k in np.where(eta < 0)[0]:
                psi_yedge[i, j_edge, k, :] = a + bB * (-eta[k])

    # --- enforce boundary inflow from bc (as you already do) ---
    def as_dir_array(val):
        val = np.asarray(val)
        if val.shape == ():
            return val * np.ones(N_dir)
        assert val.shape == (N_dir,)
        return val

    bcL = as_dir_array(bc.get("left", 0.0))
    bcR = as_dir_array(bc.get("right", 0.0))
    bcB = as_dir_array(bc.get("bottom", 0.0))
    bcT = as_dir_array(bc.get("top", 0.0))

    for k in np.where(mu > 0)[0]:
        psi_xedge[0, :, k, :] = bcL[k]
    for k in np.where(mu < 0)[0]:
        psi_xedge[-1, :, k, :] = bcR[k]
    for k in np.where(eta > 0)[0]:
        psi_yedge[:, 0, k, :] = bcB[k]
    for k in np.where(eta < 0)[0]:
        psi_yedge[:, -1, k, :] = bcT[k]

    return psi_xedge, psi_yedge




def smm(
    psi, psi_xedge, psi_yedge,
    q_cell, sig_t, sig_s, dx, dy, mu, eta, w, bc, update="yavuz"
):
    phi_half, Qxx, Qxy = compute_cell_moments_inplane(psi, mu, eta, w)

    sig_a = sig_t - sig_s
    if np.any(sig_a < 0.0):
        mn = sig_a.min()
        raise ValueError(f"sig_a < 0 somewhere (min {mn}); need sig_s <= sig_t everywhere.")

    ddQ = div_div_Q_inplane(Qxx, Qxy, dx, dy)
    s_sm = ddQ / (sig_t + 1e-300)

    Ux, Uy = div_Q_vector_inplane(Qxx, Qxy, dx, dy)

    Jm = compute_incident_partial_currents_2d_inplane(bc, mu, eta, w)

    omega_total = np.sum(w)

    M1x = np.sum(w[mu > 0] * mu[mu > 0])
    M1y = np.sum(w[eta > 0] * eta[eta > 0])

    alpha_x = 2.0 * M1x / omega_total
    alpha_y = 2.0 * M1y / omega_total

    Nx, Ny = q_cell.shape

    g_left   = (-2.0 * Jm["left"])   + (Ux[0, :] / (sig_t[0, :] + 1e-300))
    g_right  = (-2.0 * Jm["right"])  - (Ux[Nx-1, :] / (sig_t[Nx-1, :] + 1e-300))
    g_bottom = (-2.0 * Jm["bottom"]) + (Uy[:, 0] / (sig_t[:, 0] + 1e-300))
    g_top    = (-2.0 * Jm["top"])    - (Uy[:, Ny-1] / (sig_t[:, Ny-1] + 1e-300))

    robin = dict(
        left   = dict(alpha=alpha_x, g=g_left),
        right  = dict(alpha=alpha_x, g=g_right),
        bottom = dict(alpha=alpha_y, g=g_bottom),
        top    = dict(alpha=alpha_y, g=g_top),
    )

    D0 = 1.0 / (2.0 * (sig_t + 1e-300))
    rhs = q_cell + s_sm
    phi_acc = solve_diffusion(rhs, sig_a, D0, dx, dy, robin)

    if update == "rescale":
        eps = 1e-14
        scale = phi_acc / (phi_half + eps)
        scale = np.maximum(scale, 0.0)
        psi *= scale[:, :, None, None]

        # Recompute edges after acceleration
        psi_xedge, psi_yedge = compute_edge_fluxes(psi, mu, eta, bc)

    elif update == "yavuz":
        # moments of current (half) iterate
        phi_half2, Jx_half, Jy_half = compute_cell_moments(psi, mu, eta, w)

        gx, gy = grad_center(phi_acc, dx, dy)
        Jx_acc = -(0.5*gx + Ux) / (sig_t + 1e-300)
        Jy_acc = -(0.5*gy + Uy) / (sig_t + 1e-300)

        dphi = phi_acc - phi_half2
        dJx  = Jx_acc  - Jx_half
        dJy  = Jy_acc  - Jy_half

        # --- under-relaxation (critical) ---
        omega = 1.0   # start 0.3–0.7; 0.5 is a good default

        inv2pi = 1.0/(2.0*np.pi)
        invpi  = 1.0/np.pi

        # apply P1 correction directly to cell angular flux
        for k in range(len(w)):
            delta = inv2pi * dphi + invpi * (mu[k]*dJx + eta[k]*dJy)  # (Nx,Ny)
            psi[:, :, k, :] += omega * delta[:, :, None]             # add to all 4 corners

        # moments of current HO iterate (optional; mostly for diagnostics now)
        phi_half2, Jx_half, Jy_half = compute_cell_moments(psi, mu, eta, w)

        # LO face currents consistent with diffusion stencil
        Jx_face, Jy_face = diffusion_face_currents(phi_acc, D0, dx, dy, robin)

        # HOLO coupling: overwrite interior inflow on faces using LO (phi, Jn) on faces
        psi_xedge, psi_yedge = holo_set_interior_inflow_from_LO(
            psi_xedge, psi_yedge, mu, eta, w, bc,
            phi_acc, Jx_face, Jy_face
        )

    return psi, psi_xedge, psi_yedge, phi_half, phi_acc


# -----------------------------
# OCI transport solve (2D) + SMA
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

    if sig_t == None:
        sig_t = sig_t_val * np.ones((Nx, Ny))
    if sig_s == None:
        assert (sig_s_val < sig_t).all
        sig_s = sig_s_val * np.ones((Nx, Ny))
    if q == None:
        q_cell = q_val * np.ones((Nx, Ny))

    mu, eta, w, omega_total = make_circle_quadrature(N_dir)

    psi = np.zeros((Nx, Ny, N_dir, 4))
    psi_last = psi.copy()

    psi_xedge, psi_yedge = compute_edge_fluxes(psi, mu, eta, bc)

    err_last = 1.0
    for it in range(max_it):
        # --- transport half-step (OCI) ---
        for i in range(Nx):
            for j in range(Ny):
                A = build_cell_matrix(dx, dy, sig_t[i, j], sig_s[i, j], mu, eta, w, omega_total)
                b = build_cell_rhs(i, j, dx, dy, q_cell[i, j], mu, eta, w, omega_total,
                                   psi_xedge, psi_yedge)
                xloc = np.linalg.solve(A, b)
                psi[i, j, :, :] = xloc.reshape(N_dir, 4)

        # Update edges by upwind closure (consistent “half iterate”)
        psi_xedge, psi_yedge = compute_edge_fluxes(psi, mu, eta, bc)

        # --- second-moment acceleration step (optional) ---
        if smm_acc and sig_s_val > 0.0:
            psi, psi_xedge, psi_yedge, phi_half, phi_acc = smm(
            psi=psi,
            psi_xedge=psi_xedge,
            psi_yedge=psi_yedge,
            q_cell=q_cell,
            sig_t=sig_t,
            sig_s=sig_s,
            dx=dx, dy=dy,
            mu=mu, eta=eta, w=w,
            bc=bc,
            update=update,   # or whatever you want
        )

        # Convergence check (on accelerated iterate)
        err = np.linalg.norm((psi - psi_last).ravel(), ord=2)
        rho = err / err_last if it > 0 else np.nan

        if printer:
            print(f"it {it:4d}  err {err:.3e}  ρ {rho:.5f}")

        if it > 2 and err < tol * (1.0 - min(max(rho, 0.0), 0.999999)):
            break

        psi_last[:] = psi
        err_last = max(err, 1e-300)

    # Scalar flux (cell-average)
    psi_avg = np.mean(psi, axis=3)                 # (Nx,Ny,N_dir)
    phi = np.tensordot(psi_avg, w, axes=([2],[0])) # (Nx,Ny)

    angles = dict(mu=mu, eta=eta, w=w)
    mesh = dict(x=x, y=y, X=X, Y=Y, dx=dx, dy=dy)

    return phi, psi, angles, mesh, rho, it


if __name__ == "__main__":

    q = 2.0
    sigma = 5.0
    c = 0.7
    sigma_s = sigma*c
    inf_homo = q / ( sigma*(1-c) )
    inf_homo /= (2*np.pi)
    
    bc = dict(left=inf_homo, right=inf_homo, bottom=inf_homo, top=inf_homo)

    phi, psi, ang, mesh, rho, it = transport_2d_oci(
        Nx=30, Ny=30, Lx=1.0, Ly=1.0,
        N_dir=8,
        sig_t_val=sigma, sig_s_val=sigma_s,
        q_val=q,
        bc=bc,
        tol=1e-6, max_it=500, printer=True,
        smm_acc=True, update="yavuz"
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
