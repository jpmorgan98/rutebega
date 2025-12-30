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

    beta = (sig_s * (dx*dy) / 4.0) / omega_total

    for c in range(4):
        row_idx = np.arange(N_dir) * 4 + c
        col_idx = np.arange(N_dir) * 4 + c
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


def solve_robin_diffusion(phi_rhs, sig_a, D, dx, dy, robin):
    Nx, Ny = phi_rhs.shape
    N = Nx * Ny
    A = np.zeros((N, N))
    b = phi_rhs.reshape(N).copy()  # C-order flatten consistent with idx below

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    def idx(i, j):
        # C-order flattening for (Nx, Ny): offset = i*Ny + j
        return i * Ny + j

    def robin_ab(alpha, g, h):
        # D*(phi_g - phi_c)/h + alpha*(phi_c + phi_g)/2 = g
        denom = (D / h + alpha / 2.0)
        a = (D / h - alpha / 2.0) / denom
        c = g / denom
        return a, c

    alphaL, gL = robin["left"]["alpha"], robin["left"]["g"]      # gL shape (Ny,)
    alphaR, gR = robin["right"]["alpha"], robin["right"]["g"]    # gR shape (Ny,)
    alphaB, gB = robin["bottom"]["alpha"], robin["bottom"]["g"]  # gB shape (Nx,)
    alphaT, gT = robin["top"]["alpha"], robin["top"]["g"]        # gT shape (Nx,)

    for i in range(Nx):
        for j in range(Ny):
            p = idx(i, j)

            diag = sig_a + 2.0 * D * inv_dx2 + 2.0 * D * inv_dy2

            # West (i-1) or left Robin
            if i - 1 >= 0:
                A[p, idx(i - 1, j)] = -D * inv_dx2
            else:
                a, c = robin_ab(alphaL, gL[j], dx)
                diag += (-D * inv_dx2) * a
                b[p] += (D * inv_dx2) * c

            # East (i+1) or right Robin
            if i + 1 < Nx:
                A[p, idx(i + 1, j)] = -D * inv_dx2
            else:
                a, c = robin_ab(alphaR, gR[j], dx)
                diag += (-D * inv_dx2) * a
                b[p] += (D * inv_dx2) * c

            # South (j-1) or bottom Robin
            if j - 1 >= 0:
                A[p, idx(i, j - 1)] = -D * inv_dy2
            else:
                a, c = robin_ab(alphaB, gB[i], dy)
                diag += (-D * inv_dy2) * a
                b[p] += (D * inv_dy2) * c

            # North (j+1) or top Robin
            if j + 1 < Ny:
                A[p, idx(i, j + 1)] = -D * inv_dy2
            else:
                a, c = robin_ab(alphaT, gT[i], dy)
                diag += (-D * inv_dy2) * a
                b[p] += (D * inv_dy2) * c

            A[p, p] = diag

    phi = np.linalg.solve(A, b).reshape(Nx, Ny)  # matches idx + flatten
    return phi



def compute_cell_moments_inplane(psi, mu, eta, w):
    psi_avg = np.mean(psi, axis=3)  # (Nx,Ny,N_dir)
    phi = np.tensordot(psi_avg, w, axes=([2], [0]))

    qxx_basis = (mu**2 - 0.5)
    qxy_basis = (mu * eta)

    Qxx = np.tensordot(psi_avg, w * qxx_basis, axes=([2], [0]))
    Qxy = np.tensordot(psi_avg, w * qxy_basis, axes=([2], [0]))
    return phi, Qxx, Qxy


def apply_second_moment_acceleration_with_inflow_bc(
    psi, q_cell, sig_t_val, sig_s_val, dx, dy, mu, eta, w, bc
):
    """
    Same SMA logic, but diffusion solve uses prescribed inflow via incident-partial-current Robin BC:
      D0 ∂φ/∂n + (2/π) φ = -2 J^-_inc - (1/σ_t)(div Q)·n
    """
    phi_half, Qxx, Qxy = compute_cell_moments_inplane(psi, mu, eta, w)

    sig_a_val = sig_t_val - sig_s_val
    if sig_a_val < 0:
        raise ValueError("sig_a < 0 (sig_s > sig_t) is not physical for this acceleration.")

    # Second-moment driving term (known from half-iterate)
    s_sm = (1.0 / sig_t_val) * div_div_Q_inplane(Qxx, Qxy, dx, dy)

    # Also need (div Q)·n on boundaries
    Ux, Uy = div_Q_vector_inplane(Qxx, Qxy, dx, dy)

    # Incident partial currents from prescribed inflow
    Jm = compute_incident_partial_currents_2d_inplane(bc, mu, eta, w)

    # Robin parameters
    alpha = 2.0 / np.pi
    Nx, Ny = q_cell.shape

    # g = -2 J^-_inc - (1/σ_t)(div Q)·n
    g_left   = (-2.0 * Jm["left"])   + (Ux[0, :] / sig_t_val)          # n=(-1,0)
    g_right  = (-2.0 * Jm["right"])  - (Ux[Nx-1, :] / sig_t_val)       # n=(+1,0)
    g_bottom = (-2.0 * Jm["bottom"]) + (Uy[:, 0] / sig_t_val)          # n=(0,-1)
    g_top    = (-2.0 * Jm["top"])    - (Uy[:, Ny-1] / sig_t_val)       # n=(0,+1)

    robin = dict(
        left=dict(alpha=alpha, g=g_left),
        right=dict(alpha=alpha, g=g_right),
        bottom=dict(alpha=alpha, g=g_bottom),
        top=dict(alpha=alpha, g=g_top),
    )

    # Low-order solve:
    #   -div(D0 grad φ) + σa φ = q + s_sm
    D0 = 1.0 / (2.0 * sig_t_val)
    rhs = q_cell + s_sm
    phi_acc = solve_robin_diffusion(rhs, sig_a_val, D0, dx, dy, robin)

    # Rescale angular flux to match accelerated scalar flux
    eps = 1e-14
    scale = phi_acc / (phi_half + eps)
    scale = np.maximum(scale, 0.0)
    psi *= scale[:, :, None, None]

    return psi, phi_half, phi_acc



# -----------------------------
# OCI transport solve (2D) + SMA
# -----------------------------
def transport_2d_oci(
    Nx=30, Ny=30,
    Lx=3.0, Ly=3.0,
    N_dir=4,
    sig_t_val=1.0, sig_s_val=0.0,
    q_val=1.0,
    bc=None,
    tol=1e-4, max_it=2000, printer=True,
    use_second_moment_accel=True,
):
    if bc is None:
        bc = dict(left=0.0, right=0.0, bottom=0.0, top=0.0)

    dx = Lx / Nx
    dy = Ly / Ny

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")

    sig_t = sig_t_val * np.ones((Nx, Ny))
    sig_s = sig_s_val * np.ones((Nx, Ny))
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
        if use_second_moment_accel and sig_s_val > 0.0:
            psi, phi_half, phi_acc = apply_second_moment_acceleration_with_inflow_bc(
                psi=psi,
                q_cell=q_cell,
                sig_t_val=sig_t_val,
                sig_s_val=sig_s_val,
                dx=dx, dy=dy,
                mu=mu, eta=eta, w=w,
                bc=bc
            )
            # Recompute edges after acceleration
            psi_xedge, psi_yedge = compute_edge_fluxes(psi, mu, eta, bc)

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

    return phi, psi, angles, mesh


if __name__ == "__main__":
    bc = dict(left=0.25, right=0.0, bottom=0.1, top=0.0)

    phi, psi, ang, mesh = transport_2d_oci(
        Nx=15, Ny=20, Lx=4.0, Ly=4.0,
        N_dir=8,
        sig_t_val=5.0, sig_s_val=4.0,
        q_val=1.0,
        bc=bc,
        tol=1e-6, max_it=500, printer=True,
        use_second_moment_accel=False,
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
