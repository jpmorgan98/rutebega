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


# -----------------------------
# OCI transport solve (2D) + SMA
# -----------------------------
def transport_2d_oci(
    Nx=30, Ny=30,
    Lx=3.0, Ly=3.0,
    N_dir=4,
    sig_t=None, sig_s=None, q=None,
    bc=None,
    tol=1e-4, max_it=2000, printer=True,
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
        assert np.all(sig_s_val < sig_t)
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

                # plt.figure()
                # plt.spy(A)
                # plt.show()

                b = build_cell_rhs(i, j, dx, dy, q_cell[i, j], mu, eta, w, omega_total,
                                   psi_xedge, psi_yedge)
                xloc = np.linalg.solve(A, b)
                psi[i, j, :, :] = xloc.reshape(N_dir, 4)

        # Update edges by upwind closure (consistent “half iterate”)
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

    return phi, psi, angles, mesh, rho, it


if __name__ == "__main__":
    q = 2.0
    sigma = 5
    c = 0.9
    sigma_s = sigma*c
    inf_homo = q / ( sigma*(1-c) )
    inf_homo /= (2*np.pi)
    
    bc = dict(left=inf_homo, right=inf_homo, bottom=inf_homo, top=inf_homo)

    phi, psi, ang, mesh, rho, it = transport_2d_oci(
        Nx=20, Ny=20, Lx=4.0, Ly=4.0,
        N_dir=8,
        sig_t_val=sigma, sig_s_val=sigma_s,
        q_val=q,
        bc=bc,
        tol=1e-6, max_it=500, printer=True,
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
