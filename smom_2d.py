import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 2D Discrete Ordinates (in-plane)
# -----------------------------
def make_circle_quadrature(N_dir: int):
    """
    Simple 2D (in-plane) angular quadrature on the unit circle:
      theta_k = 2*pi*(k+0.5)/N_dir, w_k = 2*pi/N_dir
    Returns mu_x, mu_y, weights, omega_total
    """
    theta = 2.0 * np.pi * (np.arange(N_dir) + 0.5) / N_dir
    mu = np.cos(theta)   # x-direction cosine
    eta = np.sin(theta)  # y-direction cosine
    w = (2.0 * np.pi / N_dir) * np.ones(N_dir)
    omega_total = np.sum(w)  # = 2*pi
    return mu, eta, w, omega_total


# Corner ordering in each cell:
# 0: SW (xL,yB)
# 1: SE (xR,yB)
# 2: NW (xL,yT)
# 3: NE (xR,yT)

# -----------------------------
# Corner-balance local operator
# -----------------------------
def A_dir_corner_balance(mu, eta, dx, dy, sig_t):
    """
    4x4 spatial streaming+absorption matrix for ONE direction in ONE cell,
    using a simple tensor-product corner-balance / LD form.

    Unknowns per direction: [SW, SE, NW, NE] (x varies fastest inside y blocks)
    """
    hx = dx / 2.0
    hy = dy / 2.0
    area = dx * dy

    # "Streaming-only" 1D-style pieces (no absorption here)
    Ax = hy * np.array([[abs(mu)/2.0,  mu/2.0],
                        [-mu/2.0,     abs(mu)/2.0]])

    Ay = hx * np.array([[abs(eta)/2.0,  eta/2.0],
                        [-eta/2.0,     abs(eta)/2.0]])

    # Tensor product assembly:
    # - x-coupling inside each y-level block
    # - y-coupling between bottom/top blocks
    A = np.kron(np.eye(2), Ax) + np.kron(Ay, np.eye(2))

    # Absorption/total interaction term (lumped to each corner equation)
    A += (sig_t * area / 4.0) * np.eye(4)
    return A


def build_cell_matrix(dx, dy, sig_t, sig_s, mu, eta, w, omega_total):
    """
    Build the full (4*N_dir) x (4*N_dir) matrix for a single cell.
    This is the 2D analog of your buildA(i): direction blocks + isotropic scattering coupling.
    """
    N_dir = len(w)
    n = 4 * N_dir
    A = np.zeros((n, n))

    # Direction block-diagonal: each direction has a 4x4 spatial block
    for k in range(N_dir):
        Ak = A_dir_corner_balance(mu[k], eta[k], dx, dy, sig_t)
        A[4*k:4*k+4, 4*k:4*k+4] = Ak

    # Isotropic scattering couples directions within the SAME corner DOF
    # Generalizing your 1D Sbuild:
    #   term ~ (sig_s * volume_corner / Omega) * w[j]
    beta = (sig_s * (dx*dy) / 4.0) / omega_total

    for c in range(4):  # corner dof
        row_idx = np.arange(N_dir) * 4 + c
        col_idx = np.arange(N_dir) * 4 + c
        # A[row, col] -= beta * w[col_dir]
        # i.e., each row gets the same weighted sum over directions
        A[np.ix_(row_idx, col_idx)] -= beta * w[None, :]

    return A


# -----------------------------
# Upwind edge closure (2D analog of compute_aflux_edge)
# -----------------------------
def compute_edge_fluxes(psi_cell, mu, eta, bc):
    """
    psi_cell: (Nx, Ny, N_dir, 4)
    Returns:
      psi_xedge: (Nx+1, Ny, N_dir, 2)  # vertical edges, two y-corners: [bottom, top]
      psi_yedge: (Nx, Ny+1, N_dir, 2)  # horizontal edges, two x-corners: [left, right]
    bc is a dict with constant inflow values:
      bc['left'], bc['right'], bc['bottom'], bc['top'] (each can be scalar or array(N_dir))
    """
    Nx, Ny, N_dir, _ = psi_cell.shape

    # Expand BCs to arrays per direction
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

    # Vertical edges (x-edges): i_edge = 0..Nx, between (i_edge-1) and (i_edge)
    for i_edge in range(Nx+1):
        for j in range(Ny):
            for k in range(N_dir):
                if i_edge == 0:
                    # left boundary: inflow if mu>0
                    psi_xedge[i_edge, j, k, :] = bcL[k]
                elif i_edge == Nx:
                    # right boundary: inflow if mu<0
                    psi_xedge[i_edge, j, k, :] = bcR[k]
                else:
                    # interior: upwind closure
                    if mu[k] > 0:
                        # from left cell (i_edge-1): take its east corners (SE, NE)
                        iL = i_edge - 1
                        psi_xedge[i_edge, j, k, 0] = psi_cell[iL, j, k, 1]  # SE -> bottom
                        psi_xedge[i_edge, j, k, 1] = psi_cell[iL, j, k, 3]  # NE -> top
                    else:
                        # from right cell (i_edge): take its west corners (SW, NW)
                        iR = i_edge
                        psi_xedge[i_edge, j, k, 0] = psi_cell[iR, j, k, 0]  # SW -> bottom
                        psi_xedge[i_edge, j, k, 1] = psi_cell[iR, j, k, 2]  # NW -> top

    # Horizontal edges (y-edges): j_edge = 0..Ny, between (j_edge-1) and (j_edge)
    for i in range(Nx):
        for j_edge in range(Ny+1):
            for k in range(N_dir):
                if j_edge == 0:
                    # bottom boundary: inflow if eta>0
                    psi_yedge[i, j_edge, k, :] = bcB[k]
                elif j_edge == Ny:
                    # top boundary: inflow if eta<0
                    psi_yedge[i, j_edge, k, :] = bcT[k]
                else:
                    if eta[k] > 0:
                        # from bottom cell (j_edge-1): take its north corners (NW, NE)
                        jB = j_edge - 1
                        psi_yedge[i, j_edge, k, 0] = psi_cell[i, jB, k, 2]  # NW -> left
                        psi_yedge[i, j_edge, k, 1] = psi_cell[i, jB, k, 3]  # NE -> right
                    else:
                        # from top cell (j_edge): take its south corners (SW, SE)
                        jT = j_edge
                        psi_yedge[i, j_edge, k, 0] = psi_cell[i, jT, k, 0]  # SW -> left
                        psi_yedge[i, j_edge, k, 1] = psi_cell[i, jT, k, 1]  # SE -> right

    return psi_xedge, psi_yedge


# -----------------------------
# RHS builder (2D analog of buildb)
# -----------------------------
def build_cell_rhs(i, j, dx, dy, q_cell, mu, eta, w, omega_total,
                   psi_xedge, psi_yedge):
    """
    Returns b of length (4*N_dir) for cell (i,j).
    Isotropic fixed source q_cell(i,j) is treated like your 1D: q/Omega integrated over corner volume.
    Adds inflow contributions from the 4 boundaries of the cell for each direction.
    """
    Nx_edges = psi_xedge.shape[0]  # Nx+1
    Ny_edges = psi_yedge.shape[1]  # Ny+1
    N_dir = len(w)

    hx = dx / 2.0
    hy = dy / 2.0
    area = dx * dy

    # isotropic source per unit angle
    q_iso = q_cell / omega_total

    b = np.zeros(4 * N_dir)

    for k in range(N_dir):
        base = (area / 4.0) * q_iso
        # start with volumetric source in each corner equation
        b[4*k + 0] = base  # SW
        b[4*k + 1] = base  # SE
        b[4*k + 2] = base  # NW
        b[4*k + 3] = base  # NE

        # x-inflow
        if mu[k] > 0:
            # inflow from west edge at i_edge=i
            # ycorner 0=bottom touches SW, 1=top touches NW
            inflow_bot = psi_xedge[i, j, k, 0]
            inflow_top = psi_xedge[i, j, k, 1]
            b[4*k + 0] += mu[k] * hy * inflow_bot
            b[4*k + 2] += mu[k] * hy * inflow_top
        elif mu[k] < 0:
            # inflow from east edge at i_edge=i+1 into SE/NE
            inflow_bot = psi_xedge[i+1, j, k, 0]
            inflow_top = psi_xedge[i+1, j, k, 1]
            b[4*k + 1] += (-mu[k]) * hy * inflow_bot
            b[4*k + 3] += (-mu[k]) * hy * inflow_top

        # y-inflow
        if eta[k] > 0:
            # inflow from south edge at j_edge=j into SW/SE
            inflow_left  = psi_yedge[i, j,   k, 0]
            inflow_right = psi_yedge[i, j,   k, 1]
            b[4*k + 0] += eta[k] * hx * inflow_left
            b[4*k + 1] += eta[k] * hx * inflow_right
        elif eta[k] < 0:
            # inflow from north edge at j_edge=j+1 into NW/NE
            inflow_left  = psi_yedge[i, j+1, k, 0]
            inflow_right = psi_yedge[i, j+1, k, 1]
            b[4*k + 2] += (-eta[k]) * hx * inflow_left
            b[4*k + 3] += (-eta[k]) * hx * inflow_right

    return b



# -----------------------------
# OCI transport solve (2D)
# -----------------------------
def transport_2d_oci(
    Nx=30, Ny=30,
    Lx=3.0, Ly=3.0,
    N_dir=4,
    sig_t_val=1.0, sig_s_val=0.0,
    q_val=1.0,
    bc=None,
    tol=1e-4, max_it=2000, printer=True,
):
    if bc is None:
        bc = dict(left=0.0, right=0.0, bottom=0.0, top=0.0)

    dx = Lx / Nx
    dy = Ly / Ny

    # Mesh centers for plotting
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Materials (could be arrays Nx x Ny)
    sig_t = sig_t_val * np.ones((Nx, Ny))
    sig_s = sig_s_val * np.ones((Nx, Ny))
    sig_a = sig_t - sig_s
    D = 1.0 / (3.0 * sig_t_val) 
    q_cell = q_val * np.ones((Nx, Ny))

    # Angles/weights
    mu, eta, w, omega_total = make_circle_quadrature(N_dir)

    # Unknowns: psi_cell[i,j,k,corner]
    psi = np.random.rand(Nx, Ny, N_dir, 4) * 0.0
    psi_last = psi.copy()

    # Initial edge fluxes
    psi_xedge, psi_yedge = compute_edge_fluxes(psi, mu, eta, bc)

    err_last = 1.0
    for it in range(max_it):
        # Solve each cell with edges frozen (OCI)
        for i in range(Nx):
            for j in range(Ny):
                A = build_cell_matrix(dx, dy, sig_t[i, j], sig_s[i, j], mu, eta, w, omega_total)
                b = build_cell_rhs(i, j, dx, dy, q_cell[i, j], mu, eta, w, omega_total,
                                   psi_xedge, psi_yedge)
                x = np.linalg.solve(A, b)
                psi[i, j, :, :] = x.reshape(N_dir, 4)

        # Update edges by upwind closure
        psi_xedge, psi_yedge = compute_edge_fluxes(psi, mu, eta, bc)

        # Convergence check
        err = np.linalg.norm((psi - psi_last).ravel(), ord=2)
        rho = err / err_last if it > 0 else np.nan

        if printer:
            print(f"it {it:4d}  err {err:.3e}  ρ {rho:.5f}")

        if it > 2 and err < tol * (1.0 - min(max(rho, 0.0), 0.999999)):
            break

        psi_last[:] = psi
        err_last = max(err, 1e-300)

    # Scalar flux (cell-average) for postprocessing:
    # average over corners then integrate over directions
    psi_avg = np.mean(psi, axis=3)          # (Nx,Ny,N_dir)
    phi = np.tensordot(psi_avg, w, axes=([2],[0]))  # (Nx,Ny)

    angles = dict(mu=mu, eta=eta, w=w)
    mesh = dict(x=x, y=y, X=X, Y=Y, dx=dx, dy=dy)

    return phi, psi, angles, mesh


if __name__ == "__main__":
    bc = dict(left=0.25, right=0.0, bottom=0.1, top=0.0)  # simple inflow on left

    phi, psi, ang, mesh = transport_2d_oci(
        Nx=20, Ny=20, Lx=4.0, Ly=4.0,
        N_dir=8,
        sig_t_val=5.0, sig_s_val=4.0,
        q_val=1.0,
        bc=bc,
        tol=1e-6, max_it=500, printer=True,
    )

    x, y, X, Y = mesh["x"], mesh["y"], mesh["X"], mesh["Y"]
    # ---- Plot 1: scalar flux heatmap ----
    plt.figure()
    # phi is (Nx,Ny) in storage; transpose to (Ny,Nx) for imshow/pcolormesh with X,Y
    plt.imshow(phi.T, origin="lower",
               extent=[x.min(), x.max(), y.min(), y.max()],
               aspect="auto")
    plt.colorbar(label=r"$\phi(x,y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Corner-Balance Sn (OCI): Scalar Flux")
    plt.tight_layout()
    plt.show()


    print("phi min/max:", phi.min(), phi.max())
