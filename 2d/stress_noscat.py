import numpy as np
import smom_2d as sn

# ----------------------------
# Norms and diagnostics
# ----------------------------
def l2_cell_norm(f, dx, dy):
    return np.sqrt(np.sum(f*f) * dx * dy)

def l2_face_norm(Jx_face, Jy_face, dx, dy):
    # Vertical faces: integrate over face length dy
    # Horizontal faces: integrate over face length dx
    return np.sqrt(np.sum(Jx_face*Jx_face) * dy + np.sum(Jy_face*Jy_face) * dx)

def reaction_rate_absorption(phi, sig_a, dx, dy):
    return float(np.sum(sig_a * phi) * dx * dy)

def angular_l2_norm(psi, w, dx, dy):
    # Use corner-average angular flux (consistent with moment routines)
    psi_avg = psi.mean(axis=3)  # (Nx,Ny,N_dir)
    return np.sqrt(np.sum((psi_avg * psi_avg) * w[None, None, :]) * dx * dy)

def compute_face_currents_from_solution(psi, mu, eta, w, bc):
    psi_xedge, psi_yedge = sn.compute_edge_fluxes(psi, mu, eta, bc)
    psi_x = psi_xedge.mean(axis=3)  # (Nx+1,Ny,N_dir)
    psi_y = psi_yedge.mean(axis=3)  # (Nx,Ny+1,N_dir)

    Jx_face = np.tensordot(psi_x, w * mu, axes=([2], [0]))   # (Nx+1,Ny)
    Jy_face = np.tensordot(psi_y, w * eta, axes=([2], [0]))  # (Nx,Ny+1)
    return Jx_face, Jy_face

def rel_err(num, den, eps=1e-300):
    return float(num / (den + eps))

def make_beam_bc(mu, eta, w, strength=1.0, theta0=0.0, half_angle=0.10):
    """
    Direction-dependent inflow on LEFT only, vacuum elsewhere.
    Beam is centered at angle theta0 (radians), with a top-hat angular width.
    """
    theta = np.arctan2(eta, mu)  # (-pi,pi]
    dtheta = np.angle(np.exp(1j*(theta - theta0)))  # wrapped difference

    bcL = np.zeros_like(mu)
    inflow = (mu > 0) & (np.abs(dtheta) <= half_angle)
    bcL[inflow] = strength

    return dict(left=bcL, right=0.0, bottom=0.0, top=0.0)

# ----------------------------
# Stress cases
# ----------------------------
def case_beam_void_channel():
    """
    Beam + vacuum + void channel + high-scatter background.
    Stresses: boundary layer, ray effects, near-void, HOLO closure consistency.
    """
    Lx, Ly = 4.0, 2.0
    Nx, Ny = 20, 10
    dx, dy = Lx / Nx, Ly / Ny
    N_dir = 8

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Background: very scattering, weak absorption
    sig_t = 10.0 * np.ones((Nx, Ny))
    c = 0.9995
    sig_s = (c * sig_t)

    # Void channel across the domain
    ch = (np.abs(Y - 1.0) < 0.15)
    sig_t[ch] = 1e-3
    sig_s[ch] = 0.0

    # Add a thin strong absorber stripe to create a sharp layer
    stripe = (np.abs(X - 2.0) < 0.025)
    sig_t[stripe] = 200.0
    sig_s[stripe] = 0.0

    q = np.ones((Nx, Ny))

    mu, eta, w, _ = sn.make_circle_quadrature(N_dir)
    bc = make_beam_bc(mu, eta, w, strength=5.0, theta0=0.0, half_angle=0.08)

    return dict(name="beam_void_channel",
                Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, N_dir=N_dir,
                sig_t=sig_t, sig_s=sig_s, q=q, bc=bc)

def case_checkerboard_void_scatter():
    """
    Checkerboard near-void vs high-scatter blocks.
    Stresses: strong discontinuities, near-void, interface currents, P1 closure on faces.
    """
    Lx, Ly = 2.0, 2.0
    Nx, Ny = 10, 10
    dx, dy = Lx / Nx, Ly / Ny
    N_dir = 4

    sig_t = np.zeros((Nx, Ny))
    sig_s = np.zeros((Nx, Ny))

    # Block size
    bs = 5
    for i in range(Nx):
        for j in range(Ny):
            bi, bj = i // bs, j // bs
            if (bi + bj) % 2 == 0:
                # High-scatter, optically thick
                sig_t[i, j] = 50.0
                sig_s[i, j] = 49.95
            else:
                # Near void
                sig_t[i, j] = 1e-4
                sig_s[i, j] = 0.0

    # Internal source in a disk to avoid trivial vacuum solution
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    r2 = (X - Lx/2)**2 + (Y - Ly/2)**2
    q = np.zeros((Nx, Ny))
    q[r2 < (0.35**2)] = 5.0

    bc = dict(left=0.0, right=0.0, bottom=0.0, top=0.0)

    return dict(name="checkerboard_void_scatter",
                Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, N_dir=N_dir,
                sig_t=sig_t, sig_s=sig_s, q=q, bc=bc)

def case_plane_walls_extreme():
    """
    Yavuz-style plane walls, but with more extreme material jumps and a vacuum gap.
    Stresses: discontinuities, streaming across gaps, diffusion-like LO stress.
    """
    Lx, Ly = 4.0, 2.0
    Nx, Ny = 12, 6
    dx, dy = Lx / Nx, Ly / Ny
    N_dir = 8

    x = (np.arange(Nx) + 0.5) * dx

    sig_t = np.ones((Nx, Ny))
    sig_s = np.zeros((Nx, Ny))
    q     = np.zeros((Nx, Ny))

    r1 = (x < 1.0)
    r2 = (x >= 1.0) & (x < 2.0)
    r3 = (x >= 2.0) & (x < 3.0)
    r4 = (x >= 3.0)

    # Left wall: thick scatterer
    sig_t[r1, :] = 50.0
    sig_s[r1, :] = 49.95

    # Middle left: near-void gap
    sig_t[r2, :] = 2e-4
    sig_s[r2, :] = 0.0

    # Middle right: strong source in moderate scatterer
    sig_t[r3, :] = 5.0
    sig_s[r3, :] = 4.95
    q[r3, :]     = 2.0

    # Right wall: absorber (kills diffusion closure quality)
    sig_t[r4, :] = 200.0
    sig_s[r4, :] = 0.0

    bc = dict(left=0.0, right=0.0, bottom=0.0, top=0.0)

    return dict(name="plane_walls_extreme",
                Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, N_dir=N_dir,
                sig_t=sig_t, sig_s=sig_s, q=q, bc=bc)

# ----------------------------
# Runner
# ----------------------------
def run_transport(case, smm_acc, ho_scatter, tol=1e-6, max_it=2000,
                  update="yavuz", diff="second_moment", closure="additive"):
    old = sn.global_flag_transport_scattering
    try:
        sn.global_flag_transport_scattering = bool(ho_scatter)
        phi, psi, ang, mesh, rho, it = sn.transport_2d_oci(
            Nx=case["Nx"], Ny=case["Ny"], Lx=case["Lx"], Ly=case["Ly"], N_dir=case["N_dir"],
            sig_t=case["sig_t"], sig_s=case["sig_s"], q=case["q"], bc=case["bc"],
            tol=tol, max_it=max_it, printer=True,
            smm_acc=bool(smm_acc),
            update=update, diff=diff, closure=closure,
        )
    finally:
        sn.global_flag_transport_scattering = old
    return phi, psi, ang, mesh, rho, it

def compare_to_ref(case, ref, test, ang, mesh):
    dx, dy = mesh["dx"], mesh["dy"]
    mu, eta, w = ang["mu"], ang["eta"], ang["w"]

    phi_ref, psi_ref = ref
    phi_tst, psi_tst = test

    # Face currents (from HO angular flux on faces)
    Jx_ref, Jy_ref = compute_face_currents_from_solution(psi_ref, mu, eta, w, case["bc"])
    Jx_tst, Jy_tst = compute_face_currents_from_solution(psi_tst, mu, eta, w, case["bc"])

    # Reaction rates (absorption)
    sig_a = case["sig_t"] - case["sig_s"]
    Ra_ref = reaction_rate_absorption(phi_ref, sig_a, dx, dy)
    Ra_tst = reaction_rate_absorption(phi_tst, sig_a, dx, dy)

    # Norms and relative errors (explicit normalization)
    e_phi = l2_cell_norm(phi_tst - phi_ref, dx, dy)
    n_phi = l2_cell_norm(phi_ref, dx, dy)

    e_J = l2_face_norm(Jx_tst - Jx_ref, Jy_tst - Jy_ref, dx, dy)
    n_J = l2_face_norm(Jx_ref, Jy_ref, dx, dy)

    e_psi = angular_l2_norm(psi_tst - psi_ref, w, dx, dy)
    n_psi = angular_l2_norm(psi_ref, w, dx, dy)

    # Negativity indicators
    min_psi = float(np.min(psi_tst))
    min_phi = float(np.min(phi_tst))

    return dict(
        rel_L2_phi=rel_err(e_phi, n_phi),
        rel_L2_face_current=rel_err(e_J, n_J),
        rel_L2_psi=rel_err(e_psi, n_psi),
        rel_abs_rxn_rate=rel_err(abs(Ra_tst - Ra_ref), abs(Ra_ref)),
        min_phi=min_phi,
        min_psi=min_psi,
        Ra_ref=Ra_ref,
        Ra_tst=Ra_tst,
    )

def main():
    cases = [
        case_beam_void_channel(),
        #case_checkerboard_void_scatter(),
        #case_plane_walls_extreme(),
    ]

    for case in cases:
        print("\n" + "="*80)
        print("CASE:", case["name"])
        print("="*80)

        # Reference: transport with HO scattering ON, no acceleration
        print("\n[REF] HO scatter ON, SMM OFF")
        phi_ref, psi_ref, ang, mesh, rho_ref, it_ref = run_transport(
            case, smm_acc=False, ho_scatter=True
        )

        # Accelerated consistent: HO scattering ON, SMM ON
        print("\n[ACC] HO scatter ON, SMM ON")
        phi_acc, psi_acc, ang2, mesh2, rho_acc, it_acc = run_transport(
            case, smm_acc=True, ho_scatter=True
        )

        # Stressed: HO scattering OFF, SMM ON (your requested stress mode)
        print("\n[STRESS] HO scatter OFF, SMM ON")
        phi_str, psi_str, ang3, mesh3, rho_str, it_str = run_transport(
            case, smm_acc=True, ho_scatter=False
        )

        # Compare to reference (same physical equation only for ACC, not for STRESS)
        acc_metrics = compare_to_ref(case, (phi_ref, psi_ref), (phi_acc, psi_acc), ang, mesh)
        str_metrics = compare_to_ref(case, (phi_ref, psi_ref), (phi_str, psi_str), ang, mesh)

        # Print summary
        def print_metrics(tag, m):
            print(f"\n{tag} metrics vs REF (normalized):")
            print(f"  rel L2(phi)            = {m['rel_L2_phi']:.3e}")
            print(f"  rel L2(face currents)  = {m['rel_L2_face_current']:.3e}")
            print(f"  rel L2(psi)            = {m['rel_L2_psi']:.3e}")
            print(f"  rel abs rxn rate       = {m['rel_abs_rxn_rate']:.3e}")
            print(f"  min(phi), min(psi)     = {m['min_phi']:.3e}, {m['min_psi']:.3e}")
            print(f"  Ra_ref, Ra_test        = {m['Ra_ref']:.6e}, {m['Ra_tst']:.6e}")
        
        print_metrics("[ACC]", acc_metrics)
        print_metrics("[STRESS]", str_metrics)

        def plot_sf(phi_set):
            import matplotlib.pyplot as plt
            
            N, M = phi_set[0].shape
            for i, phi in enumerate(phi_set):
                assert (np.array([N,M]) == phi.shape).all()

            P = len(phi_set)

            x = np.linspace(0, M - 1, M)
            y = np.linspace(0, N - 1, N)
            X, Y = np.meshgrid(x,y)

            fig, axs = plt.subplots(1, P, sharey=True)

            # Plot each country on its own subplot
            for i, ax in enumerate(axs):
                ax.contourf(X, Y, phi_set[i])

            plt.show()

        plot_sf([phi_ref, phi_acc, phi_str])

        

if __name__ == "__main__":
    main()