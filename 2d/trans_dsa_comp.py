import numpy as np
import matplotlib.pyplot as plt

import smom_2d as sn  # <- your module that contains make_circle_quadrature, solve_lo, transport_2d_oci, etc.


def build_yavuz_plane_walls_2d(Lx=4.0, Ly=2.0, dx=0.02, dy=0.02):
    """
    Plane walls in x: regions [0,1), [1,2), [2,3), [3,4].
    Fields are constant in y (depends only on x).
    Returns: x, y, dx_eff, dy_eff, sig_t, sig_s, q
    """
    Nx = int(round(Lx / dx))
    Ny = int(round(Ly / dy))
    dx = Lx / Nx
    dy = Ly / Ny

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    sig_t = np.ones((Nx, Ny))
    sig_s = np.ones((Nx, Ny))

    r1 = (x < 1.0)
    r2 = (x >= 1.0) & (x < 2.0)
    r3 = (x >= 2.0) & (x < 3.0)
    r4 = (x >= 3.0)

    sig_s[r1, :] = 1.00
    sig_s[r2, :] = 0.95
    sig_s[r3, :] = 0.80
    sig_s[r4, :] = 0.95

    q = np.zeros((Nx, Ny))
    q[r2, :] = 1.0
    q[r3, :] = 1.0

    return x, y, dx, dy, sig_t, sig_s, q


def build_robin_from_quadrature(N_dir, bc):
    """
    Matches your SMM setup:
      alpha_x = 2*M1x/omega_total, alpha_y = 2*M1y/omega_total
      g = -2*Jm, where Jm is incident partial current from bc.
    For vacuum bc (all zeros), g=0 automatically.

    Returns robin dict with g arrays sized for solve_lo:
      left/right: Ny, bottom/top: Nx
    """
    mu, eta, w, omega_total = sn.make_circle_quadrature(N_dir)

    # If you want EXACTLY what your SMM computes, reuse its helper:
    Jm = sn.compute_incident_partial_currents_2d_inplane(bc, mu, eta, w)

    M1x = np.sum(w[mu > 0] * mu[mu > 0])
    M1y = np.sum(w[eta > 0] * eta[eta > 0])

    alpha_x = 2.0 * M1x / omega_total
    alpha_y = 2.0 * M1y / omega_total

    return alpha_x, alpha_y, Jm


def main():
    # ----------------------------
    # Problem / mesh
    # ----------------------------
    Lx, Ly = 4.0, 2.0
    dx, dy = 0.05, 0.05  # keep modest
    x, y, dx_eff, dy_eff, sig_t, sig_s, q = build_yavuz_plane_walls_2d(Lx=Lx, Ly=Ly, dx=dx, dy=dy)
    Nx, Ny = q.shape
    print(f"Nx={Nx}, Ny={Ny}, dx={dx_eff:.6f}, dy={dy_eff:.6f}")

    # ----------------------------
    # BC choice
    # ----------------------------
    # Vacuum on all sides (matches your transport input deck)
    bc = dict(left=0.0, right=0.0, bottom=0.0, top=0.0)

    # If you want "infinite plane" in y, use reflective (Jy=0) on top/bottom:
    # (keep vacuum left/right)
    use_reflective_y = False

    # Quadrature ONLY used to set Marshak alpha consistently with your SMM code.
    N_dir = 32
    alpha_x, alpha_y, Jm = build_robin_from_quadrature(N_dir, bc)

    # Build LO robin g = -2 Jm (constant in space here), with correct array shapes
    g_left   = (-2.0 * Jm["left"])   * np.ones(Ny)
    g_right  = (-2.0 * Jm["right"])  * np.ones(Ny)
    g_bottom = (-2.0 * Jm["bottom"]) * np.ones(Nx)
    g_top    = (-2.0 * Jm["top"])    * np.ones(Nx)

    if use_reflective_y:
        # Reflective in y => Jy=0 at y-boundaries in your current-form Robin:
        # +Jy + alpha*phi = g, so set alpha=0 and g=0 to enforce Jy=0.
        alpha_y = 0.0
        g_bottom[:] = 0.0
        g_top[:] = 0.0

    robin = dict(
        left   = dict(alpha=alpha_x, g=g_left),
        right  = dict(alpha=alpha_x, g=g_right),
        bottom = dict(alpha=alpha_y, g=g_bottom),
        top    = dict(alpha=alpha_y, g=g_top),
    )

    # ----------------------------
    # LO coefficients and U=0
    # ----------------------------
    sig_a = sig_t - sig_s
    if np.any(sig_a < -1e-14):
        raise ValueError(f"Found sig_a < 0 (min={sig_a.min():.3e}). Need sig_s <= sig_t everywhere.")

    # U on faces is identically zero (what you asked for)
    Ux_xface = np.zeros((Nx + 1, Ny))
    Uy_yface = np.zeros((Nx, Ny + 1))

    # ----------------------------
    # LO-only solve
    # ----------------------------
    lo_cache = None
    phi_lo, Jx_face_lo, Jy_face_lo, lo_cache = sn.solve_lo(
        source=q,
        sig_a=sig_a,
        sig_t=sig_t,
        Ux_xface=Ux_xface,
        Uy_yface=Uy_yface,
        dx=dx_eff,
        dy=dy_eff,
        robin=robin,
        cache=lo_cache,
        use_harmonic_sig_t=True,
    )

    rhos = np.array([])
    its = np.array([])

    print("here")

    phi_second_simple, psi_second_simple, ang, mesh, rho, it = sn.transport_2d_oci(
        sig_t=sig_t, sig_s=sig_s, q=q,
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        N_dir=N_dir,
        bc=bc,
        tol=1e-6, max_it=1000, printer=True,
        smm_acc=True,
        update="yavuz",
        diff="second_moment",     # or "diffusion"
        closure="simple_upwind",
    )

    rhos = np.append(rhos, rho)
    its = np.append(its, it)

    phi_second_additive, psi_second_additive, ang, mesh, rho, it = sn.transport_2d_oci(
        sig_t=sig_t, sig_s=sig_s, q=q,
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        N_dir=N_dir,
        bc=bc,
        tol=1e-6, max_it=1000, printer=True,
        smm_acc=True,
        update="yavuz",
        diff="second_moment",     # or "diffusion"
        closure="additive",
    )

    rhos = np.append(rhos, rho)
    its = np.append(its, it)

    phi_diffusion_simple, psi_diffusion_simple, ang, mesh, rho, it = sn.transport_2d_oci(
        sig_t=sig_t, sig_s=sig_s, q=q,
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        N_dir=N_dir,
        bc=bc,
        tol=1e-6, max_it=1000, printer=True,
        smm_acc=True,
        update="yavuz",
        diff="diffusion",     # or "diffusion"
        closure="simple_upwind",
    )
    rhos = np.append(rhos, rho)
    its = np.append(its, it)

    phi_diffusion_additive, psi_diffusion_additive, ang, mesh, rho, it = sn.transport_2d_oci(
        sig_t=sig_t, sig_s=sig_s, q=q,
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        N_dir=N_dir,
        bc=bc,
        tol=1e-6, max_it=1000, printer=True,
        smm_acc=True,
        update="yavuz",
        diff="diffusion",     # or "diffusion"
        closure="additive",
    )

    rhos = np.append(rhos, rho)
    its = np.append(its, it)

    phi_transport, psi_transport, ang, mesh, rho, it = sn.transport_2d_oci(
        sig_t=sig_t, sig_s=sig_s, q=q,
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        N_dir=N_dir,
        bc=bc,
        tol=1e-6, max_it=1000, printer=True,
        smm_acc=False,
        update="yavuz",
        diff="second_moment",     # or "diffusion"
        closure="additive",
    )

    #rhos = np.append(rhos, rho)
    #its = np.append(its, it)

    names = ["second upwind", " second additive", "diffusion simple", "diffusion additive", "transport"]

    np.savez("yavuz_slab_S64", phi_lo=phi_lo,
                           phi_second_simple=phi_second_simple, psi_second_simple=psi_second_simple,
                           phi_second_additive=phi_second_additive, psi_second_additive=psi_second_additive,
                           phi_diffusion_simple=phi_diffusion_simple, psi_diffusion_simple=psi_diffusion_simple,
                           phi_diffusion_additive=phi_diffusion_additive, psi_diffusion_additive=psi_diffusion_additive,
                           phi_transport=phi_transport, psi_transport=psi_transport,
                           ang=ang, mesh=mesh, rhos=rhos, its=its, names=names
                           )


if __name__ == "__main__":
    main()
