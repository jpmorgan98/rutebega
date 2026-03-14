
import numpy as np
import matplotlib.pyplot as plt

import smom_2d as sn


def build_yavuz_plane_walls_2d(Lx=4.0, Ly=2.0, dx=0.02, dy=0.02):
    """
    Returns:
      x (Nx,), y (Ny,), sig_t (Nx,Ny), sig_s (Nx,Ny), q (Nx,Ny)
    """
    Nx = int(round(Lx / dx))
    Ny = int(round(Ly / dy))
    dx = Lx / Nx
    dy = Ly / Ny

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    # Plane walls => fields depend only on x, constant in y
    sig_t = np.ones((Nx, Ny))
    sig_s = np.ones((Nx, Ny))

    # Region boundaries at x = 1,2,3 (since Lx=4)
    r1 = (x < 1.0)
    r2 = (x >= 1.0) & (x < 2.0)
    r3 = (x >= 2.0) & (x < 3.0)
    r4 = (x >= 3.0)

    sig_s[r1, :] = 1.00
    sig_s[r2, :] = 0.95
    sig_s[r3, :] = 0.80
    sig_s[r4, :] = 0.95

    # Source in regions 2 and 3 (middle half)
    q = np.zeros((Nx, Ny))
    q[r2, :] = 1.0
    q[r3, :] = 1.0

    return x, y, dx, dy, sig_t, sig_s, q

def main():
    # ---- knobs ----
    Lx = 4.0
    Ly = 2.0

    # Keep this modest: your solver does Nx*Ny dense solves per iter
    dx = 0.2     # try 0.01 later (it gets expensive fast)
    dy = 0.2

    N_dir = 32
    tol = 1e-6
    max_it = 1000

    # Vacuum boundaries (all sides)
    bc = dict(left=0.0, right=0.0, bottom=0.0, top=0.0)

    # Build plane-wall fields
    x, y, dx_eff, dy_eff, sig_t, sig_s, q = build_yavuz_plane_walls_2d(Lx=Lx, Ly=Ly, dx=dx, dy=dy)
    Nx, Ny = q.shape
    print(f"Nx={Nx}, Ny={Ny}, dx={dx_eff:.5f}, dy={dy_eff:.5f}")

    # Make sure transport scattering is on (it is by default in your module)
    sn.global_flag_transport_scattering = True

    # Run (start without SMM; enable once baseline works)
    phi, psi, ang, mesh, rho, it = sn.transport_2d_oci(
        sig_t=sig_t, sig_s=sig_s, q=q,
        Lx=Lx, Ly=Ly, N_dir=N_dir,
        Nx=Nx, Ny=Ny,
        bc=bc,
        tol=tol, max_it=max_it, printer=True,
        smm_acc=True,
        update="yavuz",
        diff="second_moment",
        closure="additive",
    )

    sn.global_flag_transport_scattering = False

    phi, psi_nos, ang, mesh, rho, it = sn.transport_2d_oci(
        sig_t=sig_t, sig_s=sig_s, q=q,
        Lx=Lx, Ly=Ly, N_dir=N_dir,
        Nx=Nx, Ny=Ny,
        bc=bc,
        tol=tol, max_it=max_it, printer=True,
        smm_acc=True,
        update="yavuz",
        diff="second_moment",
        closure="additive",
    )

    sn.global_flag_transport_scattering = True

    phi_t, psi_t, ang, mesh, rho, it = sn.transport_2d_oci(
        sig_t=sig_t, sig_s=sig_s, q=q,
        Lx=Lx, Ly=Ly, N_dir=N_dir,
        Nx=Nx, Ny=Ny,
        bc=bc,
        tol=tol, max_it=max_it, printer=True,
        smm_acc=False,
        update="yavuz",
        diff="diffusion",
        closure="simple_upwind",
    )

    print(np.linalg.norm(psi_nos - psi_t))
    print(np.linalg.norm(psi - psi_nos))

    print(f"\nDone: it={it}, rho~{rho}")
    print("phi min/max:", phi.min(), phi.max())

    # ---- plots ----
    x = mesh["x"]
    y = mesh["y"]

    # 2D map
    plt.figure()
    plt.imshow(phi.T, origin="lower",
               extent=[x.min(), x.max(), y.min(), y.max()],
               aspect="auto")
    plt.colorbar(label=r"$\phi(x,y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"2D Yavuz plane-walls: N_dir={N_dir}, Nx={Nx}, Ny={Ny}")
    plt.tight_layout()

    # 2D map
    plt.figure()
    plt.imshow(phi_t.T, origin="lower",
               extent=[x.min(), x.max(), y.min(), y.max()],
               aspect="auto")
    plt.colorbar(label=r"$\phi(x,y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"2D Transport plane-walls: N_dir={N_dir}, Nx={Nx}, Ny={Ny}")
    plt.tight_layout()

    # 2D map
    plt.figure()
    plt.imshow(np.abs(phi_t-phi).T, origin="lower",
               extent=[x.min(), x.max(), y.min(), y.max()],
               aspect="auto")
    plt.colorbar(label=r"$\phi(x,y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"2D Transport plane-walls: N_dir={N_dir}, Nx={Nx}, Ny={Ny}")
    plt.tight_layout()

    # Midline slice (helps compare to 1D intuition)
    jmid = Ny // 2
    plt.figure()
    plt.plot(x, phi[:, jmid], label="yavuz")
    plt.plot(x, phi_t[:, jmid], label="transport")
    plt.xlabel("x")
    plt.ylabel(r"$\phi(x, y_{mid})$")
    plt.title("Midline scalar flux slice")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()