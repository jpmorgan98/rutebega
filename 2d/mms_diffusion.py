# mms_solve_lo_diffusion.py
import sys
import types
import importlib
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CHANGE THIS to your file name (without .py)
# e.g. if you saved as sn2d.py, use MODULE_NAME="sn2d"
# -----------------------------
MODULE_NAME = "smom_2d"
tc = importlib.import_module(MODULE_NAME)

# ------------------------------------------------------------
# Helpers: face-averaged sigma_t, matching la_build_mixed choice
# ------------------------------------------------------------
def sigma_face_avg(a, b, use_harmonic=False):
    if use_harmonic:
        den = a + b
        return (2.0 * a * b / den) if den > 0 else 0.0
    return 0.5 * (a + b)

# ------------------------------------------------------------
# Build a discrete MMS case for the solve_lo diffusion operator
# ------------------------------------------------------------
def build_mms_case(
    Nx=40, Ny=35, Lx=1.0, Ly=1.0,
    sig_t_val=5.0, sig_s_val=4.5,
    alpha_x=1.0, alpha_y=1.0,
    use_harmonic_sig_t=False,
):
    dx = Lx / Nx
    dy = Ly / Ny

    # cell centers, indexing matches your (i,j) loops: phi.shape == (Nx,Ny)
    x_c = (np.arange(Nx) + 0.5) * dx
    y_c = (np.arange(Ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(x_c, y_c, indexing="ij")

    sig_t = sig_t_val * np.ones((Nx, Ny))
    sig_s = sig_s_val * np.ones((Nx, Ny))
    sig_a = sig_t - sig_s

    # Manufactured scalar flux at cell centers
    phi_ex = 1.0 + 0.25 * np.sin(2.0*np.pi*Xc/Lx) * np.sin(2.0*np.pi*Yc/Ly)

    # Allocate manufactured currents on faces
    Jx_ex = np.zeros((Nx + 1, Ny))   # vertical faces
    Jy_ex = np.zeros((Nx, Ny + 1))   # horizontal faces

    # ---- interior vertical faces: enforce discrete constitutive row exactly ----
    for i_edge in range(1, Nx):
        iL = i_edge - 1
        iR = i_edge
        for j in range(Ny):
            st_f = sigma_face_avg(sig_t[iL, j], sig_t[iR, j], use_harmonic_sig_t)
            Jx_ex[i_edge, j] = -(phi_ex[iR, j] - phi_ex[iL, j]) / (2.0 * dx * (st_f + 1e-300))

    # ---- interior horizontal faces ----
    for i in range(Nx):
        for j_edge in range(1, Ny):
            jB = j_edge - 1
            jT = j_edge
            st_f = sigma_face_avg(sig_t[i, jB], sig_t[i, jT], use_harmonic_sig_t)
            Jy_ex[i, j_edge] = -(phi_ex[i, jT] - phi_ex[i, jB]) / (2.0 * dy * (st_f + 1e-300))

    # ---- boundary faces: choose something consistent; use analytic derivative with D=1/(2*sig_t) ----
    # x-faces: x = 0 and x = Lx, y at cell centers
    y_fc = y_c
    x_left = 0.0
    x_right = Lx

    # derivative of phi(x,y) = 1 + 0.25 sin(2πx/Lx) sin(2πy/Ly)
    def dphi_dx(x, y):
        return 0.25 * (2.0*np.pi/Lx) * np.cos(2.0*np.pi*x/Lx) * np.sin(2.0*np.pi*y/Ly)

    def dphi_dy(x, y):
        return 0.25 * (2.0*np.pi/Ly) * np.sin(2.0*np.pi*x/Lx) * np.cos(2.0*np.pi*y/Ly)

    # For constant sig_t, diffusion-like J = -(1/(2*sig_t))*grad(phi)
    # If you later make sig_t spatially varying, consider evaluating sig_t at adjacent cell for boundary.
    D = 1.0 / (2.0 * sig_t_val)

    for j in range(Ny):
        Jx_ex[0,  j] = -D * dphi_dx(x_left,  y_fc[j])
        Jx_ex[Nx, j] = -D * dphi_dx(x_right, y_fc[j])

    # y-faces: y = 0 and y = Ly, x at cell centers
    x_fc = x_c
    y_bot = 0.0
    y_top = Ly
    for i in range(Nx):
        Jy_ex[i, 0 ] = -D * dphi_dy(x_fc[i], y_bot)
        Jy_ex[i, Ny] = -D * dphi_dy(x_fc[i], y_top)

    # ---- Discrete manufactured source: exactly matches your cell-balance row ----
    q_ex = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            divJ = (Jx_ex[i+1, j] - Jx_ex[i, j]) / dx + (Jy_ex[i, j+1] - Jy_ex[i, j]) / dy
            q_ex[i, j] = divJ + sig_a[i, j] * phi_ex[i, j]

    # ---- Robin data g built so boundary rows are satisfied exactly ----
    g_left   = Jx_ex[0,  :] + alpha_x * phi_ex[0,     :]
    g_right  = -Jx_ex[Nx, :] + alpha_x * phi_ex[Nx-1, :]
    g_bottom = Jy_ex[:, 0 ] + alpha_y * phi_ex[:, 0    ]
    g_top    = -Jy_ex[:, Ny] + alpha_y * phi_ex[:, Ny-1]

    robin = dict(
        left   = dict(alpha=alpha_x, g=g_left),
        right  = dict(alpha=alpha_x, g=g_right),
        bottom = dict(alpha=alpha_y, g=g_bottom),
        top    = dict(alpha=alpha_y, g=g_top),
    )

    # U fields are zero for diffusion mode
    Ux_xface = np.zeros((Nx + 1, Ny))
    Uy_yface = np.zeros((Nx, Ny + 1))

    return dx, dy, sig_a, sig_t, q_ex, robin, Ux_xface, Uy_yface, phi_ex

def main():
    Nx, Ny = 40, 35
    Lx, Ly = 1.0, 1.0

    dx, dy, sig_a, sig_t, q_ex, robin, Ux, Uy, phi_ex = build_mms_case(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
        sig_t_val=5.0, sig_s_val=4.5,
        alpha_x=1.0, alpha_y=1.0,
        use_harmonic_sig_t=False,
    )

    # Run LO solve
    phi_num, Jx_num, Jy_num, _cache = tc.solve_lo(
        source=q_ex,
        sig_a=sig_a,
        sig_t=sig_t,
        Ux_xface=Ux,
        Uy_yface=Uy,
        dx=dx, dy=dy,
        robin=robin,
        cache=None,
        use_harmonic_sig_t=False,
    )

    err = phi_num - phi_ex
    l2 = np.linalg.norm(err.ravel()) / np.sqrt(err.size)
    linf = np.max(np.abs(err))

    print(f"MMS error: L2={l2:.3e}  Linf={linf:.3e}")

    # Plot
    # Plot
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    fig, axs = plt.subplots(1, 3, figsize=(9, 4), sharex=True, sharey=True)

    # Use common levels for phi plots so colors mean the same thing
    vmin = min(phi_ex.min(), phi_num.min())
    vmax = max(phi_ex.max(), phi_num.max())
    levels_phi = np.linspace(vmin, vmax, 51)

    from cmcrameri.cm import batlow

    cs0 = axs[0].contourf(x, y, phi_ex.T, levels=levels_phi, cmap=batlow)
    cs0.set_edgecolor("face")
    axs[0].set_title(r"Manufactured")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("y")

    cs1 = axs[1].contourf(x, y, phi_num.T, levels=levels_phi, cmap=batlow)
    cs1.set_edgecolor("face")
    axs[1].set_title(r"Diffusion Sol")
    axs[1].set_xlabel("x")

    # Shared colorbar for first two plots
    fig.colorbar(cs1, ax=axs[:2], location="right", shrink=0.95, pad=0.02, label=r"$\phi$")

    # Error plot: symmetric levels around 0
    vmaxe = np.max(np.abs(err))
    levels_err = np.linspace(-vmaxe, vmaxe, 51)

    cs2 = axs[2].contourf(x, y, err.T, levels=levels_err, cmap="inferno")
    cs2.set_edgecolor("face")
    fig.colorbar(cs2, ax=axs[2], shrink=0.95, pad=0.02, label=r"$\epsilon$")
    axs[2].set_title(r"Error")
    axs[2].set_xlabel("x")

    #plt.tight_layout()
    plt.savefig("diffusion_mms.pdf")
    plt.show()



if __name__ == "__main__":
    main()
