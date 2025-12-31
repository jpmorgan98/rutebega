import numpy as np
import matplotlib as plt
from smom_2d_fast import transport_2d_oci
from tqdm import tqdm

c = np.linspace(0, 1.0, 75)
delta = np.logspace(-1, 1, 100)



rho_un = np.zeros((c.size, delta.size))
rho_yavuz = np.zeros((c.size, delta.size))
rho_rescale = np.zeros((c.size, delta.size))
it_un = np.zeros((c.size, delta.size))
it_yavuz = np.zeros((c.size, delta.size))
it_rescale = np.zeros((c.size, delta.size))


def rel_l2(a, b, eps=1e-14):
    # relative L2 error: ||a-b||2 / ||b||2
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b)
    return num / (den + eps)

def rel_linf(a, b, eps=1e-14):
    # relative Linf error: ||a-b||inf / ||b||inf
    num = np.max(np.abs(a - b))
    den = np.max(np.abs(b))
    return num / (den + eps)

# --- error storage (one scalar per condition) ---
err_phi_yavuz_l2   = np.zeros((c.size, delta.size))
err_phi_rescale_l2 = np.zeros((c.size, delta.size))
err_psi_yavuz_l2   = np.zeros((c.size, delta.size))
err_psi_rescale_l2 = np.zeros((c.size, delta.size))

err_phi_yavuz_linf   = np.zeros((c.size, delta.size))
err_phi_rescale_linf = np.zeros((c.size, delta.size))
err_psi_yavuz_linf   = np.zeros((c.size, delta.size))
err_psi_rescale_linf = np.zeros((c.size, delta.size))


total = c.size * delta.size
with tqdm(total=total, desc="Running transport_2d_oci") as pbar:
    for i in range (c.size):
        for j in range (delta.size):
            q = 1.2
            xsec = 1.5

            L = 10.0
            dx = delta[j]/xsec
            N = int(L/dx)

            xsec_scatter = xsec*c[i]

            inf_homo = q / ( xsec*(1-c[i]) )
            inf_homo /= (2*np.pi)
            
            bc = dict(left=inf_homo, right=inf_homo, bottom=inf_homo, top=inf_homo)
            
            pbar.set_description(
                f"i={i+1}/{c.size}: c={c[i]:.2f}, j={j+1}/{delta.size}: δ={delta[j]:.2e} : rescale "
            )

            phi_rescale, psi_rescale, rho_rescale[i, j], it_rescale[i, j] = transport_2d_oci(printer=False,
                            Ny=N, Nx=N, Ly=L, Lx=L, bc=bc, q_val=q,
                            sig_s_val=xsec_scatter, sig_t_val=xsec,
                            smm_acc=True, update="rescale", max_it=500,
                            )
            
            pbar.set_description(
                f"i={i+1}/{c.size}: c={c[i]:.2f}, j={j+1}/{delta.size}: δ={delta[j]:.2e} : un-accelerated "
            )

            phi_un, psi_un, rho_un[i, j], it_un[i, j] = transport_2d_oci(printer=False,
                            Ny=N, Nx=N, Ly=L, Lx=L, bc=bc, q_val=q,
                            sig_s_val=xsec_scatter, sig_t_val=xsec,
                            smm_acc=False, max_it=500,
                            )
            
            pbar.set_description(
                f"i={i+1}/{c.size}: c={c[i]:.2f}, j={j+1}/{delta.size}: δ={delta[j]:.2e} : yavuz "
            )

            phi_yavuz, psi_yavuz, rho_yavuz[i, j], it_yavuz[i, j] = transport_2d_oci(printer=False,
                            Ny=N, Nx=N, Ly=L, Lx=L, bc=bc, q_val=q,
                            sig_s_val=xsec_scatter, sig_t_val=xsec,
                            smm_acc=True, update="yavuz", max_it=500, 
                            )
            
            psi_exact = inf_homo * np.ones_like(psi_rescale)
            phi_exact = (inf_homo * 2.0 * np.pi) * np.ones_like(phi_rescale)

            # Errors vs "transport" reference (un-accelerated)
            err_phi_yavuz_l2[i, j]   = rel_l2(phi_yavuz,   phi_exact)
            err_phi_rescale_l2[i, j] = rel_l2(phi_rescale, phi_exact)
            err_psi_yavuz_l2[i, j]   = rel_l2(psi_yavuz,   psi_exact)
            err_psi_rescale_l2[i, j] = rel_l2(psi_rescale, psi_exact)

            err_phi_yavuz_linf[i, j]   = rel_linf(phi_yavuz,   phi_exact)
            err_phi_rescale_linf[i, j] = rel_linf(phi_rescale, phi_exact)
            err_psi_yavuz_linf[i, j]   = rel_linf(psi_yavuz,   psi_exact)
            err_psi_rescale_linf[i, j] = rel_linf(psi_rescale, psi_exact)


            # if it_yavuz[i, j] >= 499:
            #     print(f"SMM hit max iterations at δ {delta[j]:2e} (Σ {xsec:.3f})  c {c[i]:.2f} (Σs {xsec_scatter:.3f})")
            # if it_un[i, j] >= 499:
            #     print(f"OCI hit max iterations at δ {delta[j]:2e} (Σ {xsec:.3f})  c {c[i]:.2f} (Σs {xsec_scatter:.3f})")

            
            pbar.update(1)

np.savez("twoD_exp2", mfp=delta, c=c, spec_rad_oci=rho_un, spec_rad_yavuz=rho_yavuz, spec_rad_rescale=rho_rescale,
         err_phi_yavuz_l2=err_phi_yavuz_l2,
        err_phi_rescale_l2=err_phi_rescale_l2,
        err_psi_yavuz_l2=err_psi_yavuz_l2,
        err_psi_rescale_l2=err_psi_rescale_l2,

        err_phi_yavuz_linf=err_phi_yavuz_linf,
        err_phi_rescale_linf=err_phi_rescale_linf,
        err_psi_yavuz_linf=err_psi_yavuz_linf,
        err_psi_rescale_linf=err_psi_rescale_linf,)

print("yavuz")
print(rho_yavuz)
print("rescale")
print(rho_rescale)
print("oci")
print(rho_un)

#transport_2d_oci()
