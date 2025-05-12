import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Three slab layers with different purely-absorbing materials

# Set materials
m1 = mcdc.material(scatter=np.array([[1.0]]))
m2 = mcdc.material(capture=np.array([0.05]), scatter=np.array([[0.95]]))
m3 = mcdc.material(capture=np.array([0.20]), scatter=np.array([[0.80]]))
m4 = mcdc.material(capture=np.array([0.05]), scatter=np.array([[0.95]]))

# Set surfaces
s1 = mcdc.surface("plane-z", z=0.0, bc="vacuum")
s2 = mcdc.surface("plane-z", z=1.0)
s3 = mcdc.surface("plane-z", z=2.0)
s4 = mcdc.surface("plane-z", z=3.0)
s5 = mcdc.surface("plane-z", z=4.0, bc="vacuum")

# Set cells
mcdc.cell(+s1 & -s2, m1)
mcdc.cell(+s2 & -s3, m2)
mcdc.cell(+s3 & -s4, m3)
mcdc.cell(+s4 & -s5, m4)

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the domain

mcdc.source(z=[1.0, 3.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: cell-average fluxes and currents
mcdc.tally.mesh_tally(
    scores=["flux"],
    z=np.linspace(0.0, 4.0, 100),
    mu=np.linspace(-1.0, 1.0, 32 + 1),
)

# Setting
mcdc.setting(N_particle=1e8, N_batch=10)

# srun -N 1 -n 112 -p pdebug -A orsu python input.py --mode=numba --output=yavuz_problem
# Run
mcdc.run()
