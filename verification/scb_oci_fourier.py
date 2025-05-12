import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import math


np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

N_angle = 8

[angles, weights] = np.polynomial.legendre.leggauss(N_angle)

#angles = angles[:N_angle]
#weights = weights[:N_angle]

N_lam = 50
lam = np.pi*np.linspace(.1,2,N_lam)

dx = .1
sigma = 1
sigmas = .7

i = complex(0,1)

def Rblockpos(mu, sigma):
    return(
        np.array([[mu/2+sigma*dx/2, -mu/2],
                  [mu/2, mu/2+sigma*dx/2]])
    )
def Rblockneg(mu, sigma):
    return(
        np.array([[-mu/2 + sigma*dx/2, -mu/2],
                  [mu/2, -mu/2 + sigma*dx/2]])
    )
def Rbuild(sigma):
    A = np.zeros((2*N_angle, 2*N_angle),dtype=np.complex128)
    for a in range(N_angle):
        if angles[a] > 0:
            A[a*2:(a+1)*2,a*2:(a+1)*2] = Rblockpos(angles[a], sigma)
        elif (angles[a] < 0):
            A[a*2:(a+1)*2,a*2:(a+1)*2] = Rblockneg(angles[a], sigma)
    return(A)



def Eblockpos(mu, lamv, sigma):
    return(
        np.array([[0,0],
                  [mu*np.exp(-i*sigma*lamv*dx),0]])
    )
def Eblockneg(mu, lamv, sigma):
    return(
        np.array([[0,-mu*np.exp(i*lamv*sigma*dx)],
                  [0,0]])
    )

def Ebuild(l, sigma):
    A = np.zeros((2*N_angle, 2*N_angle),dtype=np.complex128)
    for a in range(N_angle):
        if   angles[a] > 0:
            A[a*2:(a+1)*2,a*2:(a+1)*2] = Eblockpos(angles[a],l, sigma)
        elif angles[a] < 0:
            A[a*2:(a+1)*2,a*2:(a+1)*2] = Eblockneg(angles[a],l, sigma)
    return(A)


def Gblockpos(mu):
    return(
        np.array([[mu/2, -mu/2],
                  [mu/2, mu/2-mu]])
    )
def Gblockneg(mu):
    return(
        np.array([[-mu/2+mu, -mu/2],
                  [mu/2, -mu/2]])
    )
def Gbuild():
    A = np.zeros((2*N_angle, 2*N_angle),dtype=np.complex128)
    for a in range(N_angle):
        if angles[a] > 0:
            A[a*2:(a+1)*2,a*2:(a+1)*2] = Gblockpos(angles[a])
        elif (angles[a] < 0):
            A[a*2:(a+1)*2,a*2:(a+1)*2] = Gblockneg(angles[a])
    return(A)

def Sbuild():
    
    S = np.zeros((2*N_angle,2*N_angle)).astype(np.complex128)
    #for m in range(N_angle):

    beta = sigmas * dx / 4

    for p in range(N_angle):
        for j in range(N_angle):
            S[p*2,   j*2]   = beta*weights[j]
            S[p*2+1, j*2+1] = beta*weights[j]

    return(S)



def compute(sigma):

    eig_lam = np.zeros(1).astype(np.complex128)

    R = Rbuild(sigma)
    S = Sbuild()

    Rinv = np.linalg.inv(R-S)

    for i in range(N_lam):
        E = Ebuild(lam[i], sigma)
        T =  np.matmul(Rinv, E)
        eig_val, stand_eig_mat = np.linalg.eig(T)

        eig_lam = np.append(eig_lam, eig_val)

    #max_eigval = eig_lam[np.abs(eig_lam).argmax()]
    #plt.plot(eig_lam.real, eig_lam.imag, 'b.') 
    #plt.plot(max_eigval.real, max_eigval.imag, 'rX', markersize=15, label=r'$\rho$ ({})'.format(np.abs(max_eigval)))
    #plt.ylabel('Imaginary') 
    #plt.xlabel('Real') 
    #plt.legend()
    #plt.show()

    return(np.max(np.abs(eig_lam)))


def computeSI(sigma):

    eig_lam = np.zeros(1).astype(np.complex128)

    R = Rbuild(sigma)
    S = Sbuild()

    #Rinv = np.linalg.inv(R-S)

    for i in range(N_lam):
        E = Ebuild(lam[i], sigma)
        Rinv = np.linalg.inv(R-E)
        T =  np.matmul(Rinv, S)
        eig_val, stand_eig_mat = np.linalg.eig(T)

        eig_lam = np.append(eig_lam, eig_val)

    #max_eigval = eig_lam[np.abs(eig_lam).argmax()]
    #plt.plot(eig_lam.real, eig_lam.imag, 'b.') 
    #plt.plot(max_eigval.real, max_eigval.imag, 'rX', markersize=15, label=r'$\rho$ ({})'.format(np.abs(max_eigval)))
    #plt.ylabel('Imaginary') 
    #plt.xlabel('Real') 
    #plt.legend()
    #plt.show()

    return(np.max(np.abs(eig_lam)))


def computeBJTSA(sigma):
    eig_lam = np.zeros(1).astype(np.complex128)

    R = Rbuild(sigma)
    S = Sbuild()

    Rinv = np.linalg.inv(R-S)

    sigma_limit = sigma

    for i in range(N_lam):

        #R = Rbuild()
        E = Ebuild(lam[i], sigma)
        T =  np.matmul(Rinv, E)

        # *(T-np.identity(2*N_angle)*(-E))
        #print(np.matmul(np.matmul(np.linalg.inv(G),(E)), (T-np.identity(2*N_angle))))

        beta = 1.0
        #sigma = 0
        #R = Rbuild()
        #G = Gbuild()
        #R_tsa = Rbuild(sigma)
        #E = Ebuild(lam[i], sigma)
        
        L = (R-E)
        #TSA =  np.matmul(Rinv, S)
        TSA = np.linalg.inv((L-(1-beta)*S)) #-(1-beta)*S
        #TSA = np.matmul(TSA, S)
        #DSA = Gbuild()
        
        Td = np.add (T, np.matmul( np.matmul( TSA, E ), (T-np.identity(2*N_angle))) )

        eig_val, stand_eig_mat = np.linalg.eig(Td)

        eig_lam = np.append(eig_lam, eig_val)

    max_eigval = eig_lam[np.abs(eig_lam).argmax()]

    #plt.plot(eig_lam.real, eig_lam.imag, 'b.') 
    #plt.plot(max_eigval.real, max_eigval.imag, 'rX', markersize=15, label=r'$\rho$ ({})'.format(np.abs(max_eigval)))
    #plt.ylabel('Imaginary') 
    #plt.xlabel('Real') 
    #plt.legend()
    #plt.show()

    return(np.max(np.abs(eig_lam)))


def RaccPos(mu, lamv):
    return np.array([[0, mu*np.exp(-i*lamv*sigma*dx), mu/2 + sigma*dx/2, mu/2],
              [0, 0, -mu/2, mu/2 + sigma*dx/2],
              [0, -dx*mu/2*np.exp(-i*lamv*sigma*dx), -mu/2, -mu/2-mu*np.exp(-i*lamv*sigma*dx)+ dx*mu/2*np.exp(-i*lamv*sigma*dx)],
              [0, 0, mu/2, -mu/2]])

def EaccPos(mu, lamv):
    np.array([[0, 0, 0, 0],
              [0, 0, 0, 0],
              [-mu/2, -mu/2-mu*np.exp(-i*lamv*sigma*dx), 0, 0],
              [0, 0, 0, 0]])


def RaccBuild(lamv):
    A = np.zeros((4*N_angle, 4*N_angle),dtype=np.complex128)
    for a in range(N_angle):
        if angles[a] > 0:
            A[a*4:(a+1)*4,a*4:(a+1)*4] = RaccPos(angles[a], lamv)
        elif (angles[a] < 0):
            print("Not implemented!")
            #A[a*2:(a+1)*2,a*2:(a+1)*2] = Rblockneg(angles[a])
    return(A)

def EaccBuild(lamv):
    A = np.zeros((4*N_angle, 4*N_angle),dtype=np.complex128)
    for a in range(N_angle):
        if angles[a] > 0:
            A[a*4:(a+1)*4,a*4:(a+1)*4] = EaccPos(angles[a], lamv)
        elif (angles[a] < 0):
            print("Not implemented!")
            #A[a*2:(a+1)*2,a*2:(a+1)*2] = Rblockneg(angles[a])
    return(A)


def Sacc():
    
    S = np.zeros((4*N_angle,4*N_angle)).astype(np.complex128)
    #for m in range(N_angle):

    beta = sigmas * dx / 2

    for p in range(N_angle):
        for j in range(N_angle):
            S[p*4+2, j*4+2]   = beta*weights[j]
            S[p*4+3, j*4+3] = beta*weights[j]

    return(S)


def computeACC():
    eig_lam = np.zeros(1).astype(np.complex128)

    S = Sacc()

    for i in range(N_lam):
        E = EaccBuild(lam[i])
        R = RaccBuild(lam[i])

        #print(R)
        #print(np.linalg.det(E))

        BJSOA = np.linalg.inv( E )

        Td = np.matmul ( R, BJSOA )
        eig_val, stand_eig_mat = np.linalg.eig(Td)

        eig_lam = np.append(eig_lam, eig_val)

    max_eigval = eig_lam[np.abs(eig_lam).argmax()]

    #plt.plot(eig_lam.real, eig_lam.imag, 'b.') 
    #plt.plot(max_eigval.real, max_eigval.imag, 'rX', markersize=15, label=r'$\rho$ ({})'.format(np.abs(max_eigval)))
    #plt.ylabel('Imaginary') 
    #plt.xlabel('Real') 
    #plt.legend()
    #plt.show()

    return(np.max(np.abs(eig_lam)))


if __name__ == '__main__':
    #mfp = 1
    #scat = 0.0
    #dx = mfp/sigma
    #dx = 0.4
    #sigma = 10.0
    #sigmas = 0*sigma
    #print( computeACC() )



    #exit()

    #color_si = '#ca0020'
    #color_oci = '#404040'

    N_mfp = 25
    N_c = 30

    mfp_range = np.logspace(-1,2,N_mfp)
    c_range = np.linspace(0,1,N_c)

    #mfp_range = np.array([0.01])
    #c_range = np.array([0.0])

    spec_rad = np.zeros([N_mfp, N_c])
    spec_rad_si = np.zeros([N_mfp, N_c])
    spec_rad_tsa = np.zeros([N_mfp, N_c])
    spec_rad_acc = np.zeros([N_mfp, N_c])

    itter = 0
    for y in range(mfp_range.size):
        for u in range(c_range.size):
            dx = mfp_range[y]/sigma
            sigmas = c_range[u]*sigma
            spec_rad[y,u] = compute(sigma)
            spec_rad_si[y,u] = computeSI(sigma)
            spec_rad_tsa[y,u] = computeBJTSA(sigma)
            #spec_rad_acc[y,u] = computeACC()
    

    for t in range(spec_rad_acc.shape[0]):
        for y in range(spec_rad_acc.shape[1]):
            if spec_rad_acc[t,y] > 1.0:
                spec_rad_acc[t,y] = 1.0

    c, mfp = np.meshgrid(c_range, mfp_range)

    fig, axes = plt.subplots(ncols=3, layout="constrained")
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    
    #fig.colorbar(surf)
    
    surf1 = ax1.contourf(mfp, c, spec_rad_si, levels=100, vmin=0, vmax=1, cmap=cm.viridis, antialiased=False)
    #fig.colorbar(surf2)

    surf2 = ax2.contourf(mfp, c, spec_rad, levels=100, vmin=0, vmax=1, cmap=cm.viridis, antialiased=False)
    
    surf3 = ax3.contourf(mfp, c, spec_rad_tsa, levels=100, vmin=0, vmax=1, cmap=cm.viridis, antialiased=False)
    #fig.colorbar(surf3) vmin=0, vmax=1,

    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')

    ax1.set_title("SI")
    ax2.set_title("OCI")
    ax3.set_title("TSA+OCI")

    fig.colorbar(surf1, ax=axes.ravel().tolist(), label='ρ', ticks=[0, .2, .4, .6, .8, 1.0])

    surf1.set_edgecolor("face")
    surf2.set_edgecolor("face")
    surf3.set_edgecolor("face")

    #ax3.set_xscale('log')

    # Customize the z axis.
    #ax.set_zlim(0, 1.0) #np.max(spec_rad)
    #ax1.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    #ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
     #shrink=5, aspect=5
 
    #surf3 = ax3.contourf(mfp, c, spec_rad/spec_rad_si, cmap=cm.viridis, levels=100,
    #                   linewidth=0, antialiased=False)
    ax1.set_xlabel(r"$\delta$")
    ax1.set_ylabel(r"$c$")

    ax2.set_xlabel(r"$\delta$")
    ax3.set_xlabel(r"$\delta$")


    #ax1.text(3e0, .1, 'OCI', color='w', style='italic',)# bbox={'facecolor': color_oci, 'alpha': 0.5, 'pad': 5})
    
    #ax2.text(3e0, .1, 'TSA', color='w', style='italic',)# bbox={'facecolor': color_si, 'alpha': 0.5, 'pad': 5})

    #ax3.text(3e0, .1, 'SOA', color='w', style='italic',)# bbox={'facecolor': color_si, 'alpha': 0.5, 'pad': 5})

    plt.gcf().set_size_inches(8, 4)

    ax2.label_outer()
    ax3.label_outer()
    
    #fig.tight_layout()

    #ax3.set_xlabel(r"$\delta$")
    #ax1.set_zlabel(r"$\rho$")
    #ax1.set_title(r"$\rho$ for $\lambda \in [0,2\pi]$ (at {0} points), in $S_{1}$".format(N_lam, N_angle))
    #plt.show()
    plt.savefig("ss_specrads.pdf")

    exit()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(mfp, c, spec_rad_acc, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(0, 1.0) #np.max(spec_rad)
    
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    #ax.set_yscale("log")
    ax.set_xlabel("mfp")
    ax.set_ylabel("c")
    ax.set_zlabel(r"$\rho$")
    ax.set_title(r"")
    plt.show()


