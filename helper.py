import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D

def plot_joint_3d(conjunta_dict, bins_width = 1, az=50, el=-5, ax=None, p_max=None, a_max=None, p_min=None, a_min=None, color='b'):
    if p_max is None:
        conj_array = np.array([[p,a] for p,a in conjunta_dict.keys()])
        p_max, a_max = np.max(conj_array, axis=0)
        p_min, a_min = np.min(conj_array, axis=0)
    espacio_muestral_pesos = np.linspace(p_min, p_max, p_max - p_min + 1)
    espacio_muestral_alturas = np.linspace(a_min, a_max, a_max - a_min + 1)
    xpos, ypos = np.meshgrid(espacio_muestral_pesos, espacio_muestral_alturas)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)
    dx = bins_width * np.ones_like(zpos)
    dy = dx.copy()
    
    conjunta_H = np.zeros([p_max - p_min + 1, a_max-a_min + 1])
    height, width = conjunta_H.shape
    for (p,a), f in conjunta_dict.items():
        conjunta_H[p - p_min, a - a_min] = f
        
    dz = conjunta_H.astype(int).flatten()
    if ax == None:
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color, alpha=0.5)
    ax.view_init(az, el)
    if ax == None:
        plt.show()
    return conjunta_H

def plot_joint_hists_dicts(conjunta_dict):
    conj_array = np.array([[p,a] for p,a in conjunta_dict.keys()])
    p_max, a_max = np.max(conj_array, axis=0)
    p_min, a_min = np.min(conj_array, axis=0)
    conjunta_H = np.zeros([p_max - p_min + 1, a_max-a_min + 1])
    height, width = conjunta_H.shape
    for (p,a), f in conjunta_dict.items():
        conjunta_H[p - p_min, a - a_min] = f
    
    
    
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left = 0
    top = 0
    rect_scatter = [0, 0, width/height, 1]
    rect_histx =   [0, -0.2 -0.01, width/height , 0.2]
    rect_histy =   [width/height + 0.01, 0, 0.2, 1]
    # start with a rectangular Figure
    plt.figure(figsize=(6, 6))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    # the scatter plot:
    
    axHistx.bar(np.array(range(conjunta_H.shape[1]))+a_min, conjunta_H.sum(axis=0))
    axHisty.barh(np.array(range(conjunta_H.shape[0]))+p_min,conjunta_H.sum(axis=1))
    axHisty.yaxis.tick_right()
    axScatter.matshow(np.flip(conjunta_H, axis=0), cmap='gray')
    axScatter.xaxis.set_major_formatter(nullfmt)
    axScatter.yaxis.set_major_formatter(nullfmt)
    plt.show()
    return conjunta_H.astype(int), p_min, a_min

def plot_joint_hists(conjunta_H, frec_alt_H, frec_pesos_H):
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    rect_scatter = [0, 0, 0.8*53/72, 0.8*72/53]
    rect_histx = [0, 0, 0.8*53/72, 0.2]
    rect_histy = [0.8*53/72+0.05, 0.2, 0.2, 53/72]
    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    #axHistx.xaxis.set_major_formatter(nullfmt)
    #axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.matshow(np.flip(conjunta_H, axis=0), cmap='gray')
    axScatter.xaxis.set_major_formatter(nullfmt)
    axScatter.yaxis.set_major_formatter(nullfmt)

    axHistx.bar(frec_alt_H.keys(), frec_alt_H.values())
    axHisty.barh(list(frec_pesos_H.keys()),list(frec_pesos_H.values()))
    #axScatter.set_xlim(np.array(axHistx.get_xlim())-144.96)
    #axHisty.set_ylim(axScatter.get_ylim())

    plt.show()
   
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
    
def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def plot_mv_gaussian(mu, Sigma, N = 200):
    # Our 2-dimensional distribution will be over variables X and Y
    std1 = np.sqrt(Sigma[0,0])
    std2 = np.sqrt(Sigma[1,1])
    X = np.linspace(-2*std1 + mu[0], 2*std1 + mu[0], N)
    Y = np.linspace(-2*std2 + mu[1], 2*std2 + mu[1], N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)

    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure(figsize=(20,10))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)

    count_offseet = - np.max(Z)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=count_offseet, cmap=cm.viridis)

    # Adjust the limits, ticks and view angle
    ax.set_zlim(count_offseet,-count_offseet)
    #ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(25, -21)
    plt.show()