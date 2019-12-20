import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def get_pal(ncolors, npercolor, plot=False):
    base_colors = sns.color_palette("deep")[:ncolors]
    n_off = npercolor // 3.
    pal = np.vstack([sns.light_palette(c, npercolor+n_off, reverse=True)[:npercolor] for c in base_colors])
    sns.set_palette(pal)
    if plot:
        sns.palplot(pal)
    return pal

def add_label(label, xoff=-0.1, yoff=1.3):
    ax = plt.gca()
    ax.text(xoff, yoff, '%s'%label, transform=ax.transAxes,
      fontsize=12, fontweight='bold', va='top', ha='right')

def pcolor(*args, **kwargs):
    """Version of pcolor that removes edges""" 
    h = plt.contourf(*args, **kwargs)
    #h.set_edgecolor('face')
    return h


def sigma_pcolor(q,  Length, weight_sigmas, draw_colorbar=True, **kwargs):
    if 'vmax' not in kwargs:
      #  kwargs['vmax'] = int(np.ceil(np.nanmax(q)))
        if np.max(q) < 0.7:
            kwargs['vmax'] = 0.6
        else:
            kwargs['vmax'] = 1.0
    pcolor( weight_sigmas,Length, q, cmap = 'copper', vmin=0.1, **kwargs)
   # plt.yticks(weight_sigmas[weight_sigmas == weight_sigmas.astype(int)])
   # plt.xticks(bias_sigmas[bias_sigmas == bias_sigmas.astype(int)])
   # plt.xlabel('$\sigma^2_w$')
   # plt.ylabel('$L$')
    cmax = kwargs['vmax']
    if draw_colorbar:
        plt.colorbar(ticks=(0.1, cmax/2.0, cmax))

