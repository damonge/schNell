import os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from detector import GroundDetector
from mapping import MapCalculator
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica'],
              'size': 15})
rc('text', usetex=True)

f_ref = 63.
nside = 16

dets = {'Hanford':     GroundDetector('Hanford',     46.4, -119.4, 171.8,
                                      'data/curves_May_2019/aligo_design.txt'),
        'Livingstone': GroundDetector('Livingstone', 30.7,  -90.8, 243.0,
                                      'data/curves_May_2019/aligo_design.txt'),
        'VIRGO':       GroundDetector('VIRGO',       43.6,   10.5, 116.5,
                                      'data/curves_May_2019/advirgo_sqz.txt'),
        'Kagra':       GroundDetector('Kagra',       36.3,  137.2, 225.0,
                                      'data/curves_May_2019/kagra_sqz.txt'),
        'GEO600':      GroundDetector('GEO600',      48.0,    9.8,  68.8,
                                      'data/curves_May_2019/o1.txt')}
# Initialize the map calculator
mcals = {s1: {s2: MapCalculator(d1, d2, f_pivot=f_ref)
              for s2, d2 in dets.items()}
         for s1, d1 in dets.items()}


freqs = np.linspace(10., 1010., 101)
# One day
obs_time = 24*3600.
# 10-min intervals
nframes = 24*6
t_frames = np.linspace(0, obs_time, nframes+1)[:-1]

inoi_hl = mcals['Hanford']['Livingstone'].get_Ninv_t(t_frames, freqs, nside)


def plot_inoise_map(inoi, lims=[None, None], which='',
                    figname=None):
    hp.mollview(1E-5*np.sqrt(inoi), max=lims[1], min=lims[0],
                title=which,
                unit=r'$\sigma_N^{-1}\,\,[10^5\,\,{\rm srad}^{-1/2}\,{s}^{-1/2}]$')
    f = plt.gcf().get_children()
    HpxAx = f[1]
    CbAx = f[2]

    coord_text_obj = HpxAx.get_children()[0]
    coord_text_obj.set_fontsize(15)

    unit_text_obj = CbAx.get_children()[1]
    unit_text_obj.set_fontsize(15)
    if figname:
        plt.savefig(figname, bbox_inches='tight')


def make_videos(inoi_plot, prefix, remove_frames=True):
    fig, ax = plt.subplots()
    vmax = 1E-5*np.sqrt(np.amax(inoi_plot))
    for i in range(nframes):
        print(i)
        hp.mollview(1E-5*np.sqrt(inoi_plot[i]), min=0, max=vmax,
                    cbar=False, title='', notext=True, hold=True)
        plt.savefig(prefix+"_%03d.png" % i, bbox_inches='tight',
                    dpi=300)
        plt.cla()
    plt.close(fig)
    os.system('ffmpeg -i '+prefix+'_%03d.png -qscale 0 '+prefix+'.mp4')

    inoi_cum = np.cumsum(inoi_plot, axis=0)/len(inoi_plot)
    vmax = 1E-5*np.sqrt(np.amax(inoi_cum))
    fig, ax = plt.subplots()
    for i in range(nframes):
        print(i)
        hp.mollview(1E-5*np.sqrt(inoi_cum[i]), min=0, max=vmax,
                    cbar=False, title='', notext=True, hold=True)
        plt.savefig(prefix+"_cum_%03d.png" % i, bbox_inches='tight',
                    dpi=300)
        plt.cla()
    plt.close(fig)
    os.system('ffmpeg -i '+prefix+'_cum_%03d.png -qscale 0 ' +
              prefix+'_cumul.mp4')

    if remove_frames:
        os.system('rm '+prefix+'*.png')


make_videos(inoi_hl, 'plots/vid_hl', remove_frames=True)
plot_inoise_map(inoi_hl[0], lims=[0, 4],
                which='HL, instantaneous',
                figname='plots/noivar_hl_inst.pdf')
plot_inoise_map(np.mean(inoi_hl, axis=0), lims=[0, 2.3],
                which='HL, integrated',
                figname='plots/noivar_hl_cumul.pdf')
plt.show()

inoi = inoi_hl.copy()
inoi += mcals['Hanford']['VIRGO'].get_Ninv_t(t_frames, freqs, nside)
inoi += mcals['Hanford']['Kagra'].get_Ninv_t(t_frames, freqs, nside)
inoi += mcals['Hanford']['GEO600'].get_Ninv_t(t_frames, freqs, nside)
inoi += mcals['Livingstone']['VIRGO'].get_Ninv_t(t_frames, freqs, nside)
inoi += mcals['Livingstone']['Kagra'].get_Ninv_t(t_frames, freqs, nside)
inoi += mcals['Livingstone']['GEO600'].get_Ninv_t(t_frames, freqs, nside)
inoi += mcals['VIRGO']['Kagra'].get_Ninv_t(t_frames, freqs, nside)
inoi += mcals['VIRGO']['GEO600'].get_Ninv_t(t_frames, freqs, nside)
inoi += mcals['Kagra']['GEO600'].get_Ninv_t(t_frames, freqs, nside)
inoi_plot = inoi

make_videos(inoi, 'plots/vid_all', remove_frames=True)
plot_inoise_map(inoi[0], lims=[0, 8],
                which='All, instantaneous',
                figname='plots/noivar_all_inst.pdf')
plot_inoise_map(np.mean(inoi, axis=0), lims=[0, 7.3],
                which='All, integrated',
                figname='plots/noivar_all_cumul.pdf')
plt.show()
