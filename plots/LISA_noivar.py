import os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import schnell as snl
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica'],
              'size': 15})
rc('text', usetex=True)

dets = [snl.LISADetector(i) for i in range(3)]
# Correlation between detectors
rho = snl.NoiseCorrelationLISA(dets[0])
mc = snl.MapCalculator(dets, f_pivot=1E-2,
                       corr_matrix=rho)
nside = 16


lfreqs = np.linspace(-4, 0, 101)
freqs = 10.**lfreqs

# One year
obs_time = 365*24*3600.
# 1-day intervals
nframes = 24*365
t_frames = np.linspace(0, obs_time, nframes+1)[:-1]
inoi_tot = mc.get_Ninv_t(t_frames, freqs, nside,
                         is_fspacing_log=True)


def plot_inoise_map(inoi, lims=[None, None], which='',
                    figname=None):
    hp.mollview(1E-9*np.sqrt(inoi), max=lims[1], min=lims[0],
                title=which,
                unit=r'$\sigma_N^{-1}\,\,[10^9\,\,{\rm srad}^{-1/2}\,{s}^{-1/2}]$')
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
    vmax = 1E-9*np.sqrt(np.amax(inoi_plot))
    for i in range(nframes):
        print(i)
        hp.mollview(1E-9*np.sqrt(inoi_plot[i]), min=0, max=vmax,
                    cbar=False, title='', notext=True, hold=True)
        plt.savefig(prefix+"_%03d.png" % i, bbox_inches='tight',
                    dpi=300)
        plt.cla()
    plt.close(fig)
    os.system('ffmpeg -i '+prefix+'_%03d.png -qscale 0 '+prefix+'.mp4')

    inoi_cum = np.cumsum(inoi_plot, axis=0)/len(inoi_plot)
    vmax = 1E-9*np.sqrt(np.amax(inoi_cum))
    fig, ax = plt.subplots()
    for i in range(nframes):
        print(i)
        hp.mollview(1E-9*np.sqrt(inoi_cum[i]), min=0, max=vmax,
                    cbar=False, title='', notext=True, hold=True)
        plt.savefig(prefix+"_cum_%03d.png" % i, bbox_inches='tight',
                    dpi=300)
        plt.cla()
    plt.close(fig)
    os.system('ffmpeg -i '+prefix+'_cum_%03d.png -qscale 0 ' +
              prefix+'_cumul.mp4')

    if remove_frames:
        os.system('rm '+prefix+'*.png')


make_videos(inoi_tot, 'vid_LISA_tt', remove_frames=True)
plot_inoise_map(inoi_tot[0], lims=[0, 0.75],
                which='Instantaneous',
                figname='noivar_LISA_tt_inst.pdf')
plot_inoise_map(np.mean(inoi_tot, axis=0), lims=[0, 0.4],
                which='Integrated',
                figname='noivar_LISA_tt_cumul.pdf')
plt.show()
