# coding: utf-8

import datetime
import glob
import python_method_2d_gaussian_fit as pm2g
import python_method_QuadscanParabelFit as pmqp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import sys

from matplotlib import colors as mcolors


def fitparabel(fitfkt, kwerte, fitwerte, fehler, startwerte, seite):

    if fitfkt == fitfkt1:
        c = "green"
    else:
        c = "blue"

    fkt = odr.Model(fitfkt)
    data = odr.Data(kwerte, fitwerte, we=1. / np.power(fehler, 2))

    # try:
    odrparabel = odr.ODR(data, fkt, beta0=startwerte, ifixb=ifixb, maxit=1000)
    odroutput = odrparabel.run()
    # odroutput.cov_beta = odroutput.cov_beta * (odroutput.res_var) #Nur wenn keine x Fehler berücksichtigt werden
    # uncertainty = np.sqrt(np.diagonal(odroutput.cov_beta))
    print("Beta: ", odroutput.beta, "Beta Std Error: ", odroutput.sd_beta)
    return odroutput.beta



##########################################################################

pic_dir  = "../quad_pics/180726/"
measure_time = "1248"

##########################################################################

measure_timestamp = "{}_{}".format(pic_dir[-7:-1], measure_time)
pic_path = pic_dir + measure_timestamp + "/"
infile = glob.glob(pic_path + "*result_DataFrame.dat")[0]

# Miscellanous
s = 0.04921     # effektive Länge in m
l = {1: 0.3979, 2: 0.2659, 3: 0.1339}  # Trip 020
#l = {1: 0.3575, 2: 0.2255, 3: 0.0935}  # Trip 050

# Load results
try:
    headers = pm2g.find_headers(infile)
    df = pd.read_csv(infile, comment='#', index_col=0)
except:
    print("Could not find dataframes.")
    print("Aborting.")
    sys.exit()

# Beam current information
try:
    Ibd = headers[0].split(',')[2].strip()
    quad_no = headers[0].split(',')[3].strip()
    if not Ibd[:3] == 'Ibd':
        Ibd = 'NA'
    if quad_no[:7] == 'quad_no':
        quad_no = int(quad_no[-1])
except:
    Ibd = 'NA'

# Assign for fit
k = df['k[1/m2]']
sigx, sigxerr = df['sigma_x[mm]']/1000, df['sigma_x_err[mm]']/1000  # Sigma in m
sigy, sigyerr = df['sigma_y[mm]']/1000, df['sigma_y_err[mm]']/1000

# Fit results
print("Try fitting x")
resx = pmqp.parfit(k, sigx**2, sigxerr**2, s, l[quad_no])
print("Try fitting y")
resy = pmqp.parfit(k, sigy**2, sigyerr**2, s, l[quad_no])
epsxl, epsyl = [resx['epsl'], resx['depsl']], [resy['epsl'], resy['depsl']]
epsxr, epsyr = [resx['epsr'], resx['depsr']], [resy['epsr'], resy['depsr']]
mean_epsx, dmean_epsx = resx['mean_eps'], resx['dmean_eps']
mean_epsy, dmean_epsy = resy['mean_eps'], resy['dmean_eps']


# Result path
outfile = infile[:-4] + "_emittance"
if os.path.isfile(outfile + ".pdf") or os.path.isfile(outfile + ".png"):
    overwrite = input("{} already exists, do you want to overwrite it?[yes, please]\nIt will be postfixed by timestamp instead.")
    if overwrite  != 'yes, please':
        now = datetime.datetime.now()
        postfix = now.strftime("%y%m%d_%H%M")
        outfile =  infile[:-4] + "_emittance_{}".format(postfix)

# Plot
## Figure and axes
fig, ax = plt.subplots(figsize = (12,8))
ay = ax.twinx()

## Fonts
font = {'family' : 'sans-serif',
        'size'   : 11}
mpl.rc('font', **font)

## Colors    
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
c = {'x':'red', 'xr': 'orange', 'xl': colors['orangered'], 'y':'blue', 'yr':'cyan', 'yl':colors['royalblue']}

## Labels
ax.set_title("Emittance measurement: {}".format(infile))
ax.set_xlabel(r'$k$ [m$^{-2}$]')
ax.set_ylabel(r'$\sigma_x$ [mm$^{2}$]', color=c['x'])
ay.set_ylabel(r'$\sigma_y$ [mm$^{2}$]', color=c['y'])

info_text = Ibd + "\n"
if not mean_epsx == 0:
    info_text = info_text + r'$\varepsilon_{{n,rms,x}} = ({:.4f} +/- {:.4f})$ um'.format(mean_epsx, dmean_epsx) + "\n"
else:
    info_text = info_text + r'$\varepsilon_{{n,rms,x}} = $ NA' + "\n"
if not mean_epsy == 0:
    info_text = info_text + r'$\varepsilon_{{n,rms,y}} = ({:.4f} +/- {:.4f})$ um'.format(mean_epsy, dmean_epsy) 
else:
    info_text = info_text + r'$\varepsilon_{{n,rms,y}} = $ NA'

fig.text(0.01, 0.01, info_text, fontsize = 12)
    
ln = []

## Data
ln.append(ax.errorbar(k, sigx**2*1e6, yerr=sigxerr**2*1e6, fmt=c['x'][0]+'o', ecolor=c['x'], mfc=c['x'], mec='black', label='x'))
ln.append(ay.errorbar(k, sigy**2*1e6, yerr=sigyerr**2*1e6, fmt=c['y'][0]+'^', ecolor=c['y'], mfc=c['y'], mec='black', label='y'))

## left fits
ln.append(ax.plot(k[k<0], resx['fitted_sigmasql']*1e6, '--', marker=None, color=c['xl'], label='x-fit (left)'))
ln.append(ay.plot(k[k<0], resy['fitted_sigmasql']*1e6, ':', marker=None, color=c['yl'], label='y-fit (left)'))

## right fits
ln.append(ax.plot(k[k>0], resx['fitted_sigmasqr']*1e6, '-.',  marker=None, color=c['xr'], label='x-fit (right)'))
ln.append(ay.plot(k[k>0], resy['fitted_sigmasqr']*1e6, '-', marker=None, color=c['yr'], label='y-fit (right)'))

## Shrink plot by X% to put legend below
box = ax.get_position()
shrink_factor = 0.05
ax.set_position([box.x0, box.y0 + box.height * shrink_factor,
                 box.width, box.height * (1-shrink_factor)])
ay.set_position([box.x0, box.y0 + box.height * shrink_factor,
                 box.width, box.height * (1-shrink_factor)])

## Legend
labs = []
for i,itm in enumerate(ln):
    if type(itm) == list:
        labs.append(itm[0].get_label())
        ln[i]=itm[0]
    else:
        labs.append(itm.get_label())

leg = ax.legend(ln, labs, fontsize=font['size'], loc='upper right', bbox_to_anchor=(1, -2*shrink_factor), prop = {'size': 9},
          fancybox=True, shadow=True, ncol=7)

fig.savefig(outfile + ".png")
fig.savefig(outfile + ".pdf")
plt.show()