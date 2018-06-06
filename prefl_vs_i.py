#!/usr/bin/python3
import datetime
import numpy as np
import epics
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
import sys

from time import sleep
from scipy.optimize import curve_fit
from scipy.odr import RealData, Model, ODR, Output, odr

# Confirmation dialog
response = input("This script ramps the laser and measures the current on the anode head. Do you want to continue? [Yy]")
if not any(response == yes for yes in ['Y','y']):
    print("Aborting.")
    sys.exit(0)


# PVs
pv_curr = [epics.PV('steam:anode:i_get')]
pv_qe = [epics.PV('steam:qe_{}:qe_get'.format(pv)) for pv in ['an','hv','bd']]
pv_power = [epics.PV(itm) for itm in ['steam:powme1:pow_get', 'steam:laser:pow_att_get']]

pv_all = pv_curr + pv_qe + pv_power
pv_laser = {'amp': epics.PV('steam:laser:amp_set') , 'dc': epics.PV('steam:laser:dc_set')}

# Laser
amp_range = range(7000,8100,100)

# Defaults
no_mean = 20
delay = 0.2
argv_range = len(sys.argv)
if 1 < len(sys.argv):
    try:
        no_mean = int(sys.argv[-2])
        delay = float(sys.argv[-1])
        argv_range = -2

    except:
        try:
            no_mean = int(sys.argv[-1])
            argv_range = -1
            print("A delay time was not given or could not be read.")
            print("Taking standard value.")

        except:
            print("Number of measurements or a delay time were not given or could not be read.")
            print("Taking standard values.")


# DataFrame
regex = re.compile(r'(?<=:)\w+:\w+(?=_)')
column_names = []
# df = pd.DataFrame(columns = ['Amp', 'Prefl', 'Prefl_std', 'P_att', 'P_att_std', 'I_anode', 'I_anode_std'])
for pv in pv_all:
    match = re.findall(regex, pv.pvname)[0]
    if match == []:
        match = pv.pvname
    val_name = "{}[{}]".format(match, pv.units)
    err_name = "{}_err[{}]".format(match, pv.units)
    column_names.append(val_name)
    column_names.append(err_name)
df = pd.DataFrame(columns = column_names)

# Miscellanous varibales
now = datetime.datetime.now().strftime("%y%m%d_%h%m_")
try:
    in_len = len(str(no_mean))
except:
    in_len = 4
res = []
df_idx = 0

# Calculation
print("Gathering PVs: {}".format([pv.pvname for pv in pv_all]))
print("Numbers to mean over: {}".format(no_mean))
print("Delay time: {}s".format(delay))
print("")

for amp in amp_range: 
    # Laser
    pv_laser['amp'].put(amp)
    sleep(0.5) 
    pv_laser['dc'].put(1)
    sleep(0.5) 
    
    # Measure
    print("Measured for laser amp {}: ".format(amp))
    for i in range(1, no_mean+1):
        print("{:={digits}d}".format(i, digits=in_len), end=" ", flush=True)
        if i > 1 and i % 10 == 0: 
            print("")
        sleep(delay)
        res.append([pv.get() for pv in pv_all])
    print("\nFinished")
    # Mean and std
    data = []
    mean,std  = np.mean(res, axis=0), np.std(res, axis=0)
    for i in range(len(mean)):
        data.append(mean[i])
        data.append(std[i])
    res = []
    # Save to DataFrame
    df.loc[df_idx] = data
    df_idx += 1

# Close laser shutter
print("\n Closing laser shutter!")
epics.caput("steam:laser_shutter:ls_set", 0)

# Results
print("===== Results =====")
print(df)
outfile = "{}qe_lsrramp".format(now)
df.to_csv(outfile + ".dat", sep="\t")

try:
    # x-y-Data
    plot_list = ['powme1:pow[W]', 'powme1:pow_err[W]', 'anode:i[A]', 'anode:i_err[A]']
    x, xerr, y, yerr = [df[idx] for idx in plot_list]


    def lm(B, x):
        return B[0]*x+B[1]
    
    # Fit
    data = RealData(x, y, sx=xerr, sy=yerr)
    linear = Model(lm)
    odr = ODR(data, linear, beta0=[1.,.0])
    fit_output = odr.run()
    yfit = lm(fit_output.beta, x)
    chisquared = sum(((y-yfit)*yerr)**2)

except:
    print("Could not fit and plot")

finally:
    # Plots
    fig, ax = plt.subplots(figsize = (16,12))
    ax.errorbar(x, y, xerr, yerr, label='Data', linestyle="None", mfc='gray', mec='black', ecolor='gray')
    ax.plot(x, yfit, label='ODR fit: {:.2f}*x{:+.2f}, chisq = {}'.format(*fit_output.beta, chisquared))
    ax.legend(loc='upper left')
    ax.set_title('Electron current on anode head vs reflected laser power')
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    fig.text(0, 0, "Date: {}".format(datetime.datetime.now().strftime("%d.%m.%y %H:%M")))
    fig.savefig(outfile + ".png")
    plt.show()

print("Done!")
