#!/usr/bin/python3
import datetime
import numpy as np
import epics
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import os
import re
import seaborn as sns
import sys

from matplotlib.ticker import EngFormatter
from time import time, sleep
from scipy.optimize import curve_fit
from scipy.odr import RealData, Model, ODR, Output, odr


# Methods
def check_conditions():
    #Abort if conditions are not fullfilled
    print("Aborting")
    sys.exit(0)


# Change only here
# Defaults
num_mean = 20
delay = 0.2
# Laser
amp_low, amp_high, amp_step = 5000, 20000, 1000
amp_range = range(amp_low, amp_high+amp_step, amp_step)

# No CHANGES below here

# Confirmation dialog
print("This script ramps the laser from {} to {} with stepwidth {}.".format(amp_low, amp_high, amp_step))
print("It measures the current on the beam dump and fits I_beamdump(P_refl).")
print("=============================================================")
response = input("Do you want to continue? [Yy]")
if not any(response == yes for yes in ['Y','y']):
    print("Aborting.")
    sys.exit(0)

epics.caput("steam:laser_shutter:ls_set", 0)

drvh = epics.caget("steam:laser:amp_set.DRVH")
if drvh < amp_high:
    print("Warning: steam:laser:amp_set.DRVH < {}".format(amp_high))
    sys.exit(0)

if not 64 <= epics.caget("melba_020:scan:pos_get") <= 66 or not 63 <= epics.caget("melba_050:scan:pos_get") <= 64:
    print("Warning: Scanner are not in null position!")
    sys.exit(0)

#check_conditions()

# Time of Measurement
start_now = datetime.datetime.now()
path_prefix, file_prefix = [start_now.strftime(pat) for pat in ["%y%m%d", "%y%m%d_%H%M_"]]

# Path
path = "prefl_vs_ibeamdump/{}".format(path_prefix)
if not os.path.isdir(path):
    print("Directory does not exist yet, making dir {}".format(path))
    try:
        os.system("mkdir -p {}".format(path))
    except:
        print("Could not make directory, aborting")
        sys.exit(0)

# PV: Measure
pv_lsr = [epics.PV('steam:laser:setamp_get')]
pv_curr = [epics.PV('melba_060:bd:i_get')]
pv_qe = [epics.PV('steam:qe_{}:qe_get'.format(pv)) for pv in ['an','hv','bd']]
pv_power = [epics.PV(itm) for itm in ['steam:powme1:pow_get', 'steam:laser:pow_att_get']]
pv_volt = [epics.PV('steam:hv:u_get')]


pv_all = pv_lsr + pv_curr + pv_qe + pv_power + pv_volt

# PV: Control laser
pv_laser = {'amp': epics.PV('steam:laser:amp_set') , 'dc': epics.PV('steam:laser:dc_set')}

argv_range = len(sys.argv)
if 1 < len(sys.argv):
    try:
        num_mean = int(sys.argv[-2])
        delay = float(sys.argv[-1])
        argv_range = -2

    except:
        try:
            num_mean = int(sys.argv[-1])
            argv_range = -1
            print("A delay time was not given or could not be read.")
            print("Taking standard value.")

        except:
            print("Number of measurements or a delay time were not given or could not be read.")
            print("Taking standard values.")


# DataFrame
regex = re.compile(r'(?<=:)\w+:\w+(?=_)')
column_names = ['amp']
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
try:
    in_len = len(str(num_mean))
except:
    in_len = 4
res = []
df_idx = 0

# Calculation
print("Gathering PVs: {}".format([pv.pvname for pv in pv_all]))
print("Numbers to mean over: {}".format(num_mean))
print("Delay time: {}s".format(delay))
print("")

try:
    for amp in amp_range: 
        # Timer
        if amp == amp_low:
            start = time()
        else:
            stop = time()
            remaining_steps = (amp_high-amp) / amp_step
            ert = (stop-start)*remaining_steps
            print("Estimated remaining time: {:.2f}s".format(ert))
            start = time()
    
        # Laser
        pv_laser['amp'].put(amp)
        sleep(0.5) 
        pv_laser['dc'].put(1)
        sleep(0.5) 
        pv_laser['dc'].put(1)  # Sometimes, must be set twice (bug :-()
        sleep(0.5) 
        epics.caput("steam:laser_shutter:ls_set", 1)
        sleep(0.5) 
     
        # Measure
        print("Measured for laser amp {}: ".format(amp))
        for i in range(1, num_mean+1):
            print("{:={digits}d}".format(i, digits=in_len), end=" ", flush=True)
            if i > 1 and i % 10 == 0: 
                print("")
            sleep(delay)
            res.append([pv.get() for pv in pv_all])
        print("\nFinished")
    
        # Mean and std
        data = [amp]
        mean,std  = np.mean(res, axis=0), np.std(res, axis=0)/np.sqrt(num_mean)
        for i in range(len(mean)):
            data.append(mean[i])
            data.append(std[i])
        res = []
    
        # Save to DataFrame
        df.loc[df_idx] = data
        df_idx += 1

except KeyboardInterrupt:
    print("\n Closing laser shutter and setting laser amplitude to 0!")
    epics.caput("steam:laser_shutter:ls_set", 0)
    pv_laser['amp'].put(0)
    sleep(0.5) 
    pv_laser['dc'].put(0)
    print("Keyboard interrupt detected: Aborting.")
    sys.exit()

# Close laser shutter
print("\n Closing laser shutter and setting laser amplitude to 0!")
epics.caput("steam:laser_shutter:ls_set", 0)
pv_laser['amp'].put(0)
sleep(0.5) 
pv_laser['dc'].put(0)
sleep(2)
## Measure and mean steam:hv:u_get
pv_cathode_volt_std = np.std(df['hv:u[kV]'])/np.sqrt(len(df['hv:u[kV]']))
pv_cathode_volt = np.mean(df['hv:u[kV]'])
pv_cathode_volt_str = "({:.3f} +/- {:.3f}) kV".format(pv_cathode_volt, pv_cathode_volt_std)
print("\nsteam:hv:u_get = {}".format(pv_cathode_volt_str))

# Results
print("===== Results =====")
print(df)
outfile = "{}/{}qe_lsrramp".format(path, file_prefix)
df.to_csv(outfile + ".dat", sep="\t", float_format="%.3e")

try:
    # x-y-Data
    plot_list = ['powme1:pow[W]', 'powme1:pow_err[W]', 'bd:i[A]', 'bd:i_err[A]', 'qe_bd:qe[%]', 'qe_bd:qe_err[%]']
    x, xerr, y, yerr = [df[idx] for idx in plot_list[:4]]
    qe, qeerr = [df[idx] for idx in plot_list[-2:]]


    def lm(B, x):
        return B[0]*x+B[1]
        #return B[0]*x
    

    # Fit
    data = RealData(x, y, sx=xerr, sy=yerr)
    linear = Model(lm)
    odr = ODR(data, linear, beta0=[1.,.0])
    fit_output = odr.run()
    yfit = lm(fit_output.beta, x)
    weights = 1/yerr
    chisquared = sum(weights**2 * (y-yfit)**2)

except:
    print("Could not fit and plot")

finally:
    # Plots
    ## Figure and axes
    fig, ax = plt.subplots(figsize = (12,8))
    ax2 = ax.twinx()
    
    font = {'family' : 'sans-serif',
        'size'   : 11}
    matplotlib.rc('font', **font)
    
    ## Colors
    color = {'curr':'gray', 'qe': 'blue', 'curr_fit': 'orange', 'residuals' : 'cyan'}
    
    ## x Axis
    ax.set_xlabel("$P_{refl}$")                        
    formatterx = EngFormatter(unit='W')     
    ax.xaxis.set_major_formatter(formatterx)
    
    ## Anode current
    ln1 = ax.errorbar(x, y, xerr = xerr, yerr = yerr, label='Anode current', marker='o', linestyle="None", mfc=color['curr'], mec='black', ecolor=color['curr'])
    ax.set_ylabel("Beam dump current", color=color['curr'])
    formattery = EngFormatter(unit='A') 
    ax.yaxis.set_major_formatter(formattery)
    ax.tick_params(axis='y', labelcolor=color['curr'])
    
         
    ## QE
    ln2 = ax2.errorbar(x, qe, xerr = xerr, yerr = qeerr, label='QE', marker='v', linestyle="None", mfc=color['qe'], mec='black', ecolor=color['qe'])
    ax2.set_ylabel('QE', color=color['qe'])
    ax2.tick_params(axis='y', labelcolor=color['qe'])
    formattery2 = EngFormatter(unit='%')     
    ax2.yaxis.set_major_formatter(formattery2)
          
    ## Fit                                                                                              
    ln3 = ax.plot(x, yfit, label="ODR fit beam dump current\n"+r'${:.3e}\frac{{A}}{{W}}\cdot P_{{refl}} {:+.3e}A$, $\chi^2 = {:.2e}$'.format(*fit_output.beta, chisquared), color=color['curr_fit'])  
    
    ## Residuals                                                                                              
    ln4 = ax.plot(x, yfit-y, label='Residuals: (yfit-y)', color=color['residuals'], marker='^', mfc=color['residuals'], mec='black')  
    
    ## Legend
    lns = [ln1,ln2,ln3[0],ln4[0]]
    labs = [l.get_label() for l in lns]
    leg = ax.legend(lns, labs, loc='center right', fontsize=font['size'], framealpha=1)
                    
    ## Miscellanous            
    ax.set_title('Electron current on beam dump vs reflected laser power')
    ax.grid(color=color['curr'], which='both', axis='y')                                
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) 
    fig.text(0.01, 0.01, "Date: {}, num_mean = {}, delay = {}s, U_cathode = {}".format(datetime.datetime.now().strftime("%d.%m.%y %H:%M"), num_mean, delay, pv_cathode_volt_str))
    fig.savefig(outfile + ".png")
    plt.show()

print("Done!")
