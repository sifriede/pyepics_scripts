import datetime
import epics
import math
import time
import numpy as np
import pandas as pd
import subprocess
import sys
import os



if len(sys.argv) != 5:
    print("Usage: python3 {}".format(sys.argv[0]) + 
        " <quad_no=1> <i_init=-0.05> <i_final=0.15> <di=0.05>, shutter can be closed")
    sys.exit()


# Methods
def abort_script(DISP = False):
    """Script to close shutter or turn of laser an abort script"""
    if not DISP:
        try:
            print("Closing shutter and setting dc off")
            epics.caput('steam:laser_shutter:ls_set', 0)
            epics.caput('steam:laser:dc_set', 0)
            pv_scan_trig_input.put(0)  # Cyclic Trigger off
        except:
            epics.caput('steam:laser:on_set', 0)
    try:
        if os.path.exists(outfile): 
            print("Deleting {}".format(outfile))
            os.system("rm {}".format(outfile))
    except:
        pass
            
    sys.exit()


def check_laser():
    """Safety function to check if laser power is too high while waiting"""
    if pv_laser['pow_get'].get() >= 5e-6:
        print("Laser power too high! Aborting!")
        abort_script()


def fahre_zu(pos):
    """Moves Scanner to position and handles waiting before stopping"""
    print("Fahre Scanner zu Position {}".format(pos))
    pv_scan_stoppos.put(pos)
    pv_scan_start.put(0)
    time.sleep(1.1)
    while pv_scan_run_get.value != 0:
        time.sleep(.5)
    pv_scan_start.put(1)


def set_quadrupol(i_set, i_set_pv, i_get_pv, shot_mode=False):
    """Sets quadrupol to i_set and waits until i_get is updated accordingly"""
    print("Quadrupol i_set: {:+05d}mA".format(int(i_set*1000)))
    i_set_pv.put(i_set)
    info = True
    """condition1 for i_set >= 0; condition2 for i_set < 0"""
    while not (i_set*0.95 <= i_get_pv.get() <= i_set * 1.05 + 1e-3) and 
          not (i_set*0.95 >= i_get_pv.get() >= i_set * 1.05):    
    #while not (i_set - 2e-3 <= i_get_pv.get() <= i_set + 2e-3):
        if info:
           print("Waiting for i_get to be updated...")
           info = False
        time.sleep(0.1) 
        if shot_mode: check_laser()
    return i_get_pv.get()

# Scanner
scanner = "melba_050:scan"

# Quadrupol specifications
#Arguments
try:
    quad_no = int(sys.argv[1])
    i_init = float(sys.argv[2])
    i_final = float(sys.argv[3])
    di = abs(float(sys.argv[4]))
except:
    print("Could not convert input. Taking defaults.")
    quad_no = 1 
    i_init = -0.1
    i_final =  0.22
    di = 0.01

if i_init > i_final:
    di = -di
nelm = int((i_final + di - i_init)/di)

#Scanner start- and endposition
s_start = 30
s_end = 120

# Scanner PV
pv_scan_stoppos = epics.PV(scanner+ ':stoppos_set')
pv_scan_start = epics.PV(scanner+ ':start')
pv_scan_run_get = epics.PV(scanner+ ':run_get')
pv_scan_trig_input = epics.PV(scanner+ ':trig_input_FU00_set')
pv_datach0_nos = epics.PV(scanner+ ':datach0_noofsamples_get')

# Quadrupol PV
pv_qd_set = epics.PV('melba_050:trip_q{}:i_set'.format(quad_no))
pv_qd_get = epics.PV('melba_050:trip_q{}:i_get'.format(quad_no))

# Confirmation dialog
print("This script scans through quadrupol melba_050:trip_q{}.".format(quad_no))
print("It ramps current from {} to {} with stepwidth {} <=> {} wirescans.".format(i_init, i_final, di, nelm))
print("For each current: drive scanner to start pos, open shutter, start measurement, drive to end pos, stop measurement close shutter.")
confirmation = input("Do you want to continue? [Yy]")
if not any(confirmation == yes for yes in ['Y','y']):
    print("Aborting.")
    sys.exit(0)

# Check if datastorage already contains values
print("Checking, if datastorage already contains measurements...", end="", flush=True)
if pv_datach0_nos.get() != 0:
    print("\n==============================================")
    proceed = input("Datastorage not empty! Clear before proceed.[proceed,p]")
    if not any(proceed == ans for ans in ['proceed','p']):
        print("Aborting.")
        abort_script()
else:
    print("Storage empty. Proceeding.")

# Check aver_set
averages = ['ai0', 'pmt']
aver_sets = [epics.caget("{}_{}:aver_get".format(scanner, ch)) for ch in averages]
if not all(aver_sets[0] == aver_sets[i] for i in range(len(aver_sets))):
    print("==============================================")
    proceed2 = input("Averages of channel 0 to 2 are not the same! Change before proceed.[proceed,p]")
    if not any(proceed2 == ans for ans in ['proceed','p']):
        print("Aborting.")
        abort_script()

# Preparation dialog
print("==============================================")
print("Prepare:")
print("1. HV Voltage and laser amplitude")
print("2. Close laser shutter, it will be opened and closed by script")
print("3. Deactivate all following magnetic devices after the selected quadrupol.")
print("4. Set DC on")
response = input("Ready? [Gg]")
if not any(response == yes for yes in ['G','g']):
    print("Aborting.")
    sys.exit(0)

subprocess.call(['python3', '/home/melba/messung/All_quadscan/python_pv_getall.py'])
        
# Path
now = datetime.datetime.now()
path_prefix, file_prefix = [now.strftime(fmt) for fmt in ["%y%m%d", "%y%m%d_%H%M_"]]
path = "/home/melba/messung/All_quadscan/wirescan/{}/".format(path_prefix)
if not os.path.isdir(path):
    print("Directory does not exist yet, making dir {}".format(path))
    try:
        os.system("mkdir -p {}".format(path))
    except:
        print("Could not make directory, aborting")
        sys.exit(0)
else:
    print("Results will be saved in directory {}".format(path))

# Info PVs for each scan
pv_list = [epics.PV(pv) for pv in 
                [ 'steam:laser:setamp_get',   'steam:laser:dc_get', 'steam:laser:pl_get', 'steam:laser:dt_get', 
                    'steam:laser:nos_get', 'steam:powme1:pow_get',
                    scanner + '_pmt:aver_get', scanner + '_pmt:dout_gain_get', scanner + '_pmt:offset_get',
                    scanner + '_ai0:aver_get', scanner + '_ai0:dout_gain_get', scanner + '_ai0:offset_get',
                    scanner + ':maxvel_get', scanner + ':trig_cycle_get', 
                ]]
pv_info = []  # This list will be saved at the end
for pv in pv_list:
    value = round(pv.get(),10)
    temp = "{} = {} {}".format(pv.pvname, value, pv.units)
    pv_info.append(temp)
pv_info.append("Measured on {}".format(now.strftime("%d.%m.%Y %H:%M")))
pv_info.append("Quadrupol {}".format(pv_qd_set.pvname))

outfile = "{}{}wirescan_info.dat".format(path, file_prefix)
print("Saving results to {}".format(outfile))
with open(outfile, 'w') as f:
    for pv in pv_info:
        f.write("# {}\n".format(pv))

# Result dataframe current vs nos_ch    
df = pd.DataFrame(columns=['i_set', 'i_get', 'nos_ch0'])
idf = 0

# Prepare scanner
pv_scan_trig_input.put(0)
time.sleep(.5)
fahre_zu(s_start) 
print("==============================================")
epics.caput('steam:laser_shutter:ls_set', 0)

try:
    for no, i_set in enumerate(np.linspace(i_init, i_final, nelm)):
        print("Run: {} of {}".format(no+1, nelm))
        i_set = np.round(i_set, 6)
        # Time estimation
        if i_set == i_init:
            start = time.time()
        
        # QP
        #print('Setze QP auf Strom I = {}'.format(i))
        #pv_qd_set.put(i)
    
        i_get = set_quadrupol(i_set, pv_qd_set, pv_qd_get)
        print('Quadrupol bereit. Oeffne Shutter.')
        epics.caput('steam:laser_shutter:ls_set',1)
        time.sleep(.5)
        
        pv_scan_trig_input.put(32)  # Cyclic Trigger on 
        time.sleep(.5)
        
        print('Starte Scan für Strom i_set = {}, i_get = {}'.format(i_set, i_get))
        fahre_zu(s_end)
        time.sleep(.5)
        
        pv_scan_trig_input.put(0)  # Cyclic Trigger off
        print('Scan abgeschlossen! Schliesse Shutter.')
        time.sleep(.5)
        
        epics.caput('steam:laser_shutter:ls_set',0)
        print('Fahre zur Startposition zurück')
        fahre_zu(s_start)
    
        df.loc[idf] = [i_set, i_get, pv_datach0_nos.get()]
        idf += 1
            
        stop = time.time()
        remaining_steps = nelm - no
        ert = round((stop - start)*remaining_steps)
        print("Estimated remaining time: {}s".format(ert))
        start = time.time()
        print("==============================================")

except KeyboardInterrupt:
    print("Aborting.")
    abort_script()

print("Messung abgeschlossen.")
# Save
print("Saving results to {}".format(outfile))
with open(outfile, 'a') as f:
    df.to_csv(f)
print("Done")
