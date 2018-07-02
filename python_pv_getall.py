#!/usr/bin/python3
import datetime
import pandas as pd
import time
import os
import sys

from epics import caget


# Time 
now = datetime.datetime.now()

# PV store method
def get_pv(pvname, digits=10):
    try:
        value = round(caget(pvname), digits)
        if not pvname.endswith("on_get"): unit = caget(pvname+".EGU")
        else: unit = None
    except:
        value = None
        unit = None
    return value, unit


# PVs

## Magnets
pv_steerer = ['melba_{:03d}:s{}:{}'.format(sub, s, get) 
                for sub in range(10,70,10) for s in ['h', 'v'] for get in ['i_get', 'on_get']]
pv_solenoids = ['melba_010:so:i_get', 'melba_010:so:on_get']
pv_quad_trip_q = ['melba_{:03d}:trip_q{}:{}'.format(sub, n, get)
                for sub in [20,30,50] for n in range(1,4) for get in ['i_get', 'on_get']]
pv_quad_trip_s = ['melba_{:03d}:trip_s{}{}:{}'.format(sub, s, n, get)
                for sub in [20,30,50] for s in ['h', 'v'] for n in range(1,3) for get in ['i_get', 'on_get']]
pv_alpha = ['melba_030:a:i_get', 'melba_030:a:on_get']

## Laser
pv_laser = ['steam:laser:setamp_get', 'steam:powme1:pow_get', 'steam:powme1:wavelength_get']

## All
pv_all = pv_steerer + pv_solenoids + pv_quad_trip_q + pv_quad_trip_s + pv_alpha + pv_laser


# Miscellanous
delay = 0.1
df = pd.DataFrame(index=pv_all + ['timestamp'], columns=['values', 'units'])

# Main
for pv in pv_all:
    value, unit = get_pv(pv)
    df.loc[pv] = [value, unit] 
    print("{} = {} {}".format(pv, value, unit))
    time.sleep(delay)

df.loc['timestamp'] = [now.strftime(fmt) for fmt in ["%Y-%m-%d", "%H:%M"]]
print(df)


# Path
path_prefix, file_prefix = [now.strftime(fmt) for fmt in ["%y%m%d", "%y%m%d_%H%M"]]
path = "pv_getall/{}".format(path_prefix)
if not os.path.isdir(path):
    print("Directory does not exist yet, making dir {}".format(path))
    try:
        os.system("mkdir -p {}".format(path))
    except:
        print("Could not make picture directory, aborting")
        sys.exit(0)
outfile = "{}/{}_pv_getall.csv".format(path, file_prefix)
# Save
df.to_csv(outfile)
