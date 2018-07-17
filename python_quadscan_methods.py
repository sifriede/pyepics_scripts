import datetime
import epics
import math
import time
import numpy as np
import pandas as pd
import subprocess
import sys
import os


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


def fahre_zu(pos, pv_scan_stoppos, pv_scan_start, pv_scan_run_get):
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
    interval = 0.075
    time.sleep(3)
    wait_start_time = time.time()
    while (not (i_set*(1-interval) <= i_get_pv.get() <= i_set * (1 + interval) + 3e-3) and 
          not (i_set*(1-interval) >= i_get_pv.get() >= i_set * (1 + interval))):
        if info:
           print("Waiting for i_get to be updated...")
           info = False
        time.sleep(0.1) 
        if shot_mode: check_laser()
        wait_stop_time = time.time()
        if wait_stop_time - wait_start_time > 15:
            break

    return i_get_pv.get()
