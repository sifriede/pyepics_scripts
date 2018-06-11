import datetime
import epics
import numpy as np
import time
import sys

'''
    This script measures the reflected laser power a given number of times
    for a specific laser amplitude with a specific delay between the measurements.
    Its purpose is to show whether the the power measurement is gaussian distributed or not.
'''


def prefl_meas_hist(laser_amp, no_mean = 10000, delay = 0.2, powme_aver = 100):
    
    shutter_state = epics.caget("steam:laser_shutter:ls_get")
    # Laser Shutter closed 
    if  shutter_state != 0:
        print("Shutter not closed! Aborting.")
        return None
    
    # Time of Measurement
    timestamp = datetime.datetime.now()

    # Information output
    print("Set laser amplitude: {}".format(laser_amp))
    print("Power meter set averaging: {}".format(powme_aver))
    print("Number of measurements: {}".format(no_mean))
    print("Delay time: {}s".format(delay))
    print("")

    # PVs
    pv_pow = epics.PV("steam:powme1:pow_get")
    
    ## Set PVs
    epics.caput("steam:laser:dc_set", 0)
    time.sleep(0.2)
    epics.caput("steam:laser:amp_set", laser_amp)
    time.sleep(0.2)
    epics.caput("steam:powme1:aver_set", powme_aver)
    time.sleep(0.2)
    
    ## Check PVs
    amp = round(epics.caget("steam:laser:setamp_get"))
    aver = round(epics.caget("steam:powme1:aver_get"))

    # Miscellanous varibales
    try:
        in_len = len(str(no_mean))
    except:
        in_len = 4
    res = []

    # Measurement
    epics.caput("steam:laser:dc_set", 1)
    time.sleep(1)
    print("Measured:")
    for i in range(1, no_mean+1):
        if i == 1:
            start = time.time()
                
        print("{:={digits}d}".format(i, digits=in_len), end=" ", flush=True)
        
        if i > 1 and i % 10 == 0: 
            stop = time.time()
            ert = (stop - start)*(no_mean-i)
            print("\nEstimated remaining time: {:.2f}s".format(ert))
            start = time.time()
            print("")
            
        time.sleep(delay)
        res.append(pv_pow.get())
    print("Finished")
    
    # Switch of Laser
    print("Switching of Laser")
    epics.caput("steam:laser:dc_set", 0)
    time.sleep(0.2)
    epics.caput("steam:laser:amp_set", 1)
    time.sleep(0.2)
    epics.caput("steam:laser:dc_set", 0)
    
    # Mean and std
    mean = np.mean(res, axis=0)
    std = np.std(res, axis=0)/np.sqrt(no_mean)
    print("===== Results =====")
    print("{} = ({:=+05.3g} +/- {:=+05.3g}){} ({:.3f}%)".format(pv_pow.pvname, mean, std, pv_pow.units, abs(std/mean)*100))
    print("Done!")
    
    # Result
    result = {
        'laser_amp' : laser_amp, 'powme_aver' : powme_aver, 'no_mean' : no_mean, 'delay' : delay,
        'data' : np.array(res), 'data_mean' : mean, 'data_std': std, 'timestamp' : timestamp}
    
    return result
