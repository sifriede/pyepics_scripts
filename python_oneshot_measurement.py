import sys
import numpy as np
import time
import epics

if len(sys.argv) != 5:
    print("Usage: python3 {} <quad_no=1> <i_init=-0.05> <i_final=0.15> <di=0.05>, shutter can be closed, picamera should be ready for trigger".format(sys.argv[0])) 
    sys.exit()

# Methods
def abort_script(DISP = False):
    """Script to close shutter or turn of laser an abort script"""
    if not DISP:
        try:
            epics.caput('steam:laser_shutter:ls_set', 0)
            epics.caput('steam:laser:dc_set', 0)
        except:
            epics.caput('steam:laser:on_set', 0)
    sys.exit()


def check_laser():
    """Safety function to check if laser power is too high while waiting"""
    if pv_laser['pow_get'].get() >= 5e-6:
        print("Laser power too high! Aborting!")
        abort_script()


def set_quadrupol(i_set, i_set_pv, i_get_pv, shot_mode=False):
    """Sets quadrupol to i_set and waits until i_get is updated accordingly"""
    print("Quadrupol i_set: {:+05d}mA".format(int(i_set*1000)))
    i_set_pv.put(i_set)
    info = True
    while not (i_set*0.9 <= i_get_pv.get() <= i_set + 1e-3) and not (i_set*0.9 >= i_get_pv.get() >= i_set):
        if info:
           print("Waiting for i_get to be updated...")
           info = False
        time.sleep(0.1) 
        if shot_mode: check_laser()
    return i_get_pv.get()


# Try communication with laser shutter
try:
    epics.caget('steam:laser_shutter:ls_set')
    epics.caput('steam:laser:dc_set', 0)

except:
    print("Could not read laser shutter status or dc could not be set off.")
    print("Aborting script.")
    abort_script()

else:
    if epics.caget('steam:laser_shutter:ls_set.DISP') == 1:
        print("Laser shutter can not be opened due to ls_set.DISP=1. Aborting script.")
        abort_script(True)
    if epics.caget('steam:laser:trig_get') != 1:
        epics.caput('steam:laser:trig_set',1)

#Arguments
try:
    quad_no = int(sys.argv[1])
    i_init = float(sys.argv[2])
    i_final = float(sys.argv[3])
    di = abs(float(sys.argv[4]))
except:
    print("Could not convert input. Taking defaults.")
    quad_no = 1 
    i_init = -0.2 
    i_final = 0.2
    di = 0.1 

if i_init > i_final:
    di = -di
nelm = int((i_final + di - i_init)/di)

#PVs
print("Debug")
pv_qd_param = ['i_set', 'i_get', 'on_set']
pv_laser = {'setamp_get': epics.PV('steam:laser:setamp_get'), 'pow_get': epics.PV('steam:powme1:pow_get')}
pv_all = dict()
for qd_param in pv_qd_param:
    pv_all[qd_param] = epics.PV('melba_020:trip_q{}:{}'.format(quad_no, qd_param))

#Miscellanous
wait_for_camera = 20  # Time in seconds to wait between switching to the next current step
trig_delay = round(epics.caget('steam:laser:trig_delay_get')*1e-6)  # Trigger delay converted from us to s

# Confirmation dialog
print("This script scans through quadrupol melba_020:trip_q{}.".format(quad_no))
print("It ramps current from {} to {} with stepwidth {} <=> {} pictures.".format(i_init, i_final, di, nelm))
print("It triggers a picture for each current and saves them seperately in an .npz-file and .png-file.")
confirmation = input("Do you want to continue? [Yy]")
if not any(confirmation == yes for yes in ['Y','y']):
    print("Aborting.")
    sys.exit(0)

print("Prepare: HV Voltage and laser amplitude (Attention: not to high as for screen gets destroyed!)")
print("Prepare: Drive SCREEN in beamline")
print("Prepare: Deactivate all following magnetic devices after the selected quadrupol.")
print("Prepare: Start PICAMERA and have it be READY FOR TRIGGER.")
print("Laser shutter will be opened by script")
response = input("Ready? [Gg]")
if not any(response == yes for yes in ['G','g']):
    print("Aborting.")
    sys.exit(0)

#Safety checks
epics.caput('steam:laser:dc_set', 0)
check_laser()

# Measurement loop
try:
    for pic, i_set in enumerate(np.linspace(i_init, i_final, nelm)):
        i_set = np.round(i_set, 6)
        # Time estimation
        if i_set == i_init:
            start = time.time()
    
        ## Quadrupol
        i_get = set_quadrupol(i_set, pv_qd_set, pv_qd_get, True)
        print('Quadrupol bereit. Oeffne Shutter.')        

        time.sleep(1)
        print("Quadrupol i_get: {:+05d}mA".format(int(pv_all['i_get'].get()*1000)))
        epics.caput('steam:laser_shutter:ls_set', 1)
        print("Trigger...!")
        epics.caput('steam:laser:sh_set.PROC', 1)
        time.sleep(trig_delay + 1) # Wait at least 
        print("Pew!")
        epics.caput('steam:laser_shutter:ls_set', 0)
        print("Waiting for camera {}s...".format(wait_for_camera))
        wait = 0.0
        while wait < float(wait_for_camera):
            check_laser()
            wait += 0.1
            time.sleep(.1)    
        stop = time.time()
        no_steps = (i_final+di-i_set)/di
        print("Estimated remaining time: {:.3f}s".format(no_steps*(stop-start)))
        start = time.time()
        print("======================================")

except KeyboardInterrupt:
    abort_script()  

else:
    print("Done!") 
