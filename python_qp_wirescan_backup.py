import epics
import time
import numpy as np

pv1 = epics.PV('melba_020:scan:stoppos_set')
pv2 = epics.PV('melba_020:scan:start')
pv3 = epics.PV('melba_020:scan:run_get')
pv4 = epics.PV('melba_020:scan:trig_input_FU00_set')
pv_qd = epics.PV('melba_020:trip_q1:i_set')

iinitial = -0.12
ifinal =  0.08
di = 0.005 

def fahre_zu(i):
    print("Fahre Scanner zu Position {}".format(i))
    pv1.put(i)
    pv2.put(0)
    time.sleep(1.1)
    while pv3.value != 0:
        time.sleep(.5)
    pv2.put(1)

pv4.put(0)
time.sleep(.5)
fahre_zu(25) 

for i in np.arange(iinitial,ifinal,di):
    print('Setze QP auf Strom I = {}'.format(i))
    pv_qd.put(i)
    time.sleep(.5)
    pv4.put(32)
    time.sleep(.5)
    print('Starte Scan für Strom I = {}'.format(i))
    fahre_zu(125)
    time.sleep(.5)
    pv4.put(0)
    print('Scan abgeschlossen!')
    time.sleep(.5)
    print('Fahre zur Startposition zurück')
    fahre_zu(25)

