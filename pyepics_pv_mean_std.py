import numpy as np
import epics
import sys

from time import sleep


# Defaults
no_mean = 50
delay = 0.2
argv_range = len(sys.argv)


if len(sys.argv) == 1:
    print("Please give a list of pvs, and (optionally) the number of measurements and a delay time")
    sys.exit()

else:
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

# PVs
try:
    pv_all = [epics.PV(itm) for itm in sys.argv[1:argv_range]]
except:
    print("Could not build PV object from: {}".format(sys.argv[1:argv_range]))
    sys.exit()

# Miscellanous varibales
try:
    in_len = len(str(no_mean))
except:
    in_len = 4
res = []


# Calculation
print("Gathering PVs: {}".format([pv.pvname for pv in pv_all]))
print("Numbers to mean over: {}".format(no_mean))
print("Delay time: {}s".format(delay))
print("")

print("Measured:")
for i in range(1, no_mean+1):
    print("{:={digits}d}".format(i, digits=in_len), end=" ", flush=True)
    if i > 1 and i % 10 == 0: 
        print("")
    sleep(delay)
    res.append([pv.get() for pv in pv_all])
print("\nFinished")

# Mean and std
mean = np.mean(res, axis=0)
std = np.std(res, axis=0)
print("===== Results =====")
for pv, m, s in zip(pv_all, mean, std):
    print("{} = ({:=+05.3g} +/- {:=+05.3g}){} ({:.3f}%)".format(pv.pvname, m, s, pv.units, abs(s/m)*100))

# Close laser shutter
print("Done!")
