# coding: utf-8
import datetime
import glob
import python_method_2d_gaussian_fit as pm2g
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as tm
import os
import sys

from shutil import copy2

reflected_laserp=""
##########################################################################
##########################################################################
result_filename = "result_DataFrame.dat"
#=====Quadpics=====
pic_dir  = "../quad_pics/180704/"
path_single = "single/"
path_target = "180704_1317/"

## Scale
px2mm = 20/(2480 - 1060)

##########################################################################
##########################################################################

## Miscellanous
pic_single = pic_dir + path_single + result_filename
pic_target = pic_dir + path_target + result_filename

print("This script will overwrite {} with results from {}".format(pic_target, pic_single))
confirm = input("Do you want to continue?[do it!]")
if confirm != 'do it!':
    print("Aborting")
    sys.exit()
    

headers = []
with open(pic_target, 'r') as f:
    for line in f:
        if line[0] == '#':
            headers.append(line)

print("Reading files and replacing results")
df_single = pd.read_csv(pic_single, skiprows=len(headers), index_col=0)
df_target = pd.read_csv(pic_target, skiprows=len(headers), index_col=0)
for idx, row in df_single.iterrows():
    df_target_index = df_target[df_target['pic']==row['pic']].index[0]
    df_target.iloc[df_target_index] = row

with open(pic_target[:-4] + "_new.dat", 'w') as f:
    for header in headers:
        f.write(header)
        if header[-1] != "\n":
            f.write("\n")
    f.write("\n")
    df_target

print("Copying fit_*.png from {} to {}".format(pic_dir + path_single, pic_dir + path_target))
pics = [pic for pic in glob.glob(pic_dir + path_single + "*") if pic[len(pic_dir + path_single):].startswith("fit_")]

for pic in pics:
    copy2(pic, pic_dir + path_target)

## Plot
df = df_target
print("Replot results")
outfile = pic_target[:-4] + ".png"

x = df['curr_get[A]']
y1 = df['sigma_x[px]']**2
y1err = df['sigma_x_err[px]']**2
y2 = df['sigma_y[px]']**2
y2err = df['sigma_y_err[px]']**2

fig, ax1 = plt.subplots(figsize = (11.7,8.3))

ax1.set_xlabel("Quadrupol current [A]", fontsize = 16)
ax1.set_ylabel(r'$\sigma^2$ [px]', fontsize = 16)
ax1.errorbar(x, y1, yerr=y1err, label="x", marker='o', linestyle='--', color = 'red', mfc='red', mec='black', ecolor='red')
ax1.errorbar(x, y2, yerr=y2err, label="y", marker='^', linestyle='-.', color = 'blue', mfc='blue', mec='black', ecolor='blue')
ax1.legend()
ax1.grid()

ax2 = ax1.twinx()
ax2.set_ylim(np.array(ax1.get_ylim())*px2mm**2)
ax2.set_ylabel(r'$\sigma^2$ [mm$^2$]', fontsize = 16)

fig.text(0.01,0.01, 
    "Measured  {}, ".format(path_target[:-1]) + 
    "scale px2mm =  {:.6f} mm/px\n".format(px2mm) + 
    "P_refl = {}".format(reflected_laserp))
ax1.legend()
plt.savefig(outfile[:-4] + ".png")

#DEBUG
fig2, ax1 = plt.subplots(figsize = (11.7,8.3))

ax1.set_xlabel("Quadrupol current [A]", fontsize = 16)
ax1.set_ylabel(r'$\sigma^2$ [px]', fontsize = 16)
ax1.errorbar(x, y1, yerr=y1err, label="x", marker='o', linestyle='--', color = 'red', mfc='red', mec='black', ecolor='red')
ax1.errorbar(x, y2, yerr=y2err, label="y", marker='^', linestyle='-.', color = 'blue', mfc='blue', mec='black', ecolor='blue')

#ax1.grid()

y2_tick = np.arange(-360, 360+45, 45)
ytheta = np.rad2deg(df['theta[rad]'])
ax2 = ax1.twinx()
ax2.plot(x, ytheta, 'g-', marker = '*')
ax2.set_yticks(y2_tick)
ax2.set_ylabel(r'$\theta$ [$\degree$]', fontsize = 16)
ax2.grid()
fig.text(0.01,0.01, 
    "Measured  {}, ".format(path_target[:-1]) + 
    "scale px2mm =  {:.6f} mm/px\n".format(px2mm) + 
    "P_refl = {}".format(reflected_laserp))
ax1.legend()
plt.savefig(outfile[:-4] + "_theta.png")
plt.show()