# coding: utf-8
import datetime
import glob
import python_method_2d_gaussian_fit as pm2g
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as tm
import os

##########################################################################
##########################################################################
#=====Wirescan=====
wire_dir  = "..\\wirescan\\"
wire_date = "180711\\"
quad_no = 1

## Scale
idx2mm = 152.175 / (65536 - 1)

##########################################################################
##########################################################################
wire_path = wire_dir + wire_date

# Wirescans in directory
wires = [wire for wire in glob.glob(wire_path + "*wirescan.csv") if not wire[len(wire_path):].startswith("fit_")]
nelm = len(wires)  

try:
    wires_info = ["{}_info.dat".format(wire[:-4]) for wire in wires]
except:
    print("Error finding corresponding info file")
    sys.exit()

cw = 0
wire_timestamp = wires[cw][len(wire_path):-13]
dw0 = pd.read_csv(wires[cw])
dwi0 = pd.read_csv(wires_info[cw], skiprows=pm2g.find_headers(wires_info[cw]))
indices0 = [int(i) for i in dwi0['nos_ch0']]
outdir = wire_dir + wire_date + wire_timestamp + "\\"
pm2g.make_dir(outdir)
#write_subdf_tofile(dw0[0:indices0[0]])
for i in range(len(indices0)-1):
    print("i = {}: {} - {}".format(i, indices0[i]+1, indices0[i+1]))
    file = outdir + "qd{}s_{:=+05d}mA.csv".format(quad_no, int(dwi0['i_set'][i] * 1000))
    print(file)
    pm2g.write_subdf_tofile(dw0[indices0[i]+1: indices0[i+1]], file)

#for i,idx in dwi0['nos_ch0'].iteritems():
        


## DataFrames
### Wirescan_infos
#dwinfo = pd.DataFrame(columns=['curr_set[mA]', 'wirescan'])
#idwinfo = 0
#for wire in wires:
#    try:
#        curr_set_wire = int(wire[-11:-6])
#    except:
#        print("Could not extract set current from filename! Use scheme: qd<quad_no>s_<:=+4d>mA.npz")
#        sys.exit()
#    else:
#        dwinfo.loc[idwinfo] = [curr_set_wire, wire]
#        idwinfo +=1
#dwinfo = dwinfo.sort_values(['curr_set[mA]'], ascending = True)
#dwinfo = dwinfo.reset_index(drop=True)
#
### Results
#df = pd.DataFrame(columns = ['wire', 'curr_get[A]', 'sigma_x[px]', 'sigma_x_err[px]', 
#                                            'sigma_y[px]', 'sigma_y_err[px]', 'theta[rad]', 'theta_err[rad]'])
#idx = 0
#
#
#start_start =tm.time()
## Background image for substraction
#print("Loading background image: {}{}".format(wire_path, bckg))
#img_bckg, time_bckg, curr_bckg = pm2g.load_image(wire_path, bckg)
#
#theta = np.deg2rad(theta_deg)  # From here on calculating in radians
#for i, f in dwinfo.iterrows():
#    f = f['wirescan']
#    i+=1 
#    wire = f[len(wire_path):]
#    
#    if not wire == bckg and not wire.startswith("fit_*"): 
#        start = tm.time()
#                
#        result_path = "{}fit_{}".format(wire_path, wire[:-4])
#        result_plot_path = result_path + ".png"
#        
#        print("====================")
#        print("Loading {}/{}: {}{}".format(i, nelm, wire_path, wire))
#        img, time_img, curr_get = pm2g.load_image(wire_path, wire)
#
#        print("Subtracting background from image")
#        img_wob = pm2g.subtract_background(img, img_bckg)
#        print("Start fitting subtracted image")
#        try:
#            res = pm2g.two_dim_gaussian_fit(img_wob, wire, theta)
#            print("Finished! Saving plot: {}".format(result_plot_path))
#            
#        except:
#            print("Could not fit {}".format(wire))
#        
#        else:    
#            theta = res['img_popt'][-1]
#
#            sigmax, sigmax_err = res['img_popt'][3], np.sqrt(res['img_pcov'][3,3])
#            sigmay, sigmay_err = res['img_popt'][4], np.sqrt(res['img_pcov'][4,4])
#            theta_res, theta_res_err = res['img_popt'][-1], np.sqrt(res['img_pcov'][-1,-1])
#            
#            pm2g.plot_image_and_fit(wire, res['img'], res['img_fit'], result_plot_path, 
#                time_img, curr_get, sigmax, sigmax_err, sigmay, sigmay_err)
#                
#            pm2g.save_result(wire[:-4], res, curr_get, result_path)
#            
#            df.loc[idx] = [wire, curr_get, sigmax, sigmax_err, sigmay, sigmay_err, theta_res, theta_res_err]
#            idx +=1
#            
#            stop = tm.time()
#            ert = (stop - start)*(nelm - i)
#            print("Done! Estimated remaining time: {:.2f}".format(ert))
#            start = tm.time()
#
#stop_stop = tm.time()
#print("==================================================")
#print("Calculation took {:.3f}s".format(stop_stop-start_start))
#print("==================================================")
#
## DataFrame post processing
#df = df.sort_values(['curr_get[A]'], ascending=True)
#df = df.reset_index(drop=True)
#for col in df.columns:
#    if col.endswith('[px]'):
#        df[col[:-4] +'[mm]'] = pd.Series(px2mm*df[col], index=df.index)
#
#
## Save result
#outfile = wire_path + "result_DataFrame.dat"
#if os.path.isfile(outfile):
#    overwrite = input("{} already exists, do you want to overwrite it?[yes, please]\nIt will be prefixed by timestamp instead.")
#    if overwrite  == 'yes, please':
#        overwrite_prefix = ''
#    else:
#        now = datetime.datetime.now()
#        overwrite_prefix = now.strftime("%y%m%d_%H%M_")
#
#with open(outfile, 'w') as f:
#    f.write("#Measured  {}\n".format(path[:-1]))
#    f.write("#Scale px2mm =  {:.6f}\n".format(px2mm))
#    f.write("#I_e     = {}\n".format(electron_current))
#    f.write("#P_refl = {}\n".format(reflected_laserp))
#    f.write("\n")
#    df.to_csv(f, float_format='%11.3e')
#
## Plot
#x = df['curr_get[A]']
#y1 = df['sigma_x[px]']**2
#y1err = df['sigma_x_err[px]']**2
#y2 = df['sigma_y[px]']**2
#y2err = df['sigma_y_err[px]']**2
#
#fig, ax1 = plt.subplots(figsize = (11.7,8.3))
#
#ax1.set_xlabel("Quadrupol current [A]", fontsize = 16)
#ax1.set_ylabel(r'$\sigma^2$ [px]', fontsize = 16)
#ax1.errorbar(x, y1, yerr=y1err, label="x", marker='o', linestyle='--', color = 'red', mfc='red', mec='black', ecolor='red')
#ax1.errorbar(x, y2, yerr=y2err, label="y", marker='^', linestyle='-.', color = 'blue', mfc='blue', mec='black', ecolor='blue')
#ax1.legend()
#ax1.grid()
#
#ax2 = ax1.twinx()
#ax2.set_ylim(np.array(ax1.get_ylim())*px2mm**2)
#ax2.set_ylabel(r'$\sigma^2$ [mm$^2$]', fontsize = 16)
#
#fig.text(0.01,0.01, 
#    "Measured  {}, ".format(path[:-1]) + 
#    "scale px2mm =  {:.6f} mm/px\n".format(px2mm) + 
#    "I_e = {}\n".format(electron_current) +
#    "P_refl = {}".format(reflected_laserp))
#ax1.legend()
#plt.savefig(outfile[:-4] + ".png")
#
##DEBUG
#fig2, ax1 = plt.subplots(figsize = (11.7,8.3))
#
#ax1.set_xlabel("Quadrupol current [A]", fontsize = 16)
#ax1.set_ylabel(r'$\sigma^2$ [px]', fontsize = 16)
#ax1.errorbar(x, y1, yerr=y1err, label="x", marker='o', linestyle='--', color = 'red', mfc='red', mec='black', ecolor='red')
#ax1.errorbar(x, y2, yerr=y2err, label="y", marker='^', linestyle='-.', color = 'blue', mfc='blue', mec='black', ecolor='blue')
#
##ax1.grid()
#
#y2_tick = np.arange(-360, 360+45, 45)
#ytheta = np.rad2deg(df['theta[rad]'])
#ax2 = ax1.twinx()
#ax2.plot(x, ytheta, 'g-', marker = '*')
#ax2.set_yticks(y2_tick)
#ax2.set_ylabel(r'$\theta$ [$\degree$]', fontsize = 16)
#ax2.grid()
#fig.text(0.01,0.01, 
#    "Measured  {}, ".format(path[:-1]) + 
#    "scale px2mm =  {:.6f} mm/px\n".format(px2mm) + 
#    "I_e = {}\n".format(electron_current) +
#    "P_refl = {}".format(reflected_laserp))
#ax1.legend()
#plt.savefig(outfile[:-4] + "_theta.png")
#plt.show()