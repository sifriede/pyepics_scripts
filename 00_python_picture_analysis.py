# coding: utf-8
import datetime
import glob
import python_method_2d_gaussian_fit as pm2g
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as tm
import os

reflected_laserp=""
##########################################################################
##########################################################################
# Start angle for first fit (going from negativ to positiv, e.g. -200mA, -180mA..., 0mA, ... +200mA)
theta_deg = 90.0
#=====Quadpics=====
pic_dir  = "../quad_pics/180704/"
path = "single/"
#path = "180704_1317/"
reflected_laserp = "(+0.0021 +/- +5.44e-07)W (0.026%)"
#path = "180704_1306/"
#reflected_laserp = "(+0.000242 +/- +2.64e-08)W (0.011%)"
#path = "180704_1114/"
#reflected_laserp = "(+0.000138 +/- +2.39e-08)W (0.017%)"
#path = "180704_1046/"
#reflected_laserp = "(+3.22e-05 +/- +2.02e-08)W (0.063%)"
#reflected_laserp = "(+3.09e-05 +/- +1.13e-08)W (0.037%)"

#=====Oneshot=====
# pic_dir  = "../oneshot_pics/180723/"
#path = "180723_1047/"
#reflected_laserp = "(+2.04e-05 +/- +1.82e-08)W (0.089%)"
#path = "180723_1059/"
#reflected_laserp = "(+2.04e-05 +/- +1.82e-08)W (0.089%)"
#path = "180723_1121/"
#reflected_laserp = "(+7.15e-05 +/- +2.23e-08)W (0.031%)"
# path = "180723_1215/"
# reflected_laserp="11800 Amp"

## Scale
px2mm = 20/(2480 - 1060)

##########################################################################
##########################################################################

# Miscellanous
pic_path = pic_dir + path
bckg = 'background.npz'


# Number of images to fit without background image
pics = [pic for pic in glob.glob(pic_path + "*mA.npz") if not pic[len(pic_path):].startswith("fit_")]
nelm = len(pics)  

# DataFrames
## Pictures
dp = pd.DataFrame(columns=['curr_set[mA]', 'picture'])
idp = 0
for pic in pics:
    try:
        curr_set_pic = int(pic[-11:-6])
    except:
        print("Could not extract set current from filename! Use scheme: qd<quad_no>s_<:=+4d>mA.npz")
        sys.exit()
    else:
        dp.loc[idp] = [curr_set_pic, pic]
        idp +=1
dp = dp.sort_values(['curr_set[mA]'], ascending = True)
dp = dp.reset_index(drop=True)

## Results
df = pd.DataFrame(columns = ['pic', 'curr_get[A]', 'sigma_x[px]', 'sigma_x_err[px]', 
                                            'sigma_y[px]', 'sigma_y_err[px]', 'theta[rad]', 'theta_err[rad]'])
idx = 0


start_start =tm.time()
# Background image for substraction
print("Loading background image: {}{}".format(pic_path, bckg))
img_bckg, time_bckg, curr_bckg = pm2g.load_image(pic_path, bckg)

theta = np.deg2rad(theta_deg)  # From here on calculating in radians
for i, f in dp.iterrows():
    f = f['picture']
    i+=1 
    pic = f[len(pic_path):]
    
    if not pic == bckg and not pic.startswith("fit_*"): 
        start = tm.time()
                
        result_path = "{}fit_{}".format(pic_path, pic[:-4])
        result_plot_path = result_path + ".png"
        
        print("====================")
        print("Loading {}/{}: {}{}".format(i, nelm, pic_path, pic))
        img, time_img, curr_get = pm2g.load_image(pic_path, pic)

        print("Subtracting background from image")
        img_wob = pm2g.subtract_background(img, img_bckg)
        print("Start fitting subtracted image")
        res = pm2g.two_dim_gaussian_fit(img_wob, pic, theta)
        print("Finished! Saving plot: {}".format(result_plot_path))
     
        theta = res['img_popt'][-1]

        sigmax, sigmax_err = res['img_popt'][3], np.sqrt(res['img_pcov'][3,3])
        sigmay, sigmay_err = res['img_popt'][4], np.sqrt(res['img_pcov'][4,4])
        theta_res, theta_res_err = res['img_popt'][-1], np.sqrt(res['img_pcov'][-1,-1])
        
        pm2g.plot_image_and_fit(pic, res['img'], res['img_fit'], result_plot_path, 
            time_img, curr_get, sigmax, sigmax_err, sigmay, sigmay_err)
            
        pm2g.save_result(pic[:-4], res, curr_get, result_path)
        
        df.loc[idx] = [pic, curr_get, sigmax, sigmax_err, sigmay, sigmay_err, theta_res, theta_res_err]
        idx +=1
        
        stop = tm.time()
        ert = (stop - start)*(nelm - i)
        print("Done! Estimated remaining time: {:.2f}".format(ert))
        start = tm.time()

stop_stop = tm.time()
print("==================================================")
print("Calculation took {:.3f}s".format(stop_stop-start_start))
print("==================================================")

# DataFrame post processing
df = df.sort_values(['curr_get[A]'], ascending=True)
df = df.reset_index(drop=True)
for col in df.columns:
    if col.endswith('[px]'):
        df[col[:-4] +'[mm]'] = pd.Series(px2mm*df[col], index=df.index)


# Save result
outfile = pic_path + "result_DataFrame.dat"
if os.path.isfile(outfile):
    overwrite = input("{} already exists, do you want to overwrite it?[yes, please]\nIt will be prefixed by timestamp instead.")
    if overwrite  == 'yes, please':
        overwrite_prefix = ''
    else:
        now = datetime.datetime.now()
        overwrite_prefix = now.strftime("%y%m%d_%H%M_")

with open(outfile, 'w') as f:
    f.write("#Measured  {}\n".format(path[:-1]))
    f.write("#Scale px2mm =  {:.6f}\n".format(px2mm))
    f.write("#P_refl = {}\n".format(reflected_laserp))
    f.write("\n")
    df.to_csv(f, float_format='%11.3e')

# Plot
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
    "Measured  {}, ".format(path[:-1]) + 
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
    "Measured  {}, ".format(path[:-1]) + 
    "scale px2mm =  {:.6f} mm/px\n".format(px2mm) + 
    "P_refl = {}".format(reflected_laserp))
ax1.legend()
plt.savefig(outfile[:-4] + "_theta.png")
plt.show()