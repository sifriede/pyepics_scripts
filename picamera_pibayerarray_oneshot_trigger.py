#!/usr/local/bin/python3.6
import datetime
import RPi.GPIO as GPIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import picamera
import picamera.array
import time
import subprocess
import sys

from fractions import Fraction

try:
    import epics
    os.environ['PYEPICS_LIBCA'] = "/home/epics/base/lib/linux-arm/libca.so"
except:
    pass

# Methods
def plot_pic(outfile, imgarray, start_now):
    """Sum raw imgarray = img; plot img with colorbar and save to outfile."""
    fig, ax = plt.subplots(figsize=(11,8))
    img = sum(imgarray[:, :, i] for i in range(3))
    ax_img = ax.imshow(img, aspect='auto', cmap=plt.get_cmap('jet'))
    cb = plt.colorbar(ax_img, ax=ax)
    fig.text(0.01,0.01, start_now.strftime("%d.%m.%y %H:%M:%S"))
    plt.suptitle(outfile)
    plt.savefig(outfile + ".png")
    plt.close('all')


def myprint(idx_name, idx_value):
    """Simple print variable name and its value."""
    print("{} = {}".format(idx_name, idx_value))


def roi(x,p=0):
    """Return equally spaced distance around p."""
    roi_f = 800
    return int(round((x+(-1)**p*roi_f)/2))


def write_camtab(outfile, camera, camlist):
    """"unless pic_name is foo; open a file named cam_tab and write camlist to it."""
    if not pic_name == "foo":
        with open("{}/cam_tab.txt".format(pic_path), 'a') as f:
            try:
                f.write("{}\t".format(outfile))
                for k in camlist:
                    attr = getattr(camera, k)
                    f.write("{}\t".format(attr))
            finally:
                f.write("\n")


def exit_script(gpio = True):
    """Clean up GPIOs and exit script"""
    print("Aborting script.")
    if gpio: GPIO.cleanup()
    sys.exit(0)


def save_to_npz(outfile, imgid, imgarray, timestamp, cam, camlist):
    """Grep camera attributes and save pic,timestamp and specs to output"""
    try:
        id = "{:03d}".format(imgid)
    except:
        id = "{}".format(imgid)
    out = "{}_{}.npz".format(outfile,id)

    picam_specs = {}
    for k in camlist:
        attr = getattr(camera, k)
        picam_specs[k] = attr
        
    res_dict = {'img': imgarray, 'timestamp': now.strftime("%Y-%m-%d %H:%M:%S")}
    res_dict['picamera']=picam_specs
    res_dict = {id : res_dict}
    
    if os.path.isfile(out):
        print("File already exists: {}".format(out))
        override = input("Do you want to override it? [Y,y]")
        if not any(override == yes for yes in ['Y','y']):
            print("Picture not saved.")
            return None
    print("Saving...", end="")
    np.savez_compressed(out, **res_dict)
    print("Done")


# Timestamp
start_now = datetime.datetime.now()
path_time_prefix, pic_time_prefix = [start_now.strftime(pat) for pat in ["%y%m%d", "%y%m%d_%H%M_"]]
    
# Arguments
try:
    pic_name = sys.argv[1]
    num_of_pics = int(sys.argv[2])
    if len(sys.argv) == 4 and sys.argv[3] == 'nts':
        pic_name_save = pic_name
    else:
        pic_name_save = pic_time_prefix + pic_name

except:
    print("Usage: python3 {} <name> <number> (<nts>)".format(sys.argv[0]))
    exit_script(False)

# Path
pic_path = "oneshot_pics/{}/{}".format(path_time_prefix,pic_name_save)
if not os.path.isdir(pic_path):
    print("Picture directory does not exist yet, making dir {}".format(pic_path))
    try:
        os.system("mkdir -p {}".format(pic_path))
        with open("{}/cam_tab.txt".format(pic_path), 'a') as f:
            f.write("pic_name\t")
            for k in camlist:
                f.write("{}\t".format(k))
            f.write("\n")
    except:
        print("Could not make picture directory, aborting")
        exit_script(False)

# Miscellanous
res_time = []

# GPIO 
GPIO.setmode(GPIO.BOARD) ## Mode (BOARD = pin number)
## Pin setup
if GPIO.getmode() == 10:
    p_led = 12
    p_trig = 16  
GPIO.setup(p_led, GPIO.OUT, initial=GPIO.LOW)  # LED
GPIO.setup(p_trig, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Trigger by default on 0V

# Camera attributes to save
camlist = ['sensor_mode', 'iso', 'shutter_speed', 'exposure_mode', 
                'awb_mode', 'awb_gains', 'framerate', 'digital_gain', 'analog_gain', 'revision', 'exposure_speed']


# Confirmation dialog
print("Take {} picture named: {}".format(num_of_pics, pic_name_save))
print("It will be saved here: {}".format(pic_path))
try:
    response = input("Do you want to continue? [Yy,b]")
    if not any(response == yes for yes in ['Y','y', 'b']):
        exit_script()
        
    # No trigger for background picture 'b'
    if response == 'b':
        background = True
    else:
        background = False
except KeyboardInterrupt:
    exit_script()
    
print("________________________________________")

# Webserver
# print("Stopping Webserver")
# subprocess.call(['/home/pi/RPi_Cam_Web_Interface/stop.sh'])

print("Starting PiCamera")
with picamera.PiCamera() as camera:
    with picamera.array.PiBayerArray(camera) as output:
        # Camera settings
        ## https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
        camera.start_preview()
        camera.sensor_mode = 2
        camera.framerate = 1
        #camera.framerate_range = (1/10, 30)
        time.sleep(1)
        camera.shutter_speed = 1000000
        #camera.awb_mode = 'auto'
        camera.exposure_mode = 'auto'
        camera.iso = 800
        print("Waiting for camera to setup:")
        for i in range(10, -1, -1): # Best is to wait 30s
            time.sleep(1)
            print("{:2d}".format(i), end=" ", flush=True)
        camera.exposure_mode = 'off'

        print("")
        print("________________________________________")
        myprint("camera.iso", camera.iso)
        myprint("camera.framerate", camera.framerate)
        myprint("camera.framerate_range", camera.framerate_range)
        myprint("camera.shutter_speed", camera.shutter_speed)
        myprint("camera.exposure_speed", camera.exposure_speed)
        myprint("camera.exposure_mode", camera.exposure_mode)
        myprint("camera.digital_gain", camera.digital_gain)
        myprint("camera.analog_gain", camera.analog_gain)
        print("________________________________________")
        
        if camera.analog_gain == 0 or camera.digital_gain == 0 or camera.exposure_speed == 0:
            print("Digital and/or analog gain is 0! Pixels will be black.")
            exit_script()
        
        # Define outfile
        outfile = "{}/{}".format(pic_path, pic_name_save)
        outfile_bg = "{}_bg".format(outfile)
        
        # Take Picture
        if not background:
            # First image as background image
            start = time.time()
            camera.capture(output, 'jpeg', bayer=True)
            stop = time.time()
            print("Background picture was taken within {:.3g}s!".format(stop-start))
            res_time.append(stop-start)
            imgarray = output.array
            now = datetime.datetime.now()
            plot_pic(outfile_bg, imgarray , now)
            write_camtab(outfile, camera, camlist)
            save_to_npz(outfile, 'bg', imgarray, now, camera, camlist)
            print("________________________________________")
            
            for pic_id in range(num_of_pics):
                outfile_pic = "{}_{:03d}".format(outfile, pic_id)
                #time.sleep(2)
                print("pic_id = {}".format(pic_id))
                print("Camera is ready. Waiting for trigger signal...".format(pic_name), flush=True)
                print("Trigger ready")
                
                try:
                    timeout = 300000
                    trigger = GPIO.wait_for_edge(p_trig, GPIO.RISING, timeout = timeout)  
                    
                    if trigger:
                        print("Triggered!")
                        start = time.time()
                        camera.capture(output, 'jpeg', bayer=True)
                        stop = time.time()
                        print("Picture {:02d} was taken within {:.3f}s".format(pic_id, stop-start))
                        res_time.append(stop-start)
                        
                        # Image processing
                        imgarray = output.array
                        now = datetime.datetime.now()

                        ## Plot
                        plot_pic(outfile_pic, imgarray , now)
                        write_camtab(outfile, camera, camlist)
                        save_to_npz(outfile, pic_id, imgarray, now, camera, camlist)

                        print("Pictures taken successfully, saving to: {}".format(outfile), flush=True)
                        print("________________________________________")
                        
                    else:
                        print("Timeout: no trigger detected withing {}ms.".format(timeout))
                        print("________________________________________")
                        break
                        
                # while not triggered:
                    # try:
                        # time.sleep(0.01)
                except KeyboardInterrupt:
                    exit_script()
                    
        # Background picture
        else:
            outfile = "{}/{}".format(pic_path, pic_name_save)
            print("Taking background picture!")
            camera.capture(output, 'jpeg', bayer=True)
            imgarray = output.array
            now = datetime.datetime.now()
            plot_pic(imgarray, outfile, now)
            write_camtab(outfile, camera, camlist)
            print("Pictures taken successfully, saving to: {}".format(outfile), flush=True)
            
        # Image processing
        # yc,xc = output.array.shape[:2]
        # imgarray_roi = output.array[roi(yc,1):roi(yc),roi(xc,1):roi(xc),:]
        

print("<time to take picture> = ({:.3f} +/- {:.3f})s".format(np.mean(res_time), np.std(res_time)))
print("Stopping picamera", flush=True)
#np.savez_compressed(outfile + ".npz", **pics_all)

    
# Start Webserver
# print("Starting Webserver", flush=True)
# subprocess.call(['/home/pi/RPi_Cam_Web_Interface/start.sh'])

GPIO.cleanup()
print("Done")