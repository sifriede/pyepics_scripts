# coding: utf-8

""" 
    This script was written to sort the i_get values that were not correctly assigned, 
    it should be obsolete now.
"""

import numpy as np
import glob

def get_lists(pic_path):
    c0, c1, cgl = [], [], []
    for f in glob.glob(pic_path + "*.npz"):
        with np.load(f) as file:
            if not f.endswith("background.npz"):
                pic = f[len(pic_path):-4]
                curr_get = file[pic].all()['curr_get']
                c0.append(f)
                c1.append(curr_get)
                cgl.append([f, curr_get])
                print("{} : {}, {}A".format(f, pic, curr_get))
    
    return c0, c1, cgl

def rotate(l, n):
    return l[-n:] + l[:-n]
    
def split(l):
    c0, c1 = [], []
    for i in range(len(cgl)):
        c0.append(cgl[i][0])
        c1.append(cgl[i][1])
    return c0, c1

def overwrite_curr_get(pic, curr_get):
    file = np.load(pic)
    print("Picture {} successfully loaded".format(pic))
    key = file.keys()[0]
    print("Key {}".format(key))
    d = file[key].all()
    d['curr_get'] = curr_get
    print("curr_get successfully overwritten")
    d = {key : d}
    file.close()
    np.savez_compressed(pic, **d)    


if __name__ == "__main__":
    pic_path = '180704_1317/'
    
    c0, c1, cgl = get_lists(pic_path)
    
    c1 = rotate(c1, -1)
    
    for i in zip(c0, c1):
        print(i)
    
    print("\n")    
    c1  =  c1[0:10] + rotate(c1[10:],2)
    
    for i in zip(c0, c1):
        print(i)
    