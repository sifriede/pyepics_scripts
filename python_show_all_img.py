import math
import matplotlib.pyplot as plt
import numpy as np
import operator
import sys


def show_img(filename):
   
    loaded_file = np.load(filename)
    
    # Fonts
    fontdict = {'family': 'serif',
                      'color': 'black',
                      'size': 14,
                      'weight': 'normal',
                      'backgroundcolor': 'white'}
    figsize = (11,8)
    ticks_size = 12
 
    def image_show(ax, img, title):
        ax.set_title(label=title, x=0.01, y=.98, fontdict=fontdict, horizontalalignment='left', verticalalignment='top')
        return ax.imshow(img, aspect='auto', cmap=plt.get_cmap('jet'))

    print("Plotting pictures from file {}, please wait.".format(filename))
    
    # Single picture in file
    if len(loaded_file.keys()) == 2:
        try:
            img = loaded_file['img']
        except:
            k = loaded_file.keys()[1]
            img = loaded_file[k].all()['img']
        if img.ndim >= 1:
            img = sum(img[:, :, i] for i in range(img.ndim))
        
        ## Plot
        fig, ax = plt.subplots(figsize=figsize)
        img_show = image_show(ax, img, filename)
        fig.colorbar(img_show, ax=ax)
        plt.show()
        
        ## Return
        return img
    
    # Multiple pictures
    elif len(loaded_file.keys()) >= 3:
        images = dict()
        for k in loaded_file.keys():            
            if 'img' in loaded_file[k].all().keys():
                
                try:
                    curr_set = loaded_file[k].all()['curr_set']
                except:
                    curr_set = k
                    
                if k == 'background':
                    curr_set = 'background'
                    
                if loaded_file[k].all()['img'].ndim > 1:
                    img = sum([loaded_file[k].all()['img'][:,:,i] for i in range(3)])
                else:
                    img = loaded_file[k].all()['img']
                images[k] = {'img':img, 'curr_set':curr_set}
                
        ## Plot
        nelm = len(images)
        if nelm > 1:
            n_row_col = math.ceil(np.sqrt(nelm))
            fontdict['size'] = 6
            ticks_size = 5
        else:
            n_row_col = 1
        
        fig, axes = plt.subplots(figsize=(11.7,8.3), nrows = n_row_col, ncols = n_row_col, sharex = 'col', sharey = 'row')
        for i,k in enumerate(images):
            if n_row_col > 1:
                ax = axes.flat[i]
            else:
                ax = axes
            ax_img = image_show(ax, images[k]['img'],images[k]['curr_set'])
            cb = plt.colorbar(ax_img, ax=ax)
            plt.suptitle(filename)
            cb.ax.tick_params(labelsize=ticks_size)        
        ## Return
        return images    



if len(sys.argv) == 2:
    images = show_img(sys.argv[1])
    plt.show()
else:
    print("Please give a file name")


