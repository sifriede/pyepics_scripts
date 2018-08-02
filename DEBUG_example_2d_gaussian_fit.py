import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import math


# define model function and pass independant variables x and y as a list
def twoD_Gaussian(xydata, amplitude, xo, yo, sigma_x, sigma_y, offset, theta):
    xo = float(xo)
    yo = float(yo)
    x, y = xydata[0], xydata[1]
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    res = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return res.ravel()


# Create x and y indices
x = np.linspace(0, 200, 201)
y = np.linspace(0, 200, 201)
xydata = np.meshgrid(x, y)

# create data
theta = range(-360, 360+45, 45)#[-90, -15, 0, 45, 90, 135, 180, 270, 360]
xo, yo, r = 100, 100, 50
g2d = [twoD_Gaussian(xydata, 3, xo, yo, 20, 10, 10, np.radians(th)) for th in theta]
nelm = len(theta)
rowcols = math.ceil(np.sqrt(nelm))
f, axarr = plt.subplots(rowcols, rowcols, figsize=(6.5, 6), sharex = True, sharey = True)
f.tight_layout()

for i, ax in enumerate(axarr.reshape(-1)):
    if i < nelm:
        im = ax.imshow(g2d[i].reshape(201, 201), aspect="auto", cmap=plt.get_cmap('jet'))
        ax.set_title(r'$\theta$ = {}$\degree$'.format(theta[i]))
        ax.plot([0,200], [yo, yo], 'w--', lw=1)
        ax.plot([xo,xo], [0, 200], 'w--', lw=1)
        ax.plot([xo,xo+r*np.cos(np.radians(theta[i]))], [yo,yo- r*np.sin(np.radians(theta[i]))], 'w^-', lw=2)
plt.show()