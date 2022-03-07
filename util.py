
from turtle import shape
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
import matplotlib as mpl
from PIL import Image, ImageFilter
import PIL

import pandas as pd

def plot_1D(gmm,x,col):
  plt.hist(x,density=True)
  x = np.linspace(x.min(), x.max(), 100, endpoint=False)
  ys = np.zeros_like(x)

  i=0
  for w in gmm.phi:
      y=sp.multivariate_normal.pdf(x, mean=gmm.mean_arr[i], cov=gmm.sigma_arr[i])*w

      plt.plot(x, y)
      ys += y
      i+=1

  plt.xlabel(col)
  plt.plot(x,ys)
  plt.show()


def make_ellipses(gmm, ax):
    colors = ['turquoise', 'orange']
    for n, color in enumerate(colors):
        covariances = gmm.sigma_arr[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 3. * np.sqrt(2.) * np.sqrt(v)
        mean=gmm.mean_arr[n]
        mean=mean.reshape(2,1)
        print(mean)
        ell = mpl.patches.Ellipse(mean, v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')

def plot_2D(gmm,x,col,label):

    h = plt.subplot(111, aspect='equal')
    make_ellipses(gmm, h)



    plt.scatter(x[:,0],x[:,1],c=label['Species'],marker='x')
    plt.xlim(-3, 9)
    plt.ylim(-3, 9)
    plt.xlabel(col[0])
    plt.ylabel(col[1])
    #plot_cov_ellipse(gmm.sigma_arr[:,:,0],gmm.mean_arr[:,1],ax=ax[0,0])
    plt.show()

    # Python3 program change RGB Color
# Model to HSV Color Model

def rgb_to_hsv(array):
    array_return=[]
    for pixel in array:
        r,g,b = pixel
        # R, G, B values are divided by 255
        # to change the range from 0..255 to 0..1:
        r, g, b = r / 255.0, g / 255.0, b / 255.0

        # h, s, v = hue, saturation, value
        cmax = max(r, g, b) # maximum of r, g, b
        cmin = min(r, g, b) # minimum of r, g, b
        diff = cmax-cmin	# diff of cmax and cmin.
        # if cmax and cmax are equal then h = 0
        if cmax == cmin:
            h = 0	
        # if cmax equal r then compute h
        elif cmax == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        # if cmax equal g then compute h
        elif cmax == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        # if cmax equal b then compute h
        elif cmax == b:
            h = (60 * ((r - g) / diff) + 240) % 360
        # if cmax equals zero
        if cmax == 0:
            s = 0
        else:
            # s = (diff / cmax) * 100
            s = (diff / cmax)
        # compute v
        # v = cmax * 100
        v = cmax
        array_return.append([h,s,v])
    return np.array(array_return)

def rgb_to_hsv_matplotlib(array):
    return mpl.colors.rgb_to_hsv(array)

def plot_img_from_rgb(array):
    red = []
    green = []
    blue = []

    for pixel in array:
        r,g,b = pixel
        red.append(r/255)
        green.append(g/255)
        blue.append(b/255)

    red = np.array(red).reshape(60,80)
    green = np.array(green).reshape(60,80)  
    blue = np.array(blue).reshape(60,80)   

    # r, g, and b are 60x80 float arrays with values >= 0 and < 1.
    rgbArray = np.empty((60,80,3), 'uint8')
    rgbArray[..., 0] = red*256
    rgbArray[..., 1] = green*256
    rgbArray[..., 2] = blue*256
    img = Image.fromarray(rgbArray)
    img.show()

def plot_img_from_hsv(array):
    pass

def img_resize(array, size):

    red = []
    green = []
    blue = []

    for pixel in array:
        r,g,b = pixel
        red.append(r/255)
        green.append(g/255)
        blue.append(b/255)

    red = np.array(red).reshape(60,80)
    green = np.array(green).reshape(60,80)  
    blue = np.array(blue).reshape(60,80)   

    # r, g, and b are 60x80 float arrays with values >= 0 and < 1.
    rgbArray = np.empty((60,80,3), 'uint8')
    rgbArray[..., 0] = red*256
    rgbArray[..., 1] = green*256
    rgbArray[..., 2] = blue*256
    img = Image.fromarray(rgbArray)
    im1 = img.resize(size)
    sharpened_img = im1.filter(ImageFilter.SHARPEN)
    sharpened_img.show()