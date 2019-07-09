import sys
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit

import numpy as np

from astropy.io import fits
import csv

import lsst
import lsst.eotest.image_utils as imutils
from AmplifierGeometry import makeAmplifierGeometry

pixel_scale = 5
def tanh_fit(x,a,b,shift,scale):
    return b + shift*(1+np.tanh(2*(x-a)/scale))/2

def get_image_data(imagefile, amp):
    """Gets necessary data from given EOTest image

    Returns: bias, image, n_overscans
    bias - List of computed average biases by row
    image - np.array of brightnesses for given image
    n_overscans - number of rows of overscan in image"""

    image_untrimmed = lsst.afw.image.ImageF(imagefile,imutils.dm_hdu(amp))
    amp = makeAmplifierGeometry(imagefile)
    n_overscans = len(image_untrimmed.array)-amp.ny
    image = imutils.trim(image_untrimmed, amp.imaging)
    bias_fn = imutils.bias_row(image_untrimmed, amp.serial_overscan)
    bias = np.array([bias_fn(i) for i in range(len(image_untrimmed.array))])
    bias = bias[-amp.ny:] # exclude overscans

    return bias, image.array, n_overscans

def find_jumps(bias, n_smooth=30, threshold=1.5):
    """Returns points in bias where the derivative of the 
    smoothed function exceeds 'threshold' standard deviations"""

    bias_smoothed = sum([bias[i:i-n_smooth:n_smooth] \
            for i in range(n_smooth)])/n_smooth
    d_smooth = (bias_smoothed[1:] - bias_smoothed[:-1])/bias.std()
    return np.where(d_smooth > threshold)[0].flatten()

def compute_bias_shift(bias, image, n_overscans, plotsdir, *sensorinfo):
    """Finds shifts in a given list of row-computed bias values.

    Returns: 
        shifts - list of (shift, brightness) pairs, 
        where shift is the bias shift in ADU counts 
        and brightness is the largest pixel value in
        the vicinity of the shift."""

    shifts = []

    n_smooth = 30
    jumps_smooth = find_jumps(bias, n_smooth)
    
    if not (plotsdir is None): 
        fig = plt.figure()

    for i in range(len(jumps_smooth)):
        fig = plt.figure(figsize=(12,7))
        left_pixel  = int((jumps_smooth[i]-1)*n_smooth)
        right_pixel = int((jumps_smooth[i]+2)*n_smooth)
        if left_pixel  < 0: left_pixel = 0
        if right_pixel >= len(bias): right_pixel = len(bias) - 1
        
        n_window = right_pixel-left_pixel
        bias_window = bias[left_pixel:right_pixel]
        
        params, cov = curve_fit(tanh_fit, np.arange(n_window), bias_window,\
                            p0=[n_window/2,bias_window.mean(), \
                            max(bias_window)-min(bias_window),pixel_scale], \
                            bounds=((0,min(bias_window),0,2), \
                                (n_window-1, max(bias_window), \
                                    max(bias_window)-min(bias_window),10)))
    
        bright_low  = left_pixel+int(params[0]-params[-1]) + n_overscans
        bright_high = left_pixel+int(params[0]+3*params[-1]) + n_overscans 
        bright_val = np.amax(image[bright_low : bright_high,:])
        bright_loc = np.where(image == bright_val)

        shifts.append([ bright_val,params[2], *sensorinfo]) # sensor, segment name
        
        if plotsdir is None : continue

        plt.plot(bias_window)
        plt.plot(tanh_fit(np.arange(n_window), *params))
        plt.ylabel('Bias (ADU)')
        plt.text(params[0] + 10, int(params[1]+params[2]/2.) \
             ,'Shift: ' + '{:.2f} ADU\nBright Pixel: {:.2e} ADU'.format(params[2], bright_val))
        fig.patch.set_facecolor('white')
        plt.savefig(plotsdir + '/' + sensorinfo[0] + '-' + sensorinfo[1] + '-' + \
                sensorinfo[2] + '-' + sensorinfo[3] + '-' + sensorinfo[4] + '-' + str(i))
        plt.close(fig)

    return shifts
    
def main():
    
    parser = argparse.ArgumentParser(description='Find bias shifts in given EOTest images.')
    parser.add_argument('indir', type=str, nargs=1, help='Directory with input files')
    parser.add_argument('out', type=str, nargs=1, help='Output file for table of shifts')
    parser.add_argument('--plots', type=str, nargs='?',const='plots', help='Save plots of bias shift fits.')

    args = parser.parse_args()
    indir = args.indir[0] 
    outfile = args.out[0] 
    makeplots = hasattr(args,'plots')

    if makeplots:
        plotsdir = args.plots
        os.makedirs(plotsdir, exist_ok=True)
    else:
        plotsdir = None
    flats_directory = os.fsencode(indir)
    shifts = [['Bright Pixel (ADU)', 'Bias Shift (ADU)', \
            'RAFTNAME', 'LSST_NUM', 'RUNNUM', 'EXTNAME', 'IMAGETAG', 'Filename']]
    out = csv.writer(open(outfile,'w'))

    for root, dir, f in os.walk(flats_directory):
        for file in f:
            filename = os.path.join(root, file).decode('UTF-8');
            if filename.endswith('.fits'): 
                print('Checking file: ' + filename)
                header = fits.getheader(filename)
                if not 'RAFTNAME' in header:
                    header['RAFTNAME'] = 'single_sensor'
                for i in range(1,16+1):
                    segheader = fits.getheader(filename, ext=i) 
                    shifts = shifts + compute_bias_shift(*get_image_data( filename, i), \
                             plotsdir, header['RAFTNAME'], header['LSST_NUM'],  \
                             header['RUNNUM'], segheader['EXTNAME'], header['IMAGETAG'],filename)

    out.writerows(shifts)
    return

if __name__ == "__main__": main()
