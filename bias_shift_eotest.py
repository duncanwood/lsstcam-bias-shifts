import sys
import os
import argparse
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit

import numpy as np

from astropy.io import fits
import csv

import lsst
import lsst.eotest.image_utils as imutils
from lsst.eotest.sensor.AmplifierGeometry import makeAmplifierGeometry

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

try:
    os.mkdir('bias_shift_logs')
except FileExistsError:
    pass
fh = logging.FileHandler(f'bias_shift_logs/{time.time()}.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def tanh_fit(x,a,b,shift,scale):
    return b + shift*(1+np.tanh(2*(x-a)/scale))/2

def get_image_data(imagefile, amp_num, amp_geom):
    """Gets necessary data from given EOTest image

    Returns: bias, image, n_overscans
        bias - List of computed average biases by row
        image - np.array of brightnesses for given image
        n_overscans - number of rows of overscan in image"""
    
    image_untrimmed = lsst.afw.image.ImageF(imagefile,imutils.dm_hdu(amp_num))
    
    n_overscans = len(image_untrimmed.array)-amp_geom.ny
    image = imutils.trim(image_untrimmed, amp_geom.imaging)
    bias_fn = imutils.bias_row(image_untrimmed, amp_geom.serial_overscan)
    bias = np.array([bias_fn(i) for i in range(len(image_untrimmed.array))])
    bias = bias[-amp_geom.ny:] # exclude prescans

    return bias, image.array, n_overscans

def find_jumps(bias, n_smooth=30, threshold=1.5):
    """Returns points in bias where the derivative of the 
    smoothed function exceeds 'threshold' standard deviations"""

    bias_smoothed = sum([bias[i:i-n_smooth:n_smooth] \
            for i in range(n_smooth)])/n_smooth
    d_smooth = (bias_smoothed[1:] - bias_smoothed[:-1])/bias.std()
    return np.where(d_smooth > threshold)[0].flatten()

def compute_bias_shift(bias, image, n_overscans, plotsdir, *sensorinfo, jump_scale=5, smoothing_scale=30):
    """Finds shifts in a given list of row-computed bias values.

    Returns: 
        shifts - list of (shift, brightness) pairs, 
        where shift is the bias shift in ADU counts 
        and brightness is the largest pixel value in
        the vicinity of the shift."""

    shifts = []

    n_smooth = smoothing_scale
    jumps_smooth = find_jumps(bias, n_smooth)

    for i in range(len(jumps_smooth)):
        left_pixel  = int((jumps_smooth[i]-1)*n_smooth)
        right_pixel = int((jumps_smooth[i]+2)*n_smooth)
        if left_pixel  < 0: left_pixel = 0
        if right_pixel >= len(bias): right_pixel = len(bias) - 1
        
        n_window = right_pixel-left_pixel
        bias_window = bias[left_pixel:right_pixel]
        
        params, cov = curve_fit(tanh_fit, np.arange(n_window), bias_window,\
                            p0=[n_window/2,bias_window.mean(), \
                            max(bias_window)-min(bias_window),jump_scale], \
                            bounds=((0,min(bias_window),0,2), \
                                (n_window-1, max(bias_window), \
                                    max(bias_window)-min(bias_window),10)))
    
        bright_low  = left_pixel+int(params[0]-params[-1]) + n_overscans
        bright_high = left_pixel+int(params[0]+3*params[-1]) + n_overscans 
        bright_val = np.amax(image[bright_low : bright_high,:])
        bright_loc = np.where(image == bright_val)
        shift_row  = left_pixel + params[0]

        shifts.append([ bright_val,'{:.3f}'.format(params[2]), '{:.3f}'.format(shift_row),'{:.3f}'.format(params[3]), *sensorinfo]) # sensor, segment name
        
        if plotsdir is None :
            continue
            
        fig = plt.figure(figsize=(12,7))

        row_range = np.arange(left_pixel, right_pixel)
        plt.plot(row_range, bias_window)
        plt.plot(row_range, tanh_fit(np.arange(n_window), *params))
        plt.xlabel('Row in segment (pixels)')
        plt.ylabel('Bias (ADU)')
        plt.text(left_pixel + params[0] + 10, int(params[1]+params[2]/2.) \
             ,'Shift: ' + '{:.2f} ADU\nBright Pixel: {:.2e} ADU'.format(params[2], bright_val))
        fig.patch.set_facecolor('white')
        plt.savefig(plotsdir + '/' + sensorinfo[0] + '-' + sensorinfo[1] + '-' + \
                sensorinfo[2] + '-' + sensorinfo[3] + '-' + sensorinfo[4] + '-' + str(i))
        plt.close(fig)

    return shifts

def find_shifts_in_files(filenames, outfile, plotsdir=None, append=False, report_negatives=True, cache_biases=None):
    
    processed_files = {}
    
    if plotsdir is not None and type(plotsdir) == str:
        os.makedirs(plotsdir, exist_ok=True)
    else:
        plotsdir = None
        
    if cache_biases is not None and type(cache_biases) == str:
        os.makedirs(cache_biases, exist_ok=True)
    else:
        cache_biases = None
    
    if append: 
        shifts = []
        try:
            with open(outfile, 'r', newline='') as outhandle:
                existing_data = pd.read_csv(outhandle)
                processed_files = set(existing_data['Filename'])
        except FileNotFoundError: 
            append = False
            pass
    if not append:
        shifts = [['Bright Pixel (ADU)', 'Bias Shift (ADU)', 'Shift row (pixels)', 'Shift speed (pixels)', \
                'RAFTNAME', 'LSST_NUM', 'RUNNUM', 'EXTNAME', 'IMAGETAG', 'Filename']]
        with open(outfile, 'w', newline='') as outhandle:
            out = csv.writer(outhandle)
            out.writerows(shifts)
        with open(outfile, 'w', newline='') as outhandle:
            out = csv.writer(outhandle)
            out.writerows(shifts)

            

    shift_counter = 0
    
    oscan_cache = {}
            
    for filenum, filename in enumerate(filenames):
        if filename in processed_files:
            logger.debug(f'Skipped file {filename} - found recorded results')
            continue
        #filename = os.path.join(root, file).decode('UTF-8');
        #sys.stdout.write("\033[K")
        logger.debug(f'Checked {filenum}/{len(filenames)} files. Found {shift_counter} bias shifts.')
        if not filename.endswith('.fits'): 
            logger.debug(f'Skipping non-FITS file {filename}...')
            continue
        else:
            #print('Checking file: ' + filename)
            logger.info(f'Analyzing file {filename}...')
            
            oscan_cache = {}
            
            header = fits.getheader(filename)
            if not 'RAFTNAME' in header:
                header['RAFTNAME'] = 'single_sensor'
                
            sensor_plotsdir = None
            if plotsdir:
                sensor_plotsdir = f'{plotsdir}/{header["RAFTNAME"]}/{header["LSST_NUM"]}'
                os.makedirs(sensor_plotsdir, exist_ok=True)
            amp_geom = makeAmplifierGeometry(filename)
            for i in range(1,16+1):
                new_shifts = []
                
                try: 
                    segheader = fits.getheader(filename, ext=i)
                    bias, image, n_overscans = get_image_data( filename, i, amp_geom)
                    new_shifts = compute_bias_shift(bias, image, n_overscans, \
                             sensor_plotsdir, header['RAFTNAME'], header['LSST_NUM'],  \
                             header['RUNNUM'], segheader['EXTNAME'], header['IMAGETAG'],filename)
                except Exception as error:
                    logger.error(error)
                    continue
                shifts = shifts + new_shifts
                shift_counter = shift_counter + len(new_shifts)
                
                
                if len(new_shifts) > 0:
                    logger.info(f'Found {len(new_shifts)} shifts in {segheader["EXTNAME"]}. Total is  {shift_counter}.')
                    with open(outfile, 'a', newline='') as outhandle:
                        out = csv.writer(outhandle)
                        out.writerows(new_shifts)
                elif report_negatives:
                    logger.debug(f'Found {len(new_shifts)} shifts in {segheader["EXTNAME"]}. Total is  {shift_counter}.')
                    with open(outfile, 'a', newline='') as outhandle:
                        out = csv.writer(outhandle)
                        negative_row = [['NA', 'NA', 'NA', 'NA', \
                header['RAFTNAME'], header['LSST_NUM'], header['RUNNUM'], segheader['EXTNAME'], header['IMAGETAG'], filename]]
                        out.writerows(negative_row)
                if cache_biases:
                    if header['IMAGETAG'] not in oscan_cache:
                        oscan_cache[header['IMAGETAG']] = { (header['LSST_NUM'], segheader['EXTNAME']) : bias } 
                    else: 
                        oscan_cache[header['IMAGETAG']][(header['LSST_NUM'], segheader['EXTNAME'])] = bias 
            if cache_biases:  
                
                
                dirs = filename.split('/') 
                cache_filename = dirs[-1][:-5]
                cache_file_dirs = f"{cache_biases}/{dirs[-3]}/{dirs[-2]}"
                cache_file_loc = f'{cache_file_dirs}/{cache_filename}.p'
                
                os.makedirs(cache_file_dirs, exist_ok=True)
                df = pd.DataFrame(oscan_cache)
                df.to_pickle(cache_file_loc)
    
    return shifts


    
def main():
    
    #logging.basicConfig(filename=f'{time.time()}.log', encoding='utf-8', level=logging.DEBUG)
    
    parser = argparse.ArgumentParser(description='Find bias shifts in given EOTest images.')
    parser.add_argument('indir', type=str, nargs=1, help='Directory with input files')
    parser.add_argument('out', type=str, nargs=1, help='Output file for table of shifts')
    parser.add_argument('--plots', type=str, nargs='?',const='plots', help='Save plots of bias shift fits.')
    parser.add_argument('--append', type=bool, nargs='?',const=False, help='Append to existing output file. Default: False (overwrite)')
    parser.add_argument('--negatives', type=bool, nargs='?',const=True, help='Report negative results')

    args = parser.parse_args()
    indir = args.indir[0] 
    outfile = args.out[0] 
    makeplots = hasattr(args,'plots')
    append = args.append
    report_negatives = args.negatives

    if makeplots:
        plotsdir = args.plots
        os.makedirs(plotsdir, exist_ok=True)
    else:
        plotsdir = None
    flats_directory = os.fsencode(indir)
    shifts = [['Bright Pixel (ADU)', 'Bias Shift (ADU)', 'Shift row', \
            'RAFTNAME', 'LSST_NUM', 'RUNNUM', 'EXTNAME', 'IMAGETAG', 'Filename']]
    with open(outfile, 'w', newline='') as outhandle:
        out = csv.writer(outhandle)
        out.writerows(shifts)

    filenames = []
    for root, dir, f in os.walk(flats_directory):
        for file in f:
            filename = os.path.join(root, file).decode('UTF-8');
            if filename.endswith('.fits'): 
                filenames.append(filename)
    find_shifts_in_files(filenames, outfile, plotsdir=None, append=append, report_negatives=report_negatives)
    return

if __name__ == "__main__": main()
