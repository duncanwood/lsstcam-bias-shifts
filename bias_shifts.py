import os, csv



import lsst.geom
import lsst.afw.cameraGeom

#import lsst.eotest.image_utils as imutils


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd

import scipy.signal
import scipy.stats

import numpy as np, numpy

def readout_order(image, amp, verbose=False):
        """Extract the image data from an amp, flipped to match readout order;
           i.e. such that the pixel at (0,0) is the chronological start of readout,
           and (n-1, m-1) is the chronological end.
        Parameters
        ----------
        image : `lsst.afw.image.Image` or `lsst.afw.image.MaskedImage`
            Image containing the amplifier of interest.
        amp : `lsst.afw.cameraGeom.Amplifier`
            Amplifier on image to extract.
        Returns
        -------
        output : `lsst.afw.image.Image`
            Image of the amplifier in the desired configuration.
        """
        flipDic = { 
                 lsst.afw.cameraGeom.ReadoutCorner.LL : (False, False),
                 lsst.afw.cameraGeom.ReadoutCorner.LR : (True, False),
                 lsst.afw.cameraGeom.ReadoutCorner.UL : (False, True),
                 lsst.afw.cameraGeom.ReadoutCorner.UR : (True, True)
               }
        if verbose:
            print(f"Readout corner: {amp.getReadoutCorner()} - Flip? (X,Y): " + str(flipDic[amp.getReadoutCorner()]))
        
        output = image
        return lsst.afw.math.flipImage(output, * flipDic[amp.getReadoutCorner()])
    
def readout_order_arr(arr, amp, verbose=False):
        """Extract the image data from an amp, flipped to match readout order;
           i.e. such that the pixel at (0,0) is the chronological start of readout,
           and (n-1, m-1) is the chronological end.
        Parameters
        ----------
        image : `lsst.afw.image.Image` or `lsst.afw.image.MaskedImage`
            Image containing the amplifier of interest.
        amp : `lsst.afw.cameraGeom.Amplifier`
            Amplifier on image to extract.
        Returns
        -------
        output : `lsst.afw.image.Image`
            Image of the amplifier in the desired configuration.
        """
        flipDic = { 
                 lsst.afw.cameraGeom.ReadoutCorner.LL : (False, False),
                 lsst.afw.cameraGeom.ReadoutCorner.LR : (True, False),
                 lsst.afw.cameraGeom.ReadoutCorner.UL : (False, True),
                 lsst.afw.cameraGeom.ReadoutCorner.UR : (True, True)
               }
        if verbose:
            print(f"Readout corner: {amp.getReadoutCorner()} - Flip? (X,Y): " + str(flipDic[amp.getReadoutCorner()]))
        
        output = arr
        flipCol, flipRow = flipDic[amp.getReadoutCorner()]
        if flipCol: output = output[:,::-1]
        if flipRow: output = output[::-1]
        return output
    
def get_oscan_rows(exp, amp, nskip=3, averageMethod="mean"):
    averageMethods = ["mean"]
    if averageMethod not in averageMethods:
        raise Exception(f'Averaging method "{averageMethod}" not recognized')
    oscan_arr = readout_order_arr(exp[amp.getRawSerialOverscanBBox()].image.array, amp)
    
    if averageMethod=="mean":
        return np.mean(oscan_arr[:,nskip:], axis=1)    
    
def get_image_rows(exp, amp, nskip=3, averageMethod="mean"):
    averageMethods = ["mean"]
    if averageMethod not in averageMethods:
        raise Exception(f'Averaging method "{averageMethod}" not recognized')
    im_arr = readout_order_arr(exp[amp.getRawDataBBox()].image.array, amp)
    
    if averageMethod=="mean":
        parallel_correction = np.mean(readout_order_arr(exp.image[amp.getRawParallelOverscanBBox()].array,amp), axis=0)
        return np.mean(im_arr-parallel_correction, axis=1)
   
   
def get_full_parallel_correction(exp, amp, parallel_skip=2):
        
    image = exp.image
    
    oscan_bbox = amp.getRawParallelOverscanBBox()
    full_bbox = amp.getRawBBox()
    
    oscan_min_y = max(oscan_bbox.getMinY(), full_bbox.getMinY())
    oscan_max_y = min(oscan_bbox.getMaxY(), full_bbox.getMaxY())
    
    parallel_oscan_bbox = lsst.geom.Box2I(minimum=lsst.geom.Point2I(full_bbox.getMinX(),oscan_min_y), maximum=lsst.geom.Point2I(full_bbox.getMaxX(),oscan_max_y))
    parallel_correction_arr = np.mean(readout_order_arr(image[parallel_oscan_bbox].array[parallel_skip:],amp), axis=0)
    return parallel_correction_arr


def smooth_rows(oscan_rows, window=30, order=1, n_smooth_rows=150):
    return smooth_rows_butter(oscan_rows, window, order, n_smooth_rows)

def smooth_rows_butter(oscan_rows, window=30, order=1, n_smooth_rows=150):
    b, a = scipy.signal.butter(1, 1/n_smooth_rows,btype='low')
    return scipy.signal.filtfilt(b,a,oscan_rows,padlen=0)

def smooth_rows_savgol(oscan_rows, window=30, order=1):
    return scipy.signal.savgol_filter(oscan_rows, window, order)

def smooth_rows_deriv(smoothed_rows, window=30, order=1):
    return scipy.signal.savgol_filter(smoothed_rows, window,order,deriv=1,delta=1,mode='mirror')


def find_mask_runs(mask):
    return numpy.flatnonzero(np.diff(numpy.r_[numpy.int8(0), mask.view(numpy.int8), numpy.int8(0)])).reshape(-1, 2)

def shift_in_rows(oscan_rows, window=30, order=1, threshold=1, n_smooth_rows=150, skip_rows=30, show_plot=False):
    smoothed_deriv = smooth_rows_deriv(smooth_rows(oscan_rows, window, order, n_smooth_rows), window, order)
    norm_factor =  (n_smooth_rows*window)**0.5/np.std(butter_highpass_filter(oscan_rows, 1, window,order=order))
    motion =np.abs(smoothed_deriv)*norm_factor
    shift_mask = motion > threshold
    shift_mask[:skip_rows] = False
    
    if show_plot:
        plt.plot(motion)
        plt.axhline(threshold)
        plt.show()
        plt.close()
    
    shift_regions = find_mask_runs(shift_mask)
    shift_peaks = []
    for region in shift_regions:
        shift_peaks.append(region[0]+np.argmax(motion[region[0]:region[1]]))
    if len(shift_regions)==0: return np.array([np.NaN]*4)
    return np.concatenate((np.array([shift_peaks]),shift_regions.T)).T

def flat_kernel(window=30):
    kernel = np.concatenate([np.ones(window), -1*np.ones(window-1)])
    return kernel/window

def odd_local_kernel(window=30):
    kernel = np.concatenate([np.arange(window), np.arange(-window+1,0)])
    kernel = kernel/np.sum(kernel[:window])
    return kernel

def shift_kernel(window=30):
    return odd_local_kernel(window)

def scan_for_shifts(data, window=30, noise_filter=30, threshold=3, skip_rows=30):
    """
    Looks for shifts in baseline in noisy data using a convolution.
    
    Returns
    -------
    An array of detected shifts with rows [peak of shift, start of shift area, end of shift area]
    
    """
    local_noise = np.std(butter_highpass_filter(data, 1/noise_filter,1))
    shift_conv = np.convolve(data, shift_kernel(window),mode='valid')
    shift_conv = np.concatenate([np.zeros(window-1), shift_conv, np.zeros(window)])
    shift_like = np.abs(shift_conv)/local_noise
    shift_mask = shift_like > threshold
    shift_mask[:skip_rows]=False
    shift_regions = find_mask_runs(shift_mask)
    shift_peaks = []
    for region in shift_regions:
        region_peak = np.argmax(shift_like[region[0]:region[1]]) + region[0]
        if satisfies_flatness(region_peak, shift_conv[region_peak], data):
            shift_peaks.append([shift_conv[region_peak],region_peak, region[0], region[1]])
    if len(shift_peaks)==0: return np.array([[np.NaN]*4]), local_noise
    return np.asarray(shift_peaks),  local_noise

def satisfies_flatness(shiftrow, shiftmag, oscans, window=30, verbose=False):
    
    prerange = np.arange(shiftrow-window, shiftrow)
    postrange = np.arange(shiftrow, shiftrow+window)
    
    preslope, preintercept, pre_r_value, p_value, std_err = scipy.stats.linregress(prerange,oscans[prerange])
    postslope, postintercept, post_r_value, p_value, std_err = scipy.stats.linregress(postrange,oscans[postrange])
    
    if verbose:
        print(f'Pre-threshold: {preslope*2*len(prerange)}')
        print(f'Post-threshold: {postslope*2*len(postrange)}')
        print(f'Shift value: {shiftmag}')
    
    pretrending  = (preslope*2*len(prerange) < shiftmag)   if (shiftmag>0) else (preslope*2*len(prerange) > shiftmag)
    posttrending = (postslope*2*len(postrange) < shiftmag) if (shiftmag>0) else (postslope*2*len(postrange) > shiftmag)
    
    return (pretrending and posttrending)

def find_shifts_in_exposure(exp, amp=None, method="conv", **options):
    methods = ["derivative","conv"]
    if method not in methods:
        raise Exception(f'Shift detection method "{method}" not understood. Available methods are {methods}')        
    if amp is None:
        det = exp.getDetector()
    else: 
        det = [amp.getName()]
    
    shifts = {}

    for thisAmp in det:
        oscan_rows = get_oscan_rows(exp, thisAmp)
#        im_rows = 
        if method=="derivative":
            shifts[thisAmp.getName()] = shift_in_rows(oscan_rows, **options)
        if method=="conv":
            shifts[thisAmp.getName()] = scan_for_shifts(oscan_rows, **options), np.mean(oscan_rows), np.mean(exp[thisAmp.getRawDataBBox()].image.array)
    
    if amp is None:
        return shifts
    else: 
        return shifts[amp.getName()]
    
    
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def report_shifts_in_det(exp, basedir='.', outdirname='shifts_results', append=True, repo_path = "/sdf/group/rubin/repo/main", report_negatives=True ):
    det = exp.getDetector()
    raft, sensor = det.getName().split('_')
    md = exp.getMetadata()
    runnum = md['RUNNUM']
    obsid  = md['OBSID']
    
    outdir = f'{basedir}/{outdirname}/{runnum}/{raft}'
    outfilename = f'shifts-{runnum}-{raft}_{sensor}.csv'
    outfile = f'{outdir}/{outfilename}'
    
    os.makedirs(outdir, exist_ok=True)
    headers = [['Bias Shift (ADU)' ,'Shift center (row)', 'Shift start (row)','Shift end (row)',  \
                'HF row noise (ADU)', 'Overscan mean (ADU)', 'Imaging mean (ADU)', 'raft', 'sensor', 'amp', 'obs_id']]
    existing_data = None
    processed_obsids = {}
    shifts=[[]]
    if append: 
        try:
            with open(outfile, 'r', newline='') as outhandle:
                existing_data = pd.read_csv(outhandle)
                processed_obsids = set(existing_data['obs_id'])
                if obsid in processed_obsids and \
                        all([amp.getName() in set(existing_data[existing_data['obs_id'] == obsid]['amp']) for amp in det]):
                    print(f"Data for {obsid}-{det.getName()} already present in {outfile} - skipping")
                    return
        except FileNotFoundError as e: 
            append = False
            pass
        except Error as e:
            append = False
            raise(e)
        
    if not append:
        with open(outfile, 'w', newline='') as outhandle:
            out = csv.writer(outhandle)
            out.writerows(headers)
        # with open(outfile, 'w', newline='') as outhandle:
        #     out = csv.writer(outhandle)
        #     out.writerows(shifts)
    
    new_shifts_dic = find_shifts_in_exposure(exp)

    new_shifts = []
    for ampname, ((shifts, hfNoise), oscanMean, imageMean) in new_shifts_dic.items():
        if (existing_data is not None) and (ampname in set(existing_data[existing_data['obs_id'] == obsid]['amp'])): continue
        for shift in shifts:
            new_shifts.append((f'{shift[0]:.3f}', shift[1], shift[2], shift[3],  \
                    f'{hfNoise:.3f}',f'{oscanMean:.3f}',f'{imageMean:.3f}', raft, sensor, ampname, obsid))
    
    
    if len(new_shifts) > 0:
        #logger.info(f'Found {len(new_shifts)} shifts in {segheader["EXTNAME"]}. Total is  {shift_counter}.')
        with open(outfile, 'a', newline='') as outhandle:
            out = csv.writer(outhandle)
            out.writerows(new_shifts)
    elif report_negatives:
        #logger.debug(f'Found {len(new_shifts)} shifts in {segheader["EXTNAME"]}. Total is  {shift_counter}.')
        with open(outfile, 'a', newline='') as outhandle:
            out = csv.writer(outhandle)
            negative_row = [['NA', 'NA', 'NA', 'NA', \
    header['RAFTNAME'], header['LSST_NUM'], header['RUNNUM'], segheader['EXTNAME'], header['IMAGETAG'], filename]]
            out.writerows(negative_row)

def report_shifts_in_exps(butler, obsIds, detname):
    repo_path = "/sdf/group/rubin/repo/main"
    
    for obsid in obsIds:
        wherestr = f"exposure.obs_id='{obsid}' and detector.full_name='{detname}'";
        try:
            dataref = list(butler.registry.queryDatasets(datasetType='raw', collections=['LSSTCam/raw/all'], where=wherestr))[0]
        except IndexError as e:
            print(f'No exposure found for {detname} in {obsid}. Skipping...')
            continue
        if has_been_analyzed(butler, obsid, detname): continue
        exp = butler.getDirect(dataref)
        report_shifts_in_det(exp, append=True)
        del exp

def has_been_analyzed(butler, obsid, detname, basedir='.', outdirname='shifts_results'):
    
    raft, sensor = detname.split('_')
    
    expinfo = list(butler.registry.queryDimensionRecords('exposure', datasets='raw', collections=['LSSTCam/raw/all'], \
                             where=f"exposure.obs_id='{obsid}' and detector.full_name='{detname}'"))[0]

    runnum = expinfo.science_program

    outdir = f'{basedir}/{outdirname}/{runnum}/{raft}'
    outfilename = f'shifts-{runnum}-{raft}_{sensor}.csv'
    outfile = f'{outdir}/{outfilename}'
    
    try:
        with open(outfile, 'r', newline='') as outhandle:
            existing_data = pd.read_csv(outhandle)
            processed_obsids = set(existing_data['obs_id'])
            if obsid in processed_obsids and \
                    len(set(existing_data[existing_data['obs_id'] == obsid]['amp']))==16:
                print(f"Data for {obsid}-{detname} already present in {outfile} - skipping")
                return True
    except FileNotFoundError as e: 
        pass
    except Error as e:
        raise(e)
    
        
def df_from_shift_files(basedir='.'):
    shift_files = []
    for path, dirs, files in os.walk(basedir):
        if path.split('/')[-1][0] == '.': continue
        for file in files:
            if file.endswith('.csv'):
                print(path + '/' + file)
                shift_files.append(path + '/' + file)
    return pd.concat((pd.read_csv(f) for f in shift_files))

def make_bias_time_seq_plot(exp, amp, shift_row, image=None, row_range=10, parallel_skip=2, savefig=True):
    
    if image is None: image = exp.image
    det = exp.getDetector()
    detname, ampname = det.getName(), amp.getName()
    metadict=exp.getMetadata()
    run, obsid = metadict['RUNNUM'], metadict['OBSID']
    
    rowstart = shift_row - row_range // 2
    rowend = rowstart + row_range
    
    oscan_bbox = amp.getRawParallelOverscanBBox()
    full_bbox = amp.getRawBBox()
    
    oscan_min_y = max(oscan_bbox.getMinY(), full_bbox.getMinY())
    oscan_max_y = min(oscan_bbox.getMaxY(), full_bbox.getMaxY())
    
    amp_im=image
    
    parallel_oscan_bbox = lsst.geom.Box2I(minimum=lsst.geom.Point2I(full_bbox.getMinX(),oscan_min_y), maximum=lsst.geom.Point2I(full_bbox.getMaxX(),oscan_max_y))
    #parallel_correction = imutils.bias_image_col(image, parallel_oscan_bbox)
    #parallel_correction_arr = readout_order(parallel_correction, amp).array
    parallel_correction_arr = np.mean(readout_order_arr(image[parallel_oscan_bbox].array[parallel_skip:],amp), axis=0)
    print("Parallel oscans: " + str(parallel_oscan_bbox)) 
    
    
    amp_im_arr = readout_order(image, amp).array
    
    parallel_corrected_im = amp_im_arr[rowstart:rowend] - parallel_correction_arr#[rowstart:rowend]
    plt.figure(figsize=(12,8))
    plt.xlim(0, len(parallel_corrected_im.flat))
    plt.plot(parallel_corrected_im.flat, label="Raw pixel values", linestyle="None",marker=".",alpha=.5)
    #plt.ylim(np.percentile(parallel_corrected_im.flat,(1,99)))
    smoothed_im = scipy.signal.savgol_filter(parallel_corrected_im.flat, 31,1)
    newlines = np.arange((parallel_corrected_im.shape[0])+1)*parallel_corrected_im.shape[1]
    rowmeans = np.mean(parallel_corrected_im, axis=-1)
    
    oscanmeans = np.mean(parallel_corrected_im, axis=-1)
    plt.hlines(rowmeans, newlines[:-1], newlines[1:], label="Mean of entire row", \
            linestyle="solid", linewidth=2, color='red')
    plt.plot(smoothed_im, label="Smoothed pixel values", linewidth=2, alpha=0.8, color='green')
    newlinelabels = []
    for i, newline in enumerate(newlines):
        rowline = plt.axvline(x=newline,color='black', linestyle='dotted')
        newlinelabels.append(f'Row {i+rowstart}')
        
    plt.xticks(newlines, newlinelabels, rotation='vertical')
        #plt.text(newline,np.min(parallel_corrected_im),,rotation=90)
    rowline.set_label("New row")
    plt.title(f"Bias shift as a readout sequence after parallel overscan correction\n{detname}_{ampname}\nRun {run} - {obsid}")
    plt.legend()
    plt.ylabel("ADU")
    plt.xlabel("Readout order (pixels)")

    plt.tight_layout()
    if savefig: plt.savefig(f"../plots/custom_shifts/shift-readout-sequence-{run}-{detname}_{ampname}-{obsid}.pdf")
    plt.show()
    plt.close()
    
def make_bias_amp_plot(exp, amp, image=None, savefig=True, notes=None):
    

    if image is None: image = exp.image
    det = exp.getDetector()
    detname, ampname = det.getName(), amp.getName()
    metadict=exp.getMetadata()
    run, obsid = metadict['RUNNUM'], metadict['OBSID']
    
    fig = plt.figure( figsize=(8,20))
    
    amp_im_arr = readout_order(image, amp).array
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.9))
    im = ax.imshow(amp_im_arr, clim=np.percentile(amp_im_arr.flat, (1,99)), cmap='gist_heat')
    plt.title(f"Bias shift in {detname}_{ampname}\nRun {run} - {obsid}")

    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="10%", pad=0.2)
    fig.add_axes(cax)
    fig.colorbar(im, cax=cax, orientation="vertical")
    fig.set_facecolor('white')

    if notes is not None: fig.text(.5, .05, notes, ha='center')

    #plt.tight_layout()

    outdir = 'plots/custom_shifts'
    outfile=f"{outdir}/shift-image-{run}-{detname}_{ampname}-{obsid}.pdf"
    os.makedirs(outdir, exist_ok=True)
    if savefig: plt.savefig(outfile)
    plt.show()

    plt.close()
    
def make_serial_profile_plot(exp, amp, save_fig=False, show_fig=True):
    obsid = exp.getMetadata()['OBSID']
    run = exp.getMetadata()['RUNNUM']
    detname = exp.getDetector().getName()
    ampname = amp.getName()
    fig = plt.figure()
    plt.plot(np.mean(exp[amp.getRawSerialOverscanBBox()].image.array[2:],axis=1))
    plt.xlabel("Row")
    plt.ylabel("Overscan mean (ADU)")
    plt.title(f'Serial profile: Run {run} - {obsid} - {detname}_{ampname}')
    if save_fig: 
        outdir = f'plots/{run}/'
        outfile = f'serial_profile-{run}-{obsid}-{detname}_{ampname}.pdf'
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(outdir+outfile)
        
    if show_fig: plt.show()