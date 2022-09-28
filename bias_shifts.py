import lsst.geom

# TODO: remove eotest dependency
import lsst.eotest.image_utils as imutils
import lsst.afw.cameraGeom


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.signal

import numpy as np

def readout_order(image, amp):
        """Extract the image data from an amp, flipped to match readout order.
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
        flip_dic = { 
                 lsst.afw.cameraGeom.ReadoutCorner.LL : (True, False),
                 lsst.afw.cameraGeom.ReadoutCorner.LR : (False, False),
                 lsst.afw.cameraGeom.ReadoutCorner.UL : (True, True),
                 lsst.afw.cameraGeom.ReadoutCorner.UR : (False, True)
               }
        print(amp.getReadoutCorner())
        print("Flip?: " + str(flip_dic[amp.getReadoutCorner()]))
        
        output = image[amp.getRawBBox()]
        return lsst.afw.math.flipImage(output, *flip_dic[amp.getReadoutCorner()])
    

def make_bias_time_seq_plot(exp, amp, shift_row, image=None, row_range=10, savefig=True):
    
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
    
    #flippedY = oscan_bbox.getMinY() == full_bbox.getMinY()
    amp_im=image#[amp.getRawBBox()]
    
    
                      
    parallel_oscan_bbox = lsst.geom.Box2I(minimum=lsst.geom.Point2I(full_bbox.getMinX(),oscan_min_y), maximum=lsst.geom.Point2I(full_bbox.getMaxX(),oscan_max_y))
    parallel_correction = imutils.bias_image_col(image, parallel_oscan_bbox)
    parallel_correction_arr = readout_order(parallel_correction, amp).array
    print("Parallel oscans: " + str(parallel_oscan_bbox)) 
    
    
    amp_im_arr = readout_order(image, amp).array
    
    parallel_corrected_im = amp_im_arr[rowstart:rowend] - parallel_correction_arr[rowstart:rowend]
    plt.figure(figsize=(12,8))
    plt.xlim(0, len(parallel_corrected_im.flat))
    plt.plot(parallel_corrected_im.flat, label="Raw pixel values", linestyle="None",marker=".",alpha=.5)
    smoothed_im = scipy.signal.savgol_filter(parallel_corrected_im.flat, 31,1)
    newlines = np.arange((parallel_corrected_im.shape[0])+1)*parallel_corrected_im.shape[1]
    rowmeans = np.mean(parallel_corrected_im, axis=-1)
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
    #plt.ylim(np.percentile(exp.image.array.flat, (1,99)))
    
def make_bias_amp_plot(exp, amp, image=None, savefig=True):
    
    if image is None: image = exp.image
    det = exp.getDetector()
    detname, ampname = det.getName(), amp.getName()
    metadict=exp.getMetadata()
    run, obsid = metadict['RUNNUM'], metadict['OBSID']
    
    fig, ax = plt.subplots( figsize=(8,20))
    
    amp_im_arr = readout_order(image, amp).array
        
    im = ax.imshow(amp_im_arr, clim=np.percentile(amp_im_arr.flat, (1,99)), cmap='gist_heat')
    plt.title(f"Bias shift in {detname}_{ampname}\nRun {run} - {obsid}")

    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="10%", pad=0.2)
    fig.add_axes(cax)
    fig.colorbar(im, cax=cax, orientation="vertical")
    fig.set_facecolor('white')
    plt.tight_layout()
    if savefig: plt.savefig(f"../plots/custom_shifts/shift-image-{run}-{detname}_{ampname}-{obsid}.pdf")
    plt.show()

    plt.close()