import numpy as np
import script
import pylab as plt
import nibabel as nib
import os
from skimage import filters
import scipy.ndimage as ndi
from scipy.ndimage.morphology import binary_erosion as be
from scipy.ndimage.morphology import binary_fill_holes as bfh
from scipy.ndimage.morphology import binary_dilation as bd
from scipy.ndimage.morphology import binary_opening as bo
from scipy.ndimage.morphology import binary_closing as bc
from scipy.ndimage.morphology import grey_dilation as gd
import nipype.interfaces.fsl as fsl

def threshold(array, low=0, high=60):
    mask1 = array<high
    mask2 = array>low
    mask3 = np.multiply(mask1,mask2) 
    mask4 = bfh(mask3)
    thresholded = np.multiply(array,mask4)
    return thresholded

def extract(thresholded, samp):
    img = nib.Nifti1Image(thresholded, np.eye(4))
    nib.save(img, 'thr-{}.nii.gz'.format(samp))
    img = nib.load('thr-{}.nii.gz'.format(samp))
    mybet = fsl.BET()
    result = mybet.run(in_file='thr-{}.nii.gz'.format(samp), out_file='BET-{}.nii.gz'.format(samp), frac=0.1)
    img = nib.load('BET-{}.nii.gz'.format(samp))
    data = img.get_data()
    return data

def spacer(data):
    grad = abs(np.gradient(data)[0])
    mask1 = grad<10
    mask2 = grad>0
    mask3 = np.multiply(mask1,mask2) 
    data3 = bfh(mask3)
    data4 = be(data3)
    data5 = bo(data3)
    data6 = bc(data5)
    result = np.multiply(data, data6)
    return result