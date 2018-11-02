
import numpy as np
import core
import pylab as plt
import nibabel as nib
import os
import skimage
from skimage import filters
import scipy.ndimage as ndi
from scipy.ndimage.morphology import binary_erosion as be
from scipy.ndimage.morphology import binary_fill_holes as bfh
from scipy.ndimage.morphology import binary_dilation as bd
from scipy.ndimage.morphology import binary_opening as bo
from scipy.ndimage.morphology import binary_closing as bc
from scipy.ndimage.morphology import grey_dilation as gd
import time
from skimage.segmentation import find_boundaries as boundaries
import skimage
import nipype.interfaces.fsl as fsl

def extract(thresholded, samp):
    img = nib.Nifti1Image(thresholded, np.eye(4))
    nib.save(img, 'thr-{}.nii.gz'.format(samp))
    img = nib.load('thr-{}.nii.gz'.format(samp))
    mybet = fsl.BET()
    result = mybet.run(in_file='thr-{}.nii.gz'.format(samp), out_file='BET-{}.nii.gz'.format(samp), frac=0.1)
    img = nib.load('BET-{}.nii.gz'.format(samp))
    data = img.get_data()
    return data

def regionseparation(array):
        # remove skin 
        skin_mask = array

        def removeskin(array):
            filtered = skimage.filters.gaussian(array, sigma=1)
            thresholded1 = np.multiply(filtered, (filtered>-200).astype(int))
            thresholded2 = np.multiply(thresholded1, (abs(thresholded1)>0.1).astype(int))
            return thresholded2

        skin_mask = array0

        for i in range(7):
            skin_mask = removeskin(skin_mask)

        skin_mask  = np.multiply(skin_mask, (skin_mask>-200).astype(int))
        array1     = np.multiply(array0,    (skin_mask>0.0).astype(int))
        array1     = np.multiply(array1,    (array1>-200).astype(int))

        # 2. Remove Skull
        skull_mask = array1>84.0
        array2     = np.multiply(array1,    (array1<84.0).astype(int))

        # 3. Remove Orbital Region

        for i in range(5):
            orbital_mask = bd(skull_mask)

        array3     = np.multiply(array2,    (orbital_mask==False).astype(int))

        # 4. Eliminate Cerebral Ventricle 
        array4     = np.multiply(array3, (array3>0.0).astype(int))

        # 5. Cannot Create Dmap
        array5     = extract(array4, samp)
        brain_mask = np.multiply(bc(bfh(bd(array5))).astype(int), array5)

        # plt.imshow(brain_mask[:,:,100], cmap='gray')
        # plt.show()
        # plt.ion()

        # 6. Get Ventricles
        ventr_mask1 = np.multiply((brain_mask>0).astype(int), (brain_mask<20).astype(int))
        ventr_mask2 = bd(ventr_mask1)
        ventr_mask3 = be(ventr_mask2)
        ventr_mask4 = bfh(ventr_mask3)

        labels = skimage.measure.label(ventr_mask4, connectivity=3)
        props = skimage.measure.regionprops(labels)
        v = [p.area for p in props]
        ind = v.index(max(v))
        vcoords = props[ind].coords
        v_mask = np.zeros((ventr_mask2.shape))
        for i in range(len(vcoords)):
            a,b,c = vcoords[i]
            v_mask[a,b,c]=1

        haemotoma_mask = (array5>50.0).astype(float)*1000

        # img = nib.Nifti1Image(h_mask, np.eye(4))
        # nib.save(img, '/Users/Sinead/DC-project/h-mask-50-{}.nii.gz'.format(samp))
        array6 = np.multiply(brain_mask, h_mask)
        array7 = be(array6)
        array8 = bc((array7))
        array9 = bfh((array8))
        array10 = bd(bd((array8)))
        array11 = bfh(bd(array10))

        # pick out connected region for haemotoma
        import skimage
        labels = skimage.measure.label(array10, connectivity=1)

        props = skimage.measure.regionprops(labels)
        h = props[0].coords
        h_mask = np.zeros((array10.shape))
        for i in range(len(h)):
            a,b,c = h[i]
            h_mask[a,b,c]=1

        return brain_mask, v_mask, h_mask



