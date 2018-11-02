#!/usr/bin/env python
# Copyright (c) Sinead Cook 2017. All rights reserved.
# This program or module is free software: you can redistribute it and/or
# modify it.

import scandata
import core
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tkinter import Tk
from tkFileDialog import askopenfilename
import numpy as np
import SimpleITK as sitk
class master:
    def run(self):
        # ask for scan file
        # we don't want a full GUI, so keep the root window from appearing
        Tk().withdraw() 
        self.path = askopenfilename()
        self.scan = scandata.Scan()
        self.analysis()

    def analysis(self):
        self.midplaneFinder()   # find the midplane
        self.midplaneLabel()     # save the midplane as a mask

    def midplaneFinder(self):
        import findeyes 

        print '\n','Reading in and thresholding scan...','\n'
        
        # reading in the scan into a np array
        if self.path.endswith('.nii') or self.path.endswith('.nii.gz'):
            self.scan.array, self.scan.pixelspacing, self.scan.affine = core.nifti2np(self.path)
            self.scan.bone        = core.thresholdnp(self.scan.array, 1100, 1500)
            self.scan.softtissue  = core.thresholdnp(self.scan.array, 0, 80)
        elif self.path.endswith('.dcm'):
            self.scan.array, self.scan.pixelspacing, self.scan.origin = core.dicom2np(self.path)
            self.scan.bone       = core.thresholdnp(self.scan.array, 1100, 1500)
            self.scan.softtissue = core.thresholdnp(self.scan.array, 0, 80)
        elif self.path.endswith('.nrrd'):
            self.scan.array, self.scan.pixelspacing, self.scan.affine= core.nrrd2np(self.path)
            self.scan.bone       = core.thresholdnp(self.scan.array, 1100, 1500)
            self.scan.softtissue = core.thresholdnp(self.scan.array, 0, 80)
                    
        print '\n',"Finished reading in scan. Starting reshaping...",'\n'
        
        # reshape the numpy arrays so that they are isotropic (for circle detection)
        bone       = core.reshape(self.scan.bone, self.scan.pixelspacing)
        softtissue = core.reshape(self.scan.softtissue, self.scan.pixelspacing)

        print '\n','Reshaping Complete. Starting midplane finder, finding eyes...','\n'

        # find the eyes.
        # histogram data of the number of circles detected in the scans in a given geometric space (the scan is split into 100 cubes). The histogram data covers circles in 3 directions (the frequencies are summed together).
        H, edges, histData2   = findeyes.hist3dAll(softtissue)
        # firstEyeRange, secondEyeRange contains the geometric space of the eyes (in the coordinate system we are currently using)
        firstEyeRange, secondEyeRange, certainty  = findeyes.ranges(H,edges)
        # we can now find the coordinates of the center of the eyes, using the histogram data
        self.c1Reshaped,self.c2Reshaped = findeyes.coords(histData2, firstEyeRange, secondEyeRange)
        # these coordinates (c1Reshaped etc.) are in the reshaped, isotropic coordinate system. The coordinates of the eyes in the original coordinate system of the CT scan are in c1 and c2.
        self.c1 = np.divide(self.c1Reshaped, np.array(self.scan.pixelspacing))
        self.c2 = np.divide(self.c2Reshaped, np.array(self.scan.pixelspacing))
        
        # now rotate the scans so that ellipses can be detected
        # first find the angles of the scan from the eyes
        angle1, angle2 = findeyes.anglesFromEyes(self.c1Reshaped,self.c2Reshaped)
        # next find a new skull array (rotatedBone) that has been rotated so it is facing up (and the centre of the eyes lie in the same plane)
        _, rotatedBone = findeyes.correctSkews(angle1, angle2, bone)
        _, rotatedSofttissue = findeyes.correctSkews(angle1, angle2, softtissue)
        # angles, and coords (isotropic coord system) of the centre of the ellipses that are fitted to the skull. 
        
        print '\n','Eyes found. Find ellipses...','\n'

        angs, xcentroids, ycentroids = core.ellipses(rotatedBone)
        # 'unreshaping' the results, so that the midplane will fit the original array
        # the slices of interest and their corresponding angles 
        slices, sliceAngles = core.selectEllipsesRange(angs)
        # the coordinates of the centroids of the ellipses in the slices of interest
        headx = [xcentroids[i] for i in slices] 
        heady = [ycentroids[i] for i in slices] 
        
        print '\n','Ellipses found. Find eyes in rotated system...','\n'
           
        H, edges, histData2   = findeyes.hist3dAll(rotatedSofttissue)
        # firstEyeRange, secondEyeRange contains the geometric space of the eyes (in the coordinate system we are currently using)
        firstEyeRange, secondEyeRange, certainty  = findeyes.ranges(H,edges)
        # we can now find the coordinates of the center of the eyes, using the histogram data
        self.c1ReshapedRotated,self.c2ReshapedRotated = findeyes.coords(histData2, firstEyeRange, secondEyeRange)
        
        print '\n','Ellipses found. Find eyes in rotated system...','\n'        
        
        # find a midplane using a combination of the eyes (for location) and the ellipse angles
        a,b,c,d, normal = core.findPlaneFromEllipses(self.scan.bone, self.c1ReshapedRotated, self.c2ReshapedRotated, slices, headx, heady, sliceAngles)
        # put this plane back into the original coordinate system
        theta1 = angle1/360.*2*np.pi # angles in radians
        theta2 = angle2/360.*2*np.pi
        
        a,b,c,d = core.correctPlaneParams(angle1, angle2, normal, self.c1, self.c2)
        self.scan.params = a,b,c,d
        print '\n','Midplane found...','\n'
                
    def midplaneLabel(self):
        import nibabel as nib
        import os
        print '\n','Creating midplane mask...','\n'
        a,b,c,d = self.scan.params
        shape = (self.scan.array).shape
        mask = -1000*np.ones((shape))
        x = np.arange(0,shape[0],1)
        z = np.linspace(0,shape[2],shape[0])
        xx, yy = np.meshgrid(z, (d-c*z-a*x)/b)
        y = yy.astype(int)
        for k in range(shape[1]):
            for i in range(shape[2]):
                mask[y[k][i]-1:y[k][i]+1, k, i] = 1
        print '\n','Saving midplane mask...','\n'
        
        mask = np.swapaxes(mask, 0, 2)
        img = sitk.GetImageFromArray(mask)
        
        if self.path.endswith('.dcm'):
            dcmimg = sitk.ReadImage(self.path)
            img.SetDirection(dcmimg.GetDirection())
            img.SetOrigin(self.scan.origin)
            img.SetSpacing(self.scan.pixelspacing)
        else:
            img.CopyInformation(sitk.ReadImage(self.path))
        midplaneMaskPath = os.path.join(os.path.split(self.path)[0], 'midplane.mha')
        sitk.WriteImage(img, midplaneMaskPath)
        print '\n','Midplane mask saved at ', midplaneMaskPath,'\n'

    # def regionextraction(self):

    #     print '\n','Midplane mask saved at ', midplaneMaskPath,'\n'
    #     self.scan.parenchyma, self.scan.ventricles, self.scan.haematoma = symmetryanalysis.regionseparation(self.scan.array)

        # img = nib.Nifti1Image((self.scan.parenchyma).astype(float)*1000, np.eye(4))
        # nib.save(img, '/Users/Sinead/DC-project/v-mask-{}.nii.gz'.format(samp))

        # img = nib.Nifti1Image(array5, np.eye(4))
        # nib.save(img, '/Users/Sinead/DC-project/pre-adapted-kondo-paper-{}.nii.gz'.format(samp))

        # img = nib.Nifti1Image((mask).astype(float)*1000, np.eye(4))
        # nib.save(img, '/Users/Sinead/DC-project/h-mask-{}.nii.gz'.format(samp))
            
        # img = nib.load(reshaped_niftis[4])
        # shape = img.get_data().shape

        # midplane_split = np.zeros(shape)

        # for k in range(shape[2]):
        #     x = np.arange(0,shape[0],1)
        #     z = np.linspace(0,shape[2],shape[0])
        #     xx, yy = np.meshgrid(z, (d-c*z-a*x)/b)
        #     y = yy.astype(int)
        #     for j in range(len(y)):
        #         midplane_split[y[k,j]:shape[0],:, k]=1


        # v_mask = (ventr_mask4).astype(int)

        # get_ipython().magic(u'matplotlib nbagg')
        # left_h = np.multiply(mask, midplane_split==0)
        # right_h = np.multiply(mask, midplane_split==1)

        # left_v = np.multiply(v_mask, midplane_split==0)
        # right_v = np.multiply(v_mask, midplane_split==1)

        # brain = np.multiply(brain_mask, (mask==0).astype(int)) # no haematoma

        # left_brain = np.multiply(brain, midplane_split==0)
        # right_brain = np.multiply(brain, midplane_split==1)

        # plt.imshow((left_v)[:,:,80], cmap='gray')
        # plt.show()
        # plt.ion()

        # left_h_vol = (left_h[left_h!=0]).size*0.001 
        # right_h_vol = (right_h[right_h!=0]).size*0.001 
        # left_brain_vol = (left_brain[left_brain!=0]).size*0.001 
        # right_brain_vol = (right_brain[right_brain!=0]).size*0.001 
        # left_ventr_vol = (left_v[left_v!=0]).size*0.001 
        # right_ventr_vol = (right_v[right_v!=0]).size*0.001 


        # mask_lh = (left_h!=0).astype(float)*1000
        # mask_rh = (right_h!=0).astype(float)*1000
        # mask_lv = (left_v!=0).astype(float)*1000
        # mask_rv = (right_v!=0).astype(float)*1000
        # mask_lb = (left_brain!=0).astype(float)*1000
        # mask_rb = (right_brain!=0).astype(float)*1000

        # print 'Volume of haematoma on LHS is {}'.format(left_h_vol)
        # print 'Volume of haematoma on RHS is {}'.format(right_h_vol)
        # print 'Volume of brain on LHS is {}'.format(left_brain_vol)
        # print 'Volume of brain on RHS is {}'.format(right_brain_vol)
        # print 'Volume of CSF in ventricles on LHS is {}'.format(left_ventr_vol)
        # print 'Volume of CSF in ventricles on RHS is {}'.format(right_ventr_vol)

        # areaLHS = np.zeros((mask_lh.shape[1], mask_lh.shape[2]))
        # areaLHS[1,:].shape

        # def area_difference(maskLHS, maskRHS):
        #     diff = np.zeros((maskLHS.shape[2], maskRHS.shape[1]))
        #     for k in range(maskLHS.shape[2]):
        #         for j in range(mask.shape[1]): 
        #             # x y z defined according to miplane vertical (not horizontal as is the default in imshow)
        #             areaLHS=np.sum(maskLHS[:,j,k])
        #             areaRHS=np.sum(maskRHS[:,j,k])
        #             diff[k,j] = (areaRHS-areaLHS)*0.0001
        #     return diff

        # h_diff = area_difference(mask_lh, mask_rh)
        # brain_diff = area_difference(mask_lb, mask_rb)

        # brain_diff.shape

        # get_ipython().magic(u'matplotlib nbagg')
        # fig, ax = plt.subplots()
        # ax.set_aspect(1)
        # brain_diff[brain_diff==0]=np.nan
        # palette = plt.cm.winter
        # palette.set_bad(alpha=0.2)
        # heatmap = ax.imshow(brain_diff, cmap=palette)
        # # heatmap = ax.pcolor(brain_diff==0, cmap=plt.cm.winter)
        # plt.xlim(0, 249)
        # plt.ylim (0,176)
        # plt.colorbar(heatmap)
        # plt.show()
        # plt.ion()
                       

if __name__ == "__main__":
    master.run(master())

