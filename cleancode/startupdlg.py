#!/usr/bin/env python
# Copyright (c) 2007-8 Qtrac Ltd. All rights reserved.
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 2 of the License, or
# version 3 of the License, or (at your option) any later version. It is
# provided for educational purposes and is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public License for more details.

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import scandata
import ui_startupdlg
import core
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()


class StartupDlg(QDialog,
        ui_startupdlg.Ui_StartupDlg):

    def __init__(self, fopen=None, parent=None):
        super(StartupDlg, self).__init__(parent)
        self.setupUi(self)

        self.scan = scandata.Scan() 

        self.progressTextBox.setText("Check desired options and select a scan")

        if fopen is not None:
            self.fileOpen

        self.pushButtonSelectScan.clicked.connect(self.fileOpen)
        self.pushButtonGo.clicked.connect(self.analysis)

    def fileOpen(self):     
        self.path = str(QFileDialog.getOpenFileName(self, filter=scandata.Scan.formats()))
        if self.path is not str():
            self.pushButtonGo.setEnabled(True)
            self.progressTextBox.setText("Press Go to start analysis")

    def analysis(self):
        self.pushButtonGo.setEnabled(False)

        self.symmetry      = int(self.checkBoxSymmetryAnalysis.checkState())
        self.midplaneMask  = int(self.checkBoxSkullMidplaneMask.checkState())
        self.brainMask     = int(self.checkBoxBrainMask.checkState())
        self.haematomaMask = int(self.checkBoxHaematomaMask.checkState())
        self.ventricleMask = int(self.checkBoxVentricleMask.checkState())
        # self.pushButtonSelectScan.setText("Abort") # figure out how to interrupt the program

        if self.symmetry == 2 or self.midplaneMask == 2:
            self.midplaneFinder()

        if self.midplaneMask == 2:
            self.midplaneMask()


    def midplaneFinder(self):
        import findeyes 

        self.progressTextBox.setText("Reading in scan")

        if self.path.endswith('.nii') or self.path.endswith('.nii.gz'):
            array, self.scan.pixelspacing = core.nifti2np(self.path)
            bone       = core.thresholdnp(self.scan.array, 1100, 1500)
            softtissue = core.thresholdnp(self.scan.array, 0, 80)
            # visual     = core.thresholdnp(self.scan.array, -100, 1500)

        elif self.path.endswith('.dcm'):
            self.scan.array, self.scan.pixelspacing = core.dicom2np(self.path)
            bone       = core.thresholdnp(self.scan.array, 1100, 1500)
            softtissue = core.thresholdnp(self.scan.array, 0, 80)
            # visual     = core.thresholdnp(self.scan.array, -100, 1500)

        self.progressTextBox.setText("Finished reading in scan")

        bone       = core.reshape(bone,       self.scan.pixelspacing, bone.shape[0],       bone.shape[2])
        softtissue = core.reshape(softtissue, self.scan.pixelspacing, softtissue.shape[0], softtissue.shape[2])
        array      = core.reshape(array, self.scan.pixelspacing, softtissue.shape[0], softtissue.shape[2])

        self.progressTextBox.setText("Reshaping Complete")

        # plt.imshow(self.scan.softtissue[:,:,50])
        # plt.show()

        H, edges, data, hist_data_2c   = findeyes.hist3d_all(softtissue)
        ranges_1, ranges_2, certainty  = findeyes.ranges(H,edges)
        c1,c2 = findeyes.coords(hist_data_2c, ranges_1, ranges_2)

        angle1, angle2               = findeyes.angles_from_eyes(c1,c2)
        _, self.scan.bone            = findeyes.correct_skews(angle1,angle2,bone)
        _, self.scan.softtissue      = findeyes.correct_skews(angle1,angle2,softtissue)
        _, self.scan.array           = findeyes.correct_skews(angle1,angle2,array)

        angles, xcentroids, ycentroids, xmajor_axis_points, ymajor_axis_points = core.ellipses(self.scan.bone)
        slices, slice_angles = core.selectellipsesrange(angles)

        head_x = [xcentroids[i] for i in slices] 
        head_y = [ycentroids[i] for i in slices] 

        a,b,c,d = script.find_plane_from_ellipses(array_bone2,array_eyes2, slices, head_x, head_y, slice_angles)

        self.scan.params = a,b,c,d

    def midplaneMask(self):
        import nibabel as nib
        import os

        a,b,c,d = planes[s]
        shape = (self.scan.array).shape
        mask = -1000*np.ones((shape))
        x = np.arange(0,shape[0],1)
        z = np.linspace(0,shape[2],shape[0])

        xx, yy = np.meshgrid(z, (d-c*z-a*x)/b)
        y = yy.astype(int)

        for k in range(shape[1]):
            for i in range(shape[2]):
                mask[y[k][i]-1:y[k][i]+1, k, i] = 1000

        img = nib.Nifti1Image(mask, np.eye(4))
        midplaneMaskPath = os.path.join(os.path.split(path)[0], 'midplane.nii')
        nib.save(img, midplaneMaskPath)                                                        

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    form = StartupDlg(0)
    form.show()
    app.exec_()

