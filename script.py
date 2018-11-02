# conda install mayavi
# conda install plotly
# pip install git+https://www.github.com/hbldh/b2ac for the ellipse function

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode()
import numpy as np
import matplotlib
from IPython.display import Image
import nibabel as nib

def reject_outliers(data, m=2):
    dx = np.abs(data[0] - np.median(data[0]))
    dy = np.abs(data[1] - np.median(data[1]))
    dz = np.abs(data[2] - np.median(data[2]))
    mdevx = np.median(dx)
    mdevy = np.median(dy)
    mdevz = np.median(dz)
    # if else is preventing from dividing by 0. if the median is 0, all
    # the differences have to be 0 becaues dx, dy, dz don't contain
    # negative numbers
    sx = dx/mdevx if mdevx else 0. 
    sy = dy/mdevy if mdevy else 0.
    sz = dz/mdevz if mdevz else 0.
    logicalx = sx<m
    logicaly = sy<m
    logicalz = sz<m
    e = data*logicalx*logicaly*logicalz
    data_return = np.array(e[:,e[1]!=0])
    return data_return

def dicom2np():
    from __main__ import PathDicom
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(PathDicom)
    reader.Update()
    
    # Get the 'vtkImageData' object from the reader
    imageData = reader.GetOutput()
    # Get the 'vtkPointData' object from the 'vtkImageData' object
    pointData = imageData.GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays()==1)
    # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)
    
    # Load dimensions using `GetDataExtent`
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
    
    # Load spacing values
    ConstPixelSpacing = reader.GetPixelSpacing()
    
    # Convert the `vtkArray` to a NumPy array
    ArrayDicom = vtk_to_numpy(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    return(ArrayDicom, ConstPixelSpacing)

def nii2np():
    # http://localhost:8888/edit/DC-project/script.py for nii
    from __main__ import PathDicom
    img = nib.load(PathDicom)
    img_data = img.get_data()
    
    img_data_shape = img_data.shape
    _extent = (0, img_data_shape[0] - 1, 0, img_data_shape[1] - 1, 0, img_data_shape[2] - 1)
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
    header = img.header
    spacing = header['pixdim'][1:4]
    # Load spacing values
    ConstPixelSpacing = spacing[0], spacing[1], spacing[2]
    
    return(img_data, ConstPixelSpacing)
    
def vtk2np(input_):
    from __main__ import PathDicom
    pointData = input_.GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays()==1)
    # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)
    
    # Load dimensions using `GetDataExtent`
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(PathDicom)
    reader.Update()
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
    
    # Load spacing values
    ConstPixelSpacing = reader.GetPixelSpacing()
    
    # Convert the `vtkArray` to a NumPy array
    ArrayDicom = vtk_to_numpy(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    return(ArrayDicom, ConstPixelSpacing)

def vtk2np_nii(input_):
    from __main__ import PathDicom
    pointData = input_.GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays()==1)
    # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)
    
    # Load dimensions using `GetDataExtent`
    from __main__ import PathDicom
    img = nib.load(PathDicom)
    img_data = img.get_data()
    img_data_shape = img_data.shape
    _extent = (0, img_data_shape[0] - 1, 0, img_data_shape[1] - 1, 0, img_data_shape[2] - 1)
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
    header = img.header
    spacing = header['pixdim'][1:4]
    # Load spacing values
    ConstPixelSpacing = spacing[0], spacing[1], spacing[2]
    # Convert the `vtkArray` to a NumPy array
    ArrayDicom = vtk_to_numpy(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    return(ArrayDicom, ConstPixelSpacing)

def heatmap(ArrayDicom, depth, x, y):

    heatmap = go.Heatmap(
            x = x,
            y = y,
            z = ArrayDicom[:,:,depth],
            colorscale = 'Greys'
            )

    layout = go.Layout(
        width = 600,
        height= 600,
        title='Slice number %i' % depth
    )
    data = [heatmap]
    fig = go.Figure(data=data, layout=layout)
    py.offline.iplot(fig)


    
def pyheatmap(ArrayDicom, slice_, axis):
    if axis==1:
        rot = np.rot90(ArrayDicom[slice_, :, :])
    elif axis==2:
        rot = np.rot90(ArrayDicom[:, slice_, :])
    elif axis==3:
        rot = np.rot90(ArrayDicom[:, :, slice_])
    else:
        print 'axis number needs to be 1, 2 or 3'
    #using pylab
    from pylab import *
    c = pcolor(rot)
    set_cmap('gray')
    colorbar()
    savefig('plt.png')
    #axis('equal')
    show()
    
def threshim_dicom(low, high):
    # http://www.programcreek.com/python/example/65395/vtk.vtkThreshold
    from __main__ import PathDicom
    #threshold data with bone
    import vtk
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(PathDicom)
    reader.Update()
    threshold = vtk.vtkImageThreshold ()
    threshold.SetInputConnection(reader.GetOutputPort())
    threshold.ThresholdByLower(low)  # remove all soft tissue
    threshold.ReplaceInOn()
    threshold.SetInValue(0)  # set all values below low to 0
    #threshold.ReplaceOutOn()
    #threshold.SetOutValue(1)  # set all values above 400 to 1
    threshold.Update()

    if high != 'none':
        threshold2 = vtk.vtkImageThreshold ()
        threshold2.SetInputConnection(threshold.GetOutputPort())
        threshold2.ThresholdByUpper(high)  # remove all soft tissue
        threshold2.Update()
        threshold2.ReplaceInOn()
        threshold2.SetInValue(0)  # set all values above high to 0
        threshold2.Update()
        thresholded = threshold2.GetOutput()
    else:
        thresholded = threshold.GetOutput()
    return thresholded

def threshim_nii(low, high):
    # http://www.programcreek.com/python/example/65395/vtk.vtkThreshold
    from __main__ import PathDicom
    #threshold data with bone
    img = nib.load(PathDicom)
    img_data = img.get_data()
    img_data_shape = img_data.shape
    header = img.header

    spacing = header['pixdim'][1:4]

    dataImporter = vtk.vtkImageImport()
    dataImporter.SetDataScalarTypeToShort()
    data_string = img_data.flatten('F').tostring()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataExtent(0, img_data_shape[0] - 1, 0, img_data_shape[1] - 1, 0, img_data_shape[2] - 1)
    dataImporter.SetWholeExtent(0, img_data_shape[0] - 1, 0, img_data_shape[1] - 1, 0, img_data_shape[2] - 1)
    dataImporter.SetDataSpacing(spacing[0], spacing[1], spacing[2])
    dataImporter.Update()
        
    threshold = vtk.vtkImageThreshold ()
    threshold.SetInputConnection(dataImporter.GetOutputPort())
    threshold.ThresholdByLower(low)  # remove all soft tissue
    threshold.ReplaceInOn()
    threshold.SetInValue(0)  # set all values below low to 0
    #threshold.ReplaceOutOn()
    #threshold.SetOutValue(1)  # set all values above 400 to 1
    threshold.Update()

    if high != 'none':
        threshold2 = vtk.vtkImageThreshold ()
        threshold2.SetInputConnection(threshold.GetOutputPort())
        threshold2.ThresholdByUpper(high)  # remove all soft tissue
        threshold2.Update()
        threshold2.ReplaceInOn()
        threshold2.SetInValue(0)  # set all values above high to 0
        threshold2.Update()
        thresholded = threshold2.GetOutput()
    else:
        thresholded = threshold.GetOutput()
    return thresholded

def orientation(numpy_array): #numpy_array needs to be binary/thresholded
    import b2ac.preprocess
    import b2ac.fit
    import b2ac.conversion
    numpy_array[abs(numpy_array)<0.01] = 0
    numpy_array[~np.isfinite(numpy_array)] = 0
    indices = np.nonzero(numpy_array)
    x = indices[1]
    y = indices[0]
    z = indices[0] - indices[0]
    xyz = reject_outliers(np.array([x,y,z]))
    x = xyz[0, :]
    y = xyz[1, :]
    points = np.zeros((len(indices[1]),2))
    for i in range(x.shape[0]):
        points[i, 0] = x[i]
        points[i, 1] = y[i]
    if points.shape[0] > 1:
        try:
            points, x_mean, y_mean = b2ac.preprocess.remove_mean_values(points)
            # Fit using NumPy methods in double precision.
            conic_double = b2ac.fit.fit_improved_B2AC_double(points)
            # Convert from conic coefficient form to general ellipse form.
            general_form_double = b2ac.conversion.conic_to_general_1(conic_double)
            general_form_double[0][0] += x_mean
            general_form_double[0][1] += y_mean
        except:
            general_form_double = 0
    else: general_form_double = 0
    return general_form_double # [x, y], [x_axis, y_axis], angle
    
def hist_control(data): # optimise number of bins to get the lowest threshold fraction. A thresholdfraction of 0.1 means that the bin with the 2nd highest frequency will have 0.1xthe frequency of the first highest bin.
    # (1) get rid of frequencies outside a standard deviation of 1. 
    """ 
    @ return: Return array. 1st number is the number of bins, 2nd number is the fraction achieved (where fraction is the 2nd highest frequency divided by the highest frequency, 3rd number is the midbin value of the bin corresponding to the highest frequency. 
    """
    from scipy.stats import norm
    hist, bins= np.histogram(data, bins=20)
    mu,sd = norm.fit(data)
    for i in range(len(hist)):
        if bins[i] < (mu-sd) or bins[i] > (mu+sd):
            hist[i] = 0
    # (2) optimise number of bins to get largest differences between 1st and second highest frequences
    fractions = np.zeros((50,3))
    for num_bins in range(10,60):
        hist, bins = np.histogram(data, bins=num_bins)
        midbins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        data = np.transpose(np.array([hist,midbins]))
        sorted_ = data[data[:,0].argsort()]
        fractions[num_bins-10][0]=num_bins # number of bins 
        fractions[num_bins-10][1]=sorted_[-2][0]/sorted_[-1][0] # fraction 
        fractions[num_bins-10][2]=sorted_[-1][1] # midbin value
    ind = np.where(fractions[1]==min(fractions[1]))[0][0]
    return fractions[ind]

def reshape(array, ConstPixelSpacing, a, c, width, depth):
    """ Reshape array to a 220 by 220 by 160 array with uniform spacing """
    reshaped_array1 = np.zeros((a,width,depth))

    xp = np.linspace(0, ConstPixelSpacing[0]*width, width)
    x = np.linspace(0, a-1, a)

    for j in range(width): # middle dimension
        for k in range(depth): # last dimension
            reshaped_array1[:,j,k] = np.interp(x, xp, array[:,j,k])

    reshaped_array2 = np.zeros((a,a,depth))

    for i in range(a): # first dimension
        for k in range(depth): # last dimension
            reshaped_array2[i,:,k] = np.interp(x, xp, reshaped_array1[i,:,k])

    reshaped_array3 = np.zeros((a,a,c))
    xp = np.linspace(0, ConstPixelSpacing[2]*depth, depth)
    x = np.linspace(0, c-1, c)

    for i in range(a): # first dimension
        for j in range(a): # middle dimension
            reshaped_array3[i,j,:] = np.interp(x, xp, reshaped_array2[i,j,:])

    return reshaped_array3

def ellipses(rotated2):
    angles=np.zeros(rotated2.shape[2])
    xcentroids=np.zeros(rotated2.shape[2])
    ycentroids=np.zeros(rotated2.shape[2])
    xmajor_axis_points=np.zeros(rotated2.shape[2])
    ymajor_axis_points=np.zeros(rotated2.shape[2])
    print "Starting ellipse fitting"
    for i in range(rotated2.shape[2]):
    #    print 'Slice number {}'.format(i)
        array_i=rotated2[:,:,i] # array to pass to 'orientation' must be thresholded already
        orientation_i=orientation(array_i)
        if orientation_i != 0: 
            angles[i]=orientation_i[2] # pick out the angle
            xcentroids[i]=orientation_i[0][0]
            ycentroids[i]=orientation_i[0][1]
            xmajor_axis_points[i] = xcentroids[i]+orientation_i[1][1]*np.cos(angles[i])
            ymajor_axis_points[i] = ycentroids[i]+orientation_i[1][1]*np.sin(angles[i])
        else: pass
    #    else: print 'Slice {} contains 0'.format(i)
    return angles, xcentroids, ycentroids, xmajor_axis_points, ymajor_axis_points

def select_ellipses_range(angles):
    # make all angles positive
    angles[angles < 0] += np.pi
    import scipy.signal as signal
    # First, design the Butterworth filter
    N  = 1   # Filter order - the higher the order, the sharper the dropoff
    pd = float(20)  # Cutoff period - the inverse is the cutoff frequency
    fs = 1 # Sample rate frequency
    nyq = 0.5*fs
    low = 1/pd
    wn = low/nyq # Cutoff frequency as a fraction
    b, a = signal.butter(N, wn, output='ba')
    filt_angles = signal.filtfilt(b,a, angles)
    angles_diff = filt_angles-angles
    indices = [] # indices holds the slice range of interest
    for i in range(len(angles)):
        if abs(angles_diff[i]) < 0.05 and 1.45<angles[i]<1.75:
            indices.append(i)
    from itertools import groupby
    z = zip(indices, indices[1:])
    tmp = ([list(j) for i, j in groupby(z, key=lambda x: (x[1] - x[0]) <= 1)])
    tmp = sorted(tmp, key=len) # longest lists at the end
    if len(tmp)>1:
        maxtmp = np.array(max(tmp, key=len))
        slices = list(range(maxtmp[0][0], maxtmp[-1][-1]+1))
        slice_angles = [angles[i] for i in slices]
    else:
        slices = []
        slice_angles = [np.pi/2]
#    if not 0<abs(np.mean(slice_angles))<20/360*2*np.pi or not np.pi/2-20/360*2*np.pi<abs(np.mean(slice_angles))<20/360*2*np.pi+np.pi/2:
#        slice_angles = ['use eyes']
#        slices = ['use eyes']
    return slices, slice_angles

def find_plane_from_ellipses(rotated2, array_eyes2, slices, head_x, head_y, head_angles):
    import find_eyes_reshape as fer
    rotated2[abs(rotated2)<0.01] = 0
    rotated2[~np.isfinite(rotated2)]=0
    H, edges, data, hist_data_2c = fer.hist3d_all(array_eyes2)
    ranges_1, ranges_2, certainty = fer.ranges(H,edges)
    c1,c2 = fer.coords(hist_data_2c, ranges_1, ranges_2)
    c = (c1+c2)/2.0
    c = np.roll(c,1)
    if slices != []:
        x=[]
        y=[]
        remaining_z = range(rotated2.shape[2]-1, slices[-1],-1)
        for i in remaining_z:
            x.append(np.mean(np.nonzero(rotated2[:,:,i])[1]))
            y.append(np.mean(np.nonzero(rotated2[:,:,i])[0]))
        centroids_array = np.concatenate((np.array([head_x,head_y,slices]).T, np.array([x,y,remaining_z]).T),axis=0).T
        centroids_array[~np.isfinite(centroids_array)]=0
        centroids_array[abs(centroids_array)<0.01] = 0
        centroids_array = reject_outliers(centroids_array)
        from scipy.optimize import curve_fit
        def f(x,A,B):
            return A*x + B
        A_x1,B_x1 = curve_fit(f, centroids_array[2], centroids_array[0])[0] # your data x, y to fit
        A_y1,B_y1 = curve_fit(f, centroids_array[2], centroids_array[1])[0] # your data x, y to fit

        z = np.median(slices)
        x = A_x1*z + B_x1
        y = A_y1*z + B_y1
        # point that the plane goes through, p
        p = np.array([x, y, z]) # coordinates when z = mean(indices)
        print p
        v1 = np.array([A_x1/2, A_y1/2, 1])
        mean_angle = np.mean(head_angles)
        #mean_angle = hist_control(head_angles)[2]
        v2a = np.array([np.cos(mean_angle+np.pi/2), np.sin(mean_angle+np.pi/2), 0])
        v2b = (p - c)
        v2b = v2b/(np.sum(np.power(v2b,2)))
        v2 = (v2a+v2b)/2.0
        normal = np.cross(v1,v2)
        # a plane is ax+by+cz = d - find d
        d = np.dot(p,normal)
        # superpose plane onto slices
        a = normal[0]
        b = normal[1]
        c = normal[2]
        # plane equation is ax+by+cz = d
        # for each slice substitute slice number into z to get equation
    else:
        v1 = np.array([0,0,1])
        v2 = np.array([1,0,0])
        normal = np.cross(v1,v2)
        d = np.dot(c,normal)
        a = normal[0]
        b = normal[1]
        c = normal[2]
    return a,b,c,d

def visualise_single(rotated2, a,b,c,d, slice_no):
    slice_ = rotated2[:,:,slice_no]
    heatmap = go.Heatmap(
            z = slice_,
            colorscale=[[0.0, 'rgb(160,160,160)'], 
                        [0.1111111111111111, 'rgb(20,20,20)'], 
                        [0.2222222222222222, 'rgb(40,40,40)'], 
                        [0.3333333333333333, 'rgb(100,100,100)'], 
                        [0.4444444444444444, 'rgb(120,120,120)'], 
                        [0.5555555555555556, 'rgb(140,140,140)'], 
                        [0.6666666666666666, 'rgb(160,160,160)'], 
                        [0.7777777777777778, 'rgb(170,170,170)'], 
                        [0.8888888888888888, 'rgb(250,250,250)'], 
                        [1.0, 'rgb(250,250,250)']]
            )
    x = np.linspace(0,slice_.shape[0], 2)
    y = (d-c*slice_no-a*x)/b
    midline = go.Scatter(
        x = x,
        y = y,
        mode = 'lines'
    )
    layout = go.Layout(
        width = 600,
        height= 600,
    #     title='Fig. 10. Slice number %i' % slice_no,
        title='Fig. 10. Slice of the midplane found', # title for report
        showlegend = False,
            margin= go.Margin(
            l=50,
            r=50,
            b=100,
            t=50,
            pad=4
            ),
        
    )

    data = [heatmap, midline]
    fig = go.Figure(data=data, layout=layout)
    py.offline.iplot(fig)
    return None

def save_midplane(rotated2, a,b,c,d, samp):
    for slice_no in range(0,rotated2.shape[2],5):
        slice_ = rotated2[:,:,slice_no]
        heatmap = go.Heatmap(
                z = slice_,
                colorscale=[[0.0, 'rgb(160,160,160)'], 
                        [0.1111111111111111, 'rgb(20,20,20)'], 
                        [0.2222222222222222, 'rgb(40,40,40)'], 
                        [0.3333333333333333, 'rgb(100,100,100)'], 
                        [0.4444444444444444, 'rgb(120,120,120)'], 
                        [0.5555555555555556, 'rgb(140,140,140)'], 
                        [0.6666666666666666, 'rgb(160,160,160)'], 
                        [0.7777777777777778, 'rgb(170,170,170)'], 
                        [0.8888888888888888, 'rgb(250,250,250)'], 
                        [1.0, 'rgb(250,250,250)']]
                )

        x = np.linspace(0,slice_.shape[0], 2)
        y = (d-c*slice_no-a*x)/b

        midline = go.Scatter(
            x = x,
            y = y,
            mode = 'lines'
        )

        layout = go.Layout(
            width = 600,
            height= 600,
            title='Slice number %i' % slice_no
        )
        data = [heatmap, midline]
        fig = go.Figure(data=data, layout=layout)
        py.plotly.image.save_as(fig, filename='{}_slice_{}.png'.format(samp, slice_no))
    return None