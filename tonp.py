import nibabel as nib
import numpy as np
import os

def dicom2np(path):
    """ 
    Converts a dicom file to a numpy array. 
    Returns img_data, the numpy array and ConstPixelSpacing, 
    a list of the spacing between datapoints in mm for each dimension
    """
    pathDicom = os.path.dirname(path)

    import dicom
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(pathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])
    ImagePositionPatient = RefDs.ImagePositionPatient

    # Load dimensions based on the number of rows, columns, 
    # and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    
    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), 
        float(RefDs.SliceThickness))
    
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    
    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  

    return(ArrayDicom, ConstPixelSpacing, ImagePositionPatient)

def nifti2np(pathNifti):
    """ 
    Converts a NIFTI file to a numpy array. 
    Returns img_data, the numpy array and ConstPixelSpacing, 
    a list of the spacing between datapoints in mm for each dimension
    """
    img = nib.load(pathNifti)
    img_data = img.get_data()
    header = img.header
    img_data_shape = img_data.shape
    _extent = (0, img_data_shape[0] - 1, 0, img_data_shape[1] - 1, 0, img_data_shape[2] - 1)
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
    affine = img.header.get_base_affine()
    spacing = header['pixdim'][1:4]
    # Load spacing values
    ConstPixelSpacing = spacing[0], spacing[1], spacing[2]
    
    return(img_data, ConstPixelSpacing, affine)

def nrrd2np(pathNrrd):
    """ 
    Converts a nrrd file to a numpy array. 
    Returns img_data, the numpy array and ConstPixelSpacing, 
    a list of the spacing between datapoints in mm for each dimension
    """
    import nrrd
    readdata, options = nrrd.read(pathNrrd)
    ConstPixelSpacing = []
    spaceDirections = np.array(options['space directions'])
    ImagePositionPatient = options['space origin']
    for k in range(len(spaceDirections)):
        ConstPixelSpacing.append(np.linalg.norm(spaceDirections[k]))
    return(readdata, ConstPixelSpacing, ImagePositionPatient)

def thresholdnp(array, lo, hi):
    thresholded1 = np.multiply(array, (array>lo).astype(int))
    thresholded2 = np.multiply(thresholded1, (array<hi).astype(int))
    return thresholded2