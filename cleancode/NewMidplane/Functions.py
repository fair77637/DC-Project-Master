import numpy as np
import nibabel as nib
import os

def featureNormalize(X):
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1.
    mu = np.mean(X,0)
    X_norm = X[...,:] - mu
    #std = np.std(X,0)
    return X_norm, mu

def pca(X):
    [m,n] = X.shape
    U = np.zeros(n)
    S = np.zeros(n)
    T = np.transpose(X)
    Sigma = np.matmul(T,X)* 1/m
    [U,S,V] = np.linalg.svd(Sigma, full_matrices=True)
    #   Note for 3D eigenvectors U, the third eigenvector U[:,3]is     #   perpendicular
    #   to Eigenvector1 and eigenvector2.
    return U,S

#   When plotting the 2D principle component plane, the eigenvector U[:,3] won't     change, but the value of d changes as you choose to use mean(X) or mean(X_norm).

def reshape(array,ConstPixelSpacing):
    #   Reshape the array to make each pixel one cubic mm, get rid of the bias of    PixelSpacing.
    [width, height, depth] = array.shape
    a = int((width)*ConstPixelSpacing[0])
    b = int((height)*ConstPixelSpacing[1])
    c = int((depth)*ConstPixelSpacing[2])
    reshapedArray1 = np.zeros((a,height,depth))
    #adjust xp as [0,1ConstPixelSpacing,2CPS..
    xp = np.linspace(0, (width-1)*ConstPixelSpacing[0], width) 
    x  = np.linspace(0, a-1, a)
    for j in range(height):
        for k in range(depth):
            reshapedArray1[:,j,k] = np.interp(x,xp,array[:,j,k])

    reshapedArray2 = np.zeros((a,b,depth))
    yp = np.linspace(0,(height-1)*ConstPixelSpacing[1],height)
    y = np.linspace(0,b-1,b)
    for i in range(a):
        for k in range(depth):
            reshapedArray2[i,:,k] = np.interp(y,yp,reshapedArray1[i,:,k])

    reshapedArray3 = np.zeros((a,b,c))
    zp = np.linspace(0,(depth-1)*ConstPixelSpacing[2],depth)
    z = np.linspace(0,c-1,c)
    for i in range(a):
        for j in range(b):
            reshapedArray3[i,j,:] = np.interp(z,zp,reshapedArray2[i,j,:])
    return reshapedArray3

def plane_parameters(U,p,ConstPixelSpacing):
    a,b,c = U[0,0],U[1,0],U[2,0]
    q = np.zeros((3,1))
    q[0],q[1],q[2] = a,b,c
    p1 = np.divide(p,ConstPixelSpacing)
    d = np.dot(p1,q)
    return a,b,c,d

def ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)
    
    
    
def sym_t(A1,B1,C1,a,b,c,d):
    "" "Calculate the normal vector to the plane """
    t = (d - A1*a - B1*b - C1*c)/(a**2 + b**2 + c**2)
    return t
    
def mirror(a,b,c,p,boneReshaped):
    q  = np.zeros((3,1));
    q[0],q[1],q[2] = a,b,c;
    d = np.dot(p,q)
    #     Preliminary Midplane split
    crossShape = boneReshaped[:,:,0].shape
    mask = np.zeros(boneReshaped.shape)
    for i in range(boneReshaped.shape[2]):
        maski = np.fromfunction(lambda x,y: x > ((d-i*c-y*b)/a), crossShape)
#         mask2 = np.fromfunction(lambda x,y: x < ((d-i*c-y*b)/a+2), crossShape)
#         maski = np.multiply(mask1, mask2)
        mask[:,:,i] = maski
        ##Left and Right skull divide
    left_skull = np.multiply(boneReshaped, mask==0)
    right_skull = np.multiply(boneReshaped, mask==1)
    ##Coordinates of Left and Right skull pixels
    X_left = np.where(left_skull>0)
    X_right = np.where(right_skull>0)
        ##Mirror the left pixels to the right about the plane
    import Functions
    t = Functions.sym_t(X_left[0],X_left[1],X_left[2],a,b,c,d)
    mir_A1,mir_B1,mir_C1 = X_left[0] + 2*t*a,X_left[1] + 2*t*b,X_left[2] + 2*t*c
    mir_A1 = mir_A1.astype(int)
    mir_B1 = mir_B1.astype(int)
    mir_C1 = mir_C1.astype(int)
    mir_A1 = np.array(mir_A1)
    mir_B1 = np.array(mir_B1)
    mir_C1 = np.array(mir_C1)

    Right_array = np.matrix.transpose(np.row_stack((X_right[0],X_right[1],X_right[2])))
    mirror_array = np.matrix.transpose(np.row_stack((mir_A1,mir_B1,mir_C1)))
    return Right_array,mirror_array
    
def cul_difference(Right_array,mirror_array,sample_size):
    from scipy.spatial import KDTree
    """Searching for the nearest neighbor for randomly selected points, and sum up the differences"""
    difference = 0
    sel = np.random.randint(0, len(Right_array)-1 , (1,sample_size))
    Selected_array = Right_array[sel,:][0]
    for i in range(0,len(Selected_array)):
        point = Selected_array[i]
        tree = KDTree(mirror_array, leafsize=mirror_array.shape[0]+1)
        distances, ndx = tree.query([point], k=1)
        Sq_distance = distances**2
        difference += Sq_distance
    return difference

def Close_Neighboring_pairs(Right_array,mirror_array,r):
    from scipy.spatial import cKDTree
    ## r is the radius for close-neighboring search
    tree1 = cKDTree(mirror_array, leafsize=mirror_array.shape[0]+1)
    tree2 = cKDTree(Right_array, leafsize=Right_array.shape[0]+1)
    Num_pairs = cKDTree.count_neighbors(tree2, tree1, r, p=2.)
    ###Note the value generally used is 3.
    return Num_pairs



def unit_vector(k):
    import numpy as np
    k_unit = k/np.sqrt(k[0]**2+ k[1]**2 + k[2]**2)
    return k_unit
    
def rotate_vector(v,k,theta):
    """v is the original vecotr, k is the rotation axis, theta is the rotation angle in degrees"""
    """Note theta must be a floating number eg 30.0"""
    rad = theta/180*np.pi
    v_rot = np.cos(rad)*v + (1-np.cos(rad))*np.dot(v,k)*k + np.sin(rad)*(np.cross(v,k))
    return v_rot
    