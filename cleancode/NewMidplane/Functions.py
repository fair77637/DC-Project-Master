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
    Sigma = np.transpose(X) * X * 1/m
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
    t = (d - A1*a - B1*b - C1*c)/(a**2 + b**2 + c**2)
    return t
    
    
    
    
    
    