
import numpy as np
from numpy.linalg import cholesky,inv,eig,solve

def cca(X,Y,n_components=1,reg=0.0001):
    sx = np.size(X,axis=1)
    sy = np.size(Y,axis=1)
    if sx>sy:
        X,Y = Y,X
        sx,sy = sy,sx

    z=np.concatenate((X,Y),axis=1)
    C=np.cov(z.transpose())
    Cxx = C[0:sx,:][:,0:sx] + reg*np.eye(sx)
    Cxy = C[0:sx,:][:,sx:sx+sy]
    Cyx = Cxy.transpose()
    Cyy = C[sx:sx+sy,:][:,sx:sx+sy] + reg*np.eye(sy)
    
    Rx = cholesky(Cxx).transpose()
    invRx = inv(Rx)
    Cyyinv_yx=solve(Cyy,Cyx)
    Z = np.dot(invRx.transpose(),Cxy)
    Z = np.dot(Z,Cyyinv_yx)
    Z = np.dot(Z,invRx)
    Z = 0.5*(Z.transpose() + Z)
    D,Wx = eig(Z) # basis in h (X)
    r = np.sqrt(np.real(D)) # as the original r we get is lamda^2
    Wx = np.dot(invRx,Wx) #  actual Wx values
    index_r = np.argsort(-r)
    r = -np.sort(-r)
    Wx = Wx[:,index_r]
    Wy = np.dot(Cyyinv_yx,Wx)

    # normalize Wy
    for i in range(np.size(Wy,axis=1)):
        if r[i] != 0:
            Wy[:,i] = Wy[:,i]/r[i]
        
    # change wx and wy
    if sx>sy:
        Wx,Wy = Wy,Wx
    
    return Wx[:,:n_components],Wy[:,:n_components],r[:n_components]

