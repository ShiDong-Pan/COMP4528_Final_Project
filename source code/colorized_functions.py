import numpy as np
import matplotlib.pyplot as plt
import imageio as im
import colorsys
import scipy
from scipy.sparse.linalg import spsolve

# Reference http://blog.sws9f.org/computer-vision/2017/09/07/colorization-using-optimization-python.html

def find_neighbors(width,center, img):
        neighbors = list()
        min_row,max_row = max(0,center[0] - width),min(img.shape[0], center[0] + width + 1)
        min_col,max_col  = max(0, center[1] - width),min(img.shape[1], center[1] + width + 1)
        for r in range(min_row, max_row ):
            for c in range(min_col, max_col):
                if r == center[0] and c == center[1]:
                    continue
                neighbors.append([r,c,img[r,c,0]])
                
        return neighbors
    
                
def affinity_a(neighbors,center):
        neigh_array = np.array(neighbors)
        neighs = neigh_array[:,2]  # neighbors
        center = center[2]   #  center
        diff = neighs - center   # distance between neighbors and center 
        sig = np.var(np.append(neighs, center))     # var of neighs and center
        sig = 1e-6 if sig < 1e-6 else sig
        wrs = np.exp(- np.power(diff,2) / (sig * 2.0))
        wrs = - wrs / np.sum(wrs)
        neigh_array[:,2] = wrs
        return neigh_array


def affinity_a_2(neighbors,center):
    
        neigh_array = np.array(neighbors)
        neighs = neigh_array[:,2]
        center = center[2]
        diss = neighs - neighs.mean()
        disc = center - neighs.mean()
        sig = np.var(np.append(neighs, center))
        sig = 1e-6 if sig < 1e-6 else sig
        wrs = 1+ (1/sig)* diss * disc
        wrs = - wrs / np.sum(wrs)
        neigh_array[:,2] = wrs

        return neigh_array


# translate (row,col) to/from sequential number
def rc2seq(r, c, rows):
        return c * rows + r


def seq2rc(seq, rows):
        r = seq % rows
        return r, int((seq - r) / rows)


# combine 3 channels of YUV to a RGB photo: n x n x 3 array
def yuv2rgb(img_yuv,y,u,v):
        (img_rows, img_cols, _) = img_yuv.shape
        rgb = [colorsys.yiq_to_rgb(y[i],u[i],v[i]) for i in range(len(y))]
        rgb = np.array(rgb)
        img_rgb = rgb.reshape(img_rows, img_cols,3, order='F')
        return img_rgb

    
def get_combined_yuv(test_image,mark_image):
    channel_Y,_,_ = colorsys.rgb_to_yiq(test_image[:,:,0],test_image[:,:,1],test_image[:,:,2])
    _,channel_U,channel_V = colorsys.rgb_to_yiq(mark_image[:,:,0],mark_image[:,:,1],mark_image[:,:,2])
    return channel_Y,channel_U,channel_V


def get_weightData(map_colored,combined_image,wd_width,affinity):
    (m,n,_) = combined_image.shape
    weightData = []
    for c in range(n):
        for r in range(m):
            center = np.hstack((r,c,combined_image[r,c][0]))
            neighbors = find_neighbors(wd_width, (r,c), combined_image)
            if not map_colored[r,c]:
                weights = affinity(neighbors,center)
                for e in weights:
                    weightData.append([center,(e[0],e[1]), e[2]]) 
            weightData.append([center, (center[0],center[1]), 1.])
    return weightData


def get_mat(weightData,rows,cols):
    size = rows * cols
    mat_data = [[rc2seq(e[0][0], e[0][1],rows), rc2seq(e[1][0], e[1][1], rows), e[2]] for e in weightData] 
    mat_rc = np.array(mat_data, dtype=np.integer)[:,0:2]
    mat_data = np.array(mat_data, dtype=np.float64)[:,2]

    mat = scipy.sparse.csr_matrix((mat_data, (mat_rc[:,0], mat_rc[:,1])), shape=(size, size))
    return mat


def get_uv(map_colored,combined_image,rows,cols):
    size = rows * cols
    u,v = np.zeros(size),np.zeros(size)
    colored = np.nonzero(map_colored.reshape(size, order='F'))
    img = combined_image.reshape(size, 3,order='F')
    u[colored],v[colored] = img[:,1][colored],img[:,2][colored]
    
    return u,v