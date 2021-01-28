from pylab import *
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist
from skimage.feature import hog

def gaussian_1(left):
    left=ndimage.gaussian_filter(left,sigma=1)
    return left

#gaussian filter with sigma=1 with the second derivatives
def gaussian_11(left):
    left = ndimage.gaussian_filter(left, sigma=1,order=1)
    return left

#gaussian_gradient_magnitude(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gauGM(left):
    left=ndimage.gaussian_gradient_magnitude(left,sigma=1)
    return left

#gaussian_laplace(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gaussian_Laplace1(left):
    left=ndimage.gaussian_laplace(left,sigma=1)
    return left

def gaussian_Laplace2(left):
    left=ndimage.gaussian_laplace(left,sigma=2)
    return left

#laplace(input, output=None, mode='reflect', cval=0.0)
def laplace(left):
    left=ndimage.laplace(left)
    return left

#sobel(input, axis=-1, output=None, mode='reflect', cval=0.0)
def sobelx(left):
    left=ndimage.sobel(left,axis=0)
    return left

def sobely(left):
    left=ndimage.sobel(left,axis=1)
    return left

def lbp(image):
    # 'uniform','default','ror','var'
    lbp = local_binary_pattern(image, 8, 1.5, method='nri_uniform')
    lbp=np.divide(lbp,59)
    return lbp

def hist_equal(image):
    equal_image = equalize_hist(image, nbins=256, mask=None)
    return equal_image

def hog_feature(image):
    img, realImage = hog(image, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L2-Hys', visualise=True,
                         transform_sqrt=False, feature_vector=True)
    return realImage

def regionS(left,x,y,windowSize):
    width,height=left.shape
    x_end = min(width, x+windowSize)
    y_end = min(height, y+windowSize)
    slice = left[x:x_end, y:y_end]
    return slice

def regionR(left, x, y, windowSize1,windowSize2):
    width, height = left.shape
    x_end = min(width, x + windowSize1)
    y_end = min(height, y + windowSize2)
    slice = left[x:x_end, y:y_end]
    return slice
