import sift_features
import numpy
from pylab import *
from scipy import ndimage
from skimage.filters import sobel
from skimage.filters import gabor
from skimage.filters import gaussian
import skimage
from skimage.feature import local_binary_pattern
from skimage.feature import hog

def root_con(*args):
    feature_vector=numpy.concatenate((args),axis=0)
    return feature_vector

def conVector(img):
    try:
        img_vector=numpy.concatenate((img))
    except ValueError:
        img_vector=img
    return img_vector

def root_conVector2(img1, img2):
    image1=conVector(img1)
    image2=conVector(img2)
    feature_vector=numpy.concatenate((image1, image2),axis=0)
    return feature_vector

def root_conVector3(img1, img2, img3):
    image1=conVector(img1)
    image2=conVector(img2)
    image3=conVector(img3)
    feature_vector=numpy.concatenate((image1, image2, image3),axis=0)
    return feature_vector

def all_lbp(image):
    #uniform_LBP
    lbp=local_binary_pattern(image, P=8, R=1.5, method='nri_uniform')
    n_bins = 59
    hist,ax=numpy.histogram(lbp,n_bins,[0,59])
    return hist

def HoGFeatures(image):
    try:
        img,realImage=hog(image,orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                    transform_sqrt=False, feature_vector=True)
        return realImage
    except:
        return image

def hog_features_patches(image,patch_size,moving_size):
    img=numpy.asarray(image)
    width, height = img.shape
    w = int(width / moving_size)
    h = int(height / moving_size)
    patch = []
    for i in range(0, w):
        for j in range(0, h):
            patch.append([moving_size * i, moving_size * j])
    hog_features = numpy.zeros((len(patch)))
    realImage=HoGFeatures(img)
    for i in range(len(patch)):
        hog_features[i] = numpy.mean(
            realImage[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)])
    return hog_features

def global_hog_small(image):
    feature_vector = hog_features_patches(image, 4, 4)
    return feature_vector

def all_sift(image):
    width,height=image.shape
    min_length=numpy.min((width,height))
    img=numpy.asarray(image[0:width,0:height])
    extractor = sift_features.SingleSiftExtractor(min_length)
    feaArrSingle = extractor.process_image(img[0:min_length,0:min_length])
    # dimension 128 for all images
    w,h=feaArrSingle.shape
    feature_vector=numpy.reshape(feaArrSingle, (h,))
    return feature_vector

def gau(left, si):
    return gaussian(left,sigma=si)

def gauD(left, si, or1, or2):
    return ndimage.gaussian_filter(left,sigma=si, order=[or1,or2])

def gab(left,the,fre):
    fmax=numpy.pi/2
    a=numpy.sqrt(2)
    freq=fmax/(a**fre)
    thea=numpy.pi*the/8
    filt_real,filt_imag=numpy.asarray(gabor(left,theta=thea,frequency=freq))
    return filt_real

def laplace(left):
    return ndimage.laplace(left)

def gaussian_Laplace1(left):
    return ndimage.gaussian_laplace(left,sigma=1)

def gaussian_Laplace2(left):
    return ndimage.gaussian_laplace(left,sigma=2)

def sobelxy(left):
    left=sobel(left)
    return left

def sobelx(left):
    left=ndimage.sobel(left,axis=0)
    return left

def sobely(left):
    left=ndimage.sobel(left,axis=1)
    return left

#max filter
def maxf(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x = ndimage.maximum_filter(x,size)
    return x

#median_filter
def medianf(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x = ndimage.median_filter(x,size)
    return x

#mean_filter
def meanf(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x = ndimage.convolve(x, numpy.full((3, 3), 1 / (size * size)))
    return x

#minimum_filter
def minf(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x=ndimage.minimum_filter(x,size)
    return x

def lbp(image):
    # 'uniform','default','ror','var'
    try:
        lbp = local_binary_pattern(image, 8, 1.5, method='nri_uniform')
        lbp = np.divide(lbp,59)
    except: lbp = image
    return lbp


def hog_feature(image):
    try:
        img, realImage = hog(image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                            transform_sqrt=False, feature_vector=True)
        return realImage
    except:
        return image

def mis_match(img1,img2):
    w1,h1=img1.shape
    w2,h2=img2.shape
    w=min(w1,w2)
    h=min(h1,h2)
    return img1[0:w,0:h],img2[0:w,0:h]

def mixconadd(img1, w1,img2, w2):
    img11,img22=mis_match(img1,img2)
    return numpy.add(img11*w1,img22*w2)

def mixconsub(img1, w1,img2, w2):
    img11,img22=mis_match(img1,img2)
    return numpy.subtract(img11*w1,img22*w2)

def sqrt(left):
    with numpy.errstate(divide='ignore',invalid='ignore'):
        x = numpy.sqrt(left,)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 1
            x[numpy.isnan(x)] = 1
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 1
    return x

def relu(left):
    return (abs(left)+left)/2

def maxP(left, kel1, kel2):
    try:
        current = skimage.measure.block_reduce(left,(kel1,kel2),numpy.max)
    except ValueError:
        current=left
    return current

