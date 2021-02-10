import numpy
from pylab import *
from scipy import ndimage
import skimage

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

def addZerosPad(final,current):
    M,N=final.shape
    m1,n1=current.shape
    pUpperSize=int((M-m1)/2)
    pDownSize=int(M-pUpperSize-m1)
    pLeftSize=int((N-n1)/2)
    pRightSize=int(N-pLeftSize-n1)
    PUpper=numpy.zeros((pUpperSize,n1))
    PDown=numpy.zeros((pDownSize,n1))
    current=numpy.concatenate((PUpper,current,PDown),axis=0)
    m2,n2=current.shape
    PLeft=numpy.zeros((m2,pLeftSize))
    PRight=numpy.zeros((m2,pRightSize))
    current=numpy.concatenate((PLeft,current,PRight),axis=1)
    return current

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

def conVector(img):
    try: 
        img_vector=numpy.concatenate((img))
    except ValueError:
        img_vector=img
    return img_vector


def maxP(left, kel1, kel2):
    try:
        current = skimage.measure.block_reduce(left,(kel1,kel2),numpy.max)
    except ValueError:
        current=left
    return current

def ZeromaxP(left, kel1, kel2):
    try:
        current = skimage.measure.block_reduce(left,(kel1,kel2),numpy.max)
    except ValueError:
        current=left
    zero_p=addZerosPad(left,current)
    return zero_p

def random_filters(filter_size):
    filters = []
    for i in range(filter_size*filter_size):
        filters.append(numpy.random.randint(-5, 5))
    return filters

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

def root_conVector4(img1, img2, img3, img4):
    image1=conVector(img1)
    image2=conVector(img2)
    image3=conVector(img3)
    image4=conVector(img4)
    feature_vector=numpy.concatenate((image1, image2, image3, image4),axis=0)
    return feature_vector

def conv_filters(image, filters):
    length = len(filters)
    size = int(sqrt(length))
    filters_resize = numpy.asarray(filters).reshape(size, size)
    img = ndimage.convolve(image, filters_resize)
    return img
