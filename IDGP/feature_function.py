from feature_extractors import fetureDIF
from feature_extractors import hog_features_patches as hog_features
from feature_extractors import histLBP
import sift_features
import numpy
from skimage.feature import hog

def root_con_number3(v1, v2, v3):
    return numpy.asarray([v1, v2, v3])

def root_con_number2(v1, v2):
    return numpy.asarray([v1, v2])

def root_con2(v1, v2):
    v11 = numpy.concatenate((v1))
    v12 = numpy.concatenate((v2))
    feature_vector=numpy.concatenate((v11, v12),axis=0)
    return feature_vector

def root_con3(v1, v2, v3):
    v11 = numpy.concatenate((v1))
    v12 = numpy.concatenate((v2))
    v13 = numpy.concatenate((v3))
    feature_vector=numpy.concatenate((v11, v12, v13),axis=0)
    return feature_vector

def root_con(*args):
    feature_vector=numpy.concatenate((args),axis=0)
    return feature_vector


def all_dif(image):
    #global and local
    feature_vector = fetureDIF(image)
    # dimension 20 for all type images
    return feature_vector

def all_histogram(image):
    # global and local
    n_bins = 32
    hist, ax = numpy.histogram(image, n_bins, [0, 1])
    # dimension 24 for all type images
    return hist

def global_hog(image):
    feature_vector = hog_features(image, 20, 10)
    # dimension 144 for 128*128
    return feature_vector

def local_hog(image):
    try:
        feature_vector=hog_features(image,10,10)
    except: feature_vector = numpy.concatenate(image)
    #dimension don't know
    return feature_vector

def HoGFeatures(image):
    img,realImage=hog(image,orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(3, 3), block_norm='L2-Hys', visualise=True,
                transform_sqrt=False, feature_vector=True)
    return realImage

def local_hog_small(image):
    try:
        feature=HoGFeatures(image)
        feature_vector = numpy.concatenate(feature)
    except:
        feature_vector = numpy.concatenate(image)
    #dimension don't know
    return feature_vector

def all_lbp(image):
    # global and local
    feature_vector = histLBP(image, 1.5, 8)
    # dimension 59 for all images
    return feature_vector

def all_sift(image):
    # global and local
    width,height=image.shape
    min_length=numpy.min((width,height))
    img=numpy.asarray(image[0:width,0:height])
    extractor = sift_features.SingleSiftExtractor(min_length)
    feaArrSingle = extractor.process_image(img[0:min_length,0:min_length])
    # dimension 128 for all images
    w,h=feaArrSingle.shape
    feature_vector=numpy.reshape(feaArrSingle, (h,))
    return feature_vector

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

def feature_length(ind, instances, toolbox):
    func=toolbox.compile(ind)
    try:
        feature_len = len(func(instances))
    except: feature_len=0
    return feature_len,
