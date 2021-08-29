import sift_features
import numpy
from pylab import *
from scipy import ndimage
from skimage.filters import gabor
import skimage
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


def combine(*args):
    output = args[0]
    for i in range(1, len(args)):
        output += args[i]
    #print(output.shape)
    return output

def linear_svm(x_train, y_train, cm=0):
    #parameters c
    c = 10**(cm)
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = LinearSVC(C=c)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = svm_train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function_svm(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def lr(x_train, y_train, cm=0):
    c = 10**(cm)
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = LogisticRegression(C=c, solver='sag', multi_class= 'auto', max_iter=1000)
    num_train = y_train.shape[0]
    if num_train==x_train.shape[0]:
        y_labels = svm_train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function_svm(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def randomforest(x_train, y_train, n_tree = 500, max_dep = 100):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = RandomForestClassifier(n_estimators=n_tree, max_depth=max_dep)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = train_model_prob(classifier, x_train, y_train)
    else:
        y_labels = test_function_prob(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def erandomforest(x_train, y_train, n_tree = 500, max_dep = 100):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = ExtraTreesClassifier(n_estimators=n_tree, max_depth=max_dep)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = train_model_prob(classifier, x_train, y_train)
    else:
        y_labels = test_function_prob(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def svm_train_model(model, x, y, k=3):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(np.asarray(x))
    kf = StratifiedKFold(n_splits=k)
    ni = np.unique(y)
    num_class = ni.shape[0]
    y_predict = np.zeros((len(y), num_class))
    for train_index, test_index in kf.split(x,y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        y_label = []
        for i in y_pred:
            binary_label = np.zeros((num_class))
            binary_label[int(i)] = 1
            y_label.append(binary_label)
        y_predict[test_index,:] = np.asarray(y_label)
    return y_predict

def test_function_svm(model, x_train, y_train, x_test):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(np.asarray(x_train))
    x_test = min_max_scaler.transform(np.asarray(x_test))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_label = []
    ni = np.unique(y_train)
    num_class = ni.shape[0]
    for i in y_pred:
        binary_label = np.zeros((num_class))
        binary_label[int(i)] = 1
        y_label.append(binary_label)
    y_predict = np.asarray(y_label)
    return y_predict

def train_model_prob(model, x, y, k=3):
##    min_max_scaler = preprocessing.MinMaxScaler()
##    x = min_max_scaler.fit_transform(np.asarray(x))
    kf = StratifiedKFold(n_splits=k)
    ni = np.unique(y)
    num_class = ni.shape[0]
    y_predict = np.zeros((len(y), num_class))
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train,y_train)
        y_predict[test_index,:] = model.predict_proba(x_test)
    return y_predict

def test_function_prob(model, x_train, y_train, x_test):
    ##min_max_scaler = preprocessing.MinMaxScaler()
    #x_train = min_max_scaler.fit_transform(np.asarray(x_train))
    #x_test = min_max_scaler.transform(np.asarray(x_test))
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_test)
    return y_pred

def conVector(img):
    try:
        img_vector=numpy.concatenate((img))
    except:
        img_vector=img
    return img_vector

def FeaCon2(img1, img2):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        feature_vector = numpy.concatenate((image1, image2), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)

def FeaCon3(img1, img2, img3):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        feature_vector = numpy.concatenate((image1, image2, image3), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)


def FeaCon4(img1, img2, img3, img4):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        image4 = conVector(img4[i, :])
        feature_vector = numpy.concatenate((image1, image2, image3, image4), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)

def histLBP(image,radius,n_points):
    # 'uniform','default','ror','var'
    lbp = local_binary_pattern(image, n_points, radius, method='nri_uniform')
    n_bins = 59
    hist,ax=numpy.histogram(lbp,n_bins,[0,59])
    return hist

def all_lbp(image):
    # global and local
    feature = []
    for i in range(image.shape[0]):
        feature_vector = histLBP(image[i,:,:], radius=1.5, n_points=8)
        feature.append(feature_vector)
    return numpy.asarray(feature)

#
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
    feature = []
    for i in range(image.shape[0]):
        feature_vector = hog_features_patches(image[i,:,:], 4, 4)
        feature.append(feature_vector)
    return numpy.asarray(feature)

def all_sift(image):
    width, height = image[0, :, :].shape
    min_length = numpy.min((width, height))
    feature = []
    for i in range(image.shape[0]):
        img = numpy.asarray(image[i, 0:width, 0:height])
        extractor = sift_features.SingleSiftExtractor(min_length)
        feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
        # dimension 128 for all images
        w, h = feaArrSingle.shape
        feature_vector = numpy.reshape(feaArrSingle, (h,))
        feature.append(feature_vector)
    return numpy.asarray(feature)


def gau(left, si):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.gaussian_filter(left[i, :, :], sigma=si))
    return np.asarray(img)

def gauD(left, si, or1, or2):
    img  = []
    for i in range(left.shape[0]):
        img.append(ndimage.gaussian_filter(left[i,:,:],sigma=si, order=[or1,or2]))
    return np.asarray(img)

def gab(left,the,fre):
    fmax=numpy.pi/2
    a=numpy.sqrt(2)
    freq=fmax/(a**fre)
    thea=numpy.pi*the/8
    img = []
    for i in range(left.shape[0]):
        filt_real,filt_imag=numpy.asarray(gabor(left[i,:,:],theta=thea,frequency=freq))
        img.append(filt_real)
    return numpy.asarray(img)

def gaussian_Laplace1(left):
    return ndimage.gaussian_laplace(left,sigma=1)

def gaussian_Laplace2(left):
    return ndimage.gaussian_laplace(left,sigma=2)

def laplace(left):
    return ndimage.laplace(left)

def sobelxy(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i, :, :]))
    return np.asarray(img)

def sobelx(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i,:,:], axis=0))
    return np.asarray(img)

def sobely(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i, :, :], axis=1))
    return np.asarray(img)

#max filter
def maxf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.maximum_filter(image[i,:,:],size))
    return np.asarray(img)

#median_filter
def medianf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.median_filter(image[i,:,:],size))
    return np.asarray(img)

#mean_filter
def meanf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.convolve(image[i,:,:], numpy.full((3, 3), 1 / (size * size))))
    return np.asarray(img)

#minimum_filter
def minf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.minimum_filter(image[i,:,:],size))
    return np.asarray(img)

def lbp(image):
    img = []
    for i in range(image.shape[0]):
        # 'uniform','default','ror','var'
        lbp = local_binary_pattern(image[i,:,:], 8, 1.5, method='nri_uniform')
        img.append(np.divide(lbp,59))
    return np.asarray(img)


def hog_feature(image):
    try:
        img = []
        for i in range(image.shape[0]):
            img1, realImage = hog(image[i, :, :], orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                                transform_sqrt=False, feature_vector=True)
            img.append(realImage)
        data = np.asarray(img)
    except: data = image
    return data

def mis_match(img1,img2):
    n, w1,h1=img1.shape
    n, w2,h2=img2.shape
    w=min(w1,w2)
    h=min(h1,h2)
    return img1[:, 0:w,0:h],img2[:, 0:w,0:h]

def mixconadd(img1, w1, img2, w2):
    img11,img22=mis_match(img1,img2)
    return numpy.add(img11*w1,img22*w2)

def mixconsub(img1, w1, img2, w2):
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
    img = []
    for i in range(left.shape[0]):
        current = skimage.measure.block_reduce(left[i,:,:], (kel1,kel2),numpy.max)
        img.append(current)
    return np.asarray(img)
