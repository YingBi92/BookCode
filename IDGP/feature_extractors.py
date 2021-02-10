import numpy
from skimage.feature import local_binary_pattern
from skimage.feature import hog


def featureMeanStd(region):
    #print(region)
    mean=numpy.mean(region)
    std=numpy.std(region)
    #print(mean,std)
    return mean,std

def fetureDIF(image):
    feature=numpy.zeros((20))
    width,height=image.shape
    width1=int(width/2)
    height1=int(height/2)
    width2=int(width/4)
    height2=int(height/4)
    #A1B1C1D1
    feature[0],feature[1]=featureMeanStd(image)
    #A1E1OG1
    feature[2],feature[3]=featureMeanStd(image[0:width1,0:height1])
    #E1B1H1O
    feature[4],feature[5]=featureMeanStd(image[0:width1,height1:height])
    #G1OF1D1
    feature[6],feature[7]=featureMeanStd(image[width1:width,0:height1])
    #OH1C1F1
    feature[8],feature[9]=featureMeanStd(image[width1:width,height1:height])
    #A2B2C2D2
    feature[10],feature[11]=featureMeanStd(image[width2:(width2+width1),height2:(height1+height2)])
    #G1H1
    feature[12],feature[13]=featureMeanStd(image[width1,:])
    #E1F1
    feature[14],feature[15]=featureMeanStd(image[:,height1])
    #G2H2
    feature[16],feature[17]=featureMeanStd(image[width1,height2:(height1+height2)])
    #E2F2
    feature[18],feature[19]=featureMeanStd(image[width2:(width2+width1),height1])
    return feature


def LBP(image,radius,n_points, method = 'nri_uniform'):
    # 'uniform','default','ror','var'
    lbp = local_binary_pattern(image, n_points, radius, method)
    return lbp

def histLBP(image,radius,n_points):
    #uniform_LBP
    lbp=LBP(image,radius=radius,n_points=n_points)
    n_bins = 59
    hist,ax=numpy.histogram(lbp,n_bins,[0,n_bins])
    return hist

def HoGFeatures(image):
    img,realImage=hog(image,orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                transform_sqrt=False, feature_vector=True)
    return realImage

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

