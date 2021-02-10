import numpy as np
from sklearn.svm import LinearSVC
from sklearn import preprocessing

def feature_length(ind, instances, toolbox):
    func=toolbox.compile(ind)
    try:
        feature_len = len(func(instances))
    except: feature_len=0
    return feature_len,


def evalTest(toolbox, individual, trainData, trainLabel, test, testL):
    func = toolbox.compile(expr=individual)
    train_tf = []
    test_tf = []
    for i in range(0, len(trainLabel)):
        train_tf.append(np.asarray(func(trainData[i, :, :])))
    for j in range(0, len(testL)):
        test_tf.append(np.asarray(func(test[j, :, :])))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    test_norm = min_max_scaler.transform(np.asarray(test_tf))
    lsvm= LinearSVC()
    lsvm.fit(train_norm, trainLabel)
    accuracy = round(100*lsvm.score(test_norm, testL),2)
    return np.asarray(train_tf), np.asarray(test_tf), trainLabel, testL, accuracy




