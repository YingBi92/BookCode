# python packages
import random
import time
import evalGP_fgp as evalGP
import gp_restrict as gp_restrict
import numpy
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Float1, Float2, Float3, Img, Img1, Vector, Vector1
import fgp_functions as fe_fs
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import saveFile
import sys

# randomSeeds = 12
# dataSetName = 'f1'
dataSetName = str(sys.argv[1])
randomSeeds = str(sys.argv[2])

x_train = numpy.load('/nesi/nobackup/nesi00416/iegp_code/FlexGP/'+dataSetName + '_train_data.npy') / 255.0
y_train = numpy.load('/nesi/nobackup/nesi00416/iegp_code/FlexGP/'+dataSetName + '_train_label.npy')
x_test = numpy.load('/nesi/nobackup/nesi00416/iegp_code/FlexGP/'+dataSetName + '_test_data.npy') / 255.0
y_test = numpy.load('/nesi/nobackup/nesi00416/iegp_code/FlexGP/'+dataSetName + '_test_label.npy')

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# parameters:
population = 500
generation = 50
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 8
##GP
pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector1, prefix='Image')
#feature concatenation
pset.addPrimitive(fe_fs.root_con, [Vector1, Vector1], Vector1, name='Root')
pset.addPrimitive(fe_fs.root_conVector2, [Img1, Img1], Vector1, name='Root2')
pset.addPrimitive(fe_fs.root_conVector3, [Img1, Img1, Img1], Vector1, name='Root3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector1, name='Roots2')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector1, name='Roots3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector, Vector], Vector1, name='Roots4')
##feature extraction
pset.addPrimitive(fe_fs.global_hog_small, [Img1], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img1], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img1], Vector, name='Global_SIFT')
pset.addPrimitive(fe_fs.global_hog_small, [Img], Vector, name='FGlobal_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='FGlobal_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='FGlobal_SIFT')
# pooling
pset.addPrimitive(fe_fs.maxP, [Img1, Int3, Int3], Img1, name='MaxPF')
#filtering
pset.addPrimitive(fe_fs.gau, [Img1, Int1], Img1, name='GauF')
pset.addPrimitive(fe_fs.gauD, [Img1, Int1, Int2, Int2], Img1, name='GauDF')
pset.addPrimitive(fe_fs.gab, [Img1, Float1, Float2], Img1, name='GaborF')
pset.addPrimitive(fe_fs.laplace, [Img1], Img1, name='LapF')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img1], Img1, name='LoG1F')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img1], Img1, name='LoG2F')
pset.addPrimitive(fe_fs.sobelxy, [Img1], Img1, name='SobelF')
pset.addPrimitive(fe_fs.sobelx, [Img1], Img1, name='SobelXF')
pset.addPrimitive(fe_fs.sobely, [Img1], Img1, name='SobelYF')
pset.addPrimitive(fe_fs.medianf, [Img1], Img1, name='MedF')
pset.addPrimitive(fe_fs.meanf, [Img1], Img1, name='MeanF')
pset.addPrimitive(fe_fs.minf, [Img1], Img1, name='MinF')
pset.addPrimitive(fe_fs.maxf, [Img1], Img1, name='MaxF')
pset.addPrimitive(fe_fs.lbp, [Img1], Img1, name='LBPF')
pset.addPrimitive(fe_fs.hog_feature, [Img1], Img1, name='HoGF')
pset.addPrimitive(fe_fs.mixconadd, [Img1, Float3, Img1, Float3], Img1, name='W_AddF')
pset.addPrimitive(fe_fs.mixconsub, [Img1, Float3, Img1, Float3], Img1, name='W_SubF')
pset.addPrimitive(fe_fs.sqrt, [Img1], Img1, name='SqrtF')
pset.addPrimitive(fe_fs.relu, [Img1], Img1, name='ReLUF')
# pooling
pset.addPrimitive(fe_fs.maxP, [Img, Int3, Int3], Img1, name='MaxP')
# filtering
pset.addPrimitive(fe_fs.gau, [Img, Int1], Img, name='Gau')
pset.addPrimitive(fe_fs.gauD, [Img, Int1, Int2, Int2], Img, name='GauD')
pset.addPrimitive(fe_fs.gab, [Img, Float1, Float2], Img, name='Gabor')
pset.addPrimitive(fe_fs.laplace, [Img], Img, name='Lap')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img], Img, name='LoG1')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img], Img, name='LoG2')
pset.addPrimitive(fe_fs.sobelxy, [Img], Img, name='Sobel')
pset.addPrimitive(fe_fs.sobelx, [Img], Img, name='SobelX')
pset.addPrimitive(fe_fs.sobely, [Img], Img, name='SobelY')
pset.addPrimitive(fe_fs.medianf, [Img], Img, name='Med')
pset.addPrimitive(fe_fs.meanf, [Img], Img, name='Mean')
pset.addPrimitive(fe_fs.minf, [Img], Img, name='Min')
pset.addPrimitive(fe_fs.maxf, [Img], Img, name='Max')
pset.addPrimitive(fe_fs.lbp, [Img], Img, name='LBP_F')
pset.addPrimitive(fe_fs.hog_feature, [Img], Img, name='HOG_F')
pset.addPrimitive(fe_fs.mixconadd, [Img, Float3, Img, Float3], Img, name='W_Add')
pset.addPrimitive(fe_fs.mixconsub, [Img, Float3, Img, Float3], Img, name='W_Sub')
pset.addPrimitive(fe_fs.sqrt, [Img], Img, name='Sqrt')
pset.addPrimitive(fe_fs.relu, [Img], Img, name='ReLU')
# Terminals
pset.renameArguments(ARG0='Image')
pset.addEphemeralConstant('Singma', lambda: random.randint(1, 4), Int1)
pset.addEphemeralConstant('Order', lambda: random.randint(0, 3), Int2)
pset.addEphemeralConstant('Theta', lambda: random.randint(0, 8), Float1)
pset.addEphemeralConstant('Frequency', lambda: random.randint(0, 5), Float2)
pset.addEphemeralConstant('n', lambda: round(random.random(), 3), Float3)
pset.addEphemeralConstant('KernelSize', lambda: random.randrange(2, 5, 2), Int3)

##
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mapp", map)

def evalTrain(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_tf = []
        for i in range(0, len(y_train)):
            train_tf.append(numpy.asarray(func(x_train[i, :, :])))
        train_tf = numpy.asarray(train_tf, dtype=float)
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(train_tf)
        lsvm = LinearSVC()
        accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=5).mean(), 2)
    except:
        accuracy = 0
    return accuracy,

def evalTrainb(individual):
    print(individual)
    func = toolbox.compile(expr=individual)
    train_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(numpy.asarray(func(x_train[i, :, :])))
    train_tf = numpy.asarray(train_tf, dtype=float)
    print(train_tf.shape)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    lsvm = LinearSVC()
    accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=5).mean(), 2)
    return accuracy,

# genetic operator
toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
# toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
# toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

def GPMain(randomSeeds):
    random.seed(randomSeeds)

    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True)

    return pop, log, hof


def evalTest(toolbox, individual, trainData, trainLabel, test, testL):
    func = toolbox.compile(expr=individual)
    train_tf = []
    test_tf = []
    for i in range(0, len(trainLabel)):
        train_tf.append(numpy.asarray(func(trainData[i, :, :])))
    for j in range(0, len(testL)):
        test_tf.append(numpy.asarray(func(test[j, :, :])))
    train_tf = numpy.asarray(train_tf, dtype=float)
    test_tf = numpy.asarray(test_tf, dtype=float)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)
    lsvm= LinearSVC()
    lsvm.fit(train_norm, trainLabel)
    accuracy = round(100*lsvm.score(test_norm, testL),2)
    return numpy.asarray(train_tf), numpy.asarray(test_tf), trainLabel, testL, accuracy


if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    train_tf, test_tf, trainLabel, testL, testResults = evalTest(toolbox, hof[0], x_train, y_train, x_test, y_test)
    testTime = time.process_time() - endTime
    saveFile.saveAllResults(randomSeeds, dataSetName, hof, trainTime, testResults, log)

    print(testResults)
    print(train_tf.shape, test_tf.shape)
    print(hof[0])
    print('End')
