# python packages
import random
import time
import operator
import algo_iegp as evalGP
import sys
import gp_restrict as gp_restrict
import numpy as np
# deap package
from deap import base, creator, tools, gp
import saveFile
import os
##image Data
from stronglGPData2 import ndarray, numInts, indexType2, testData, region
from stronglGPData2 import kernelSize, histdata, filterData, coordsX1, coordsX2, trainData
from stronglGPData2 import numClass, windowSize2, windowSize3, poolingType, imageDa, trainLabel
from stronglGPData2 import numDepth, numTree, cpanaety, learningrate
import felgp_functions as felgp_fs

# randomSeeds = 12
# dataSetName = 'f1'
randomSeeds=int(sys.argv[2])
dataSetName=str(sys.argv[1])

x_train = np.load('/nfs/scratch/biyi/iegp_code/'+dataSetName+'_train_data.npy')/255.0
y_train = np.load('/nfs/scratch/biyi/iegp_code/'+dataSetName+'_train_label.npy')
x_test = np.load('/nfs/scratch/biyi/iegp_code/'+dataSetName+'_test_data.npy')/255.0
y_test = np.load('/nfs/scratch/biyi/iegp_code/'+dataSetName+'_test_label.npy')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# parameters:
num_train = x_train.shape[0]
pop_size = 100
generation = 50
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 8
maxDepth = 8
##GP
pset = gp.PrimitiveSetTyped('MAIN', [filterData, trainLabel], trainData, prefix='Image')
##imageDa
pset.addPrimitive(felgp_fs.combine, [trainData, trainData, trainData], trainData, name='Combine23')
pset.addPrimitive(felgp_fs.combine, [imageDa, imageDa, imageDa], trainData, name='Combine3')
pset.addPrimitive(felgp_fs.combine, [imageDa, imageDa, imageDa, imageDa, imageDa], trainData, name='Combine5')
pset.addPrimitive(felgp_fs.combine, [imageDa, imageDa, imageDa, imageDa, imageDa, imageDa, imageDa], trainData,
                  name='Combine7')
# learn classifiers
pset.addPrimitive(felgp_fs.linear_svm, [histdata, trainLabel, cpanaety], imageDa, name='SVM_linear')
pset.addPrimitive(felgp_fs.lr, [histdata, trainLabel, cpanaety], imageDa, name='LR')
pset.addPrimitive(felgp_fs.randomforest, [histdata, trainLabel, numTree, numDepth], imageDa, name='RF')
pset.addPrimitive(felgp_fs.erandomforest, [histdata, trainLabel, numTree, numDepth], imageDa, name='ERF')
###learned features
pset.addPrimitive(felgp_fs.FeaCon2, [histdata, histdata], histdata, name='Root1')
pset.addPrimitive(felgp_fs.FeaCon2, [region, region], histdata, name='Root2')
pset.addPrimitive(felgp_fs.FeaCon3, [region, region, region], histdata, name='Root3')
pset.addPrimitive(felgp_fs.FeaCon4, [region, region, region, region], histdata, name='Root4')
##with other features
pset.addPrimitive(felgp_fs.global_hog_small, [filterData], region, name='F_HOG')
pset.addPrimitive(felgp_fs.all_lbp, [filterData], region, name='F_uLBP')
pset.addPrimitive(felgp_fs.all_sift, [filterData], region, name='F_SIFT')
##with other features
# pooling
pset.addPrimitive(felgp_fs.maxP, [filterData, kernelSize, kernelSize], filterData, name='MaxP1')
# aggregation
pset.addPrimitive(felgp_fs.mixconadd, [filterData, float, filterData, float], filterData, name='Mix_ConAdd')
pset.addPrimitive(felgp_fs.mixconsub, [filterData, float, filterData, float], filterData, name='Mix_ConSub')
pset.addPrimitive(felgp_fs.sqrt, [filterData], filterData, name='Sqrt')
pset.addPrimitive(felgp_fs.relu, [filterData], filterData, name='Relu')
# edge features
pset.addPrimitive(felgp_fs.sobelxy, [filterData], filterData, name='Sobel_XY')
pset.addPrimitive(felgp_fs.sobelx, [filterData], filterData, name='Sobel_X')
pset.addPrimitive(felgp_fs.sobely, [filterData], filterData, name='Sobel_Y')
# Gabor
pset.addPrimitive(felgp_fs.gab, [filterData, windowSize2, windowSize3], filterData, name='Gabor2')
pset.addPrimitive(felgp_fs.gaussian_Laplace1, [filterData], filterData, name='LoG1')
pset.addPrimitive(felgp_fs.gaussian_Laplace2, [filterData], filterData, name='LoG2')
pset.addPrimitive(felgp_fs.laplace, [filterData], filterData, name='Lap')
pset.addPrimitive(felgp_fs.lbp, [filterData], filterData, name='LBP')
pset.addPrimitive(felgp_fs.hog_feature, [filterData], filterData, name='HoG')
# Gaussian features
pset.addPrimitive(felgp_fs.gau, [filterData, coordsX2], filterData, name='Gau2')
pset.addPrimitive(felgp_fs.gauD, [filterData, coordsX2, coordsX1, coordsX1], filterData, name='Gau_D2')
# general filters
pset.addPrimitive(felgp_fs.medianf, [filterData], filterData, name='Med')
pset.addPrimitive(felgp_fs.maxf, [filterData], filterData, name='Max')
pset.addPrimitive(felgp_fs.minf, [filterData], filterData, name='Min')
pset.addPrimitive(felgp_fs.meanf, [filterData], filterData, name='Mean')
# Terminals
pset.renameArguments(ARG0='grey')
pset.addEphemeralConstant('randomD', lambda: round(random.random(), 3), float)
pset.addEphemeralConstant('kernelSize', lambda: random.randrange(2, 4, 2), kernelSize)
pset.addEphemeralConstant('Theta', lambda: random.randint(0, 8), windowSize2)
pset.addEphemeralConstant('Frequency', lambda: random.randint(0, 5), windowSize3)
pset.addEphemeralConstant('Singma', lambda: random.randint(1, 4), coordsX2)
pset.addEphemeralConstant('Order', lambda: random.randint(0, 3), coordsX1)
pset.addEphemeralConstant('C', lambda: random.randint(-2, 5), cpanaety)
pset.addEphemeralConstant('num_Tree', lambda: random.randrange(50, 501, 10), numTree)
pset.addEphemeralConstant('tree_Depth', lambda: random.randrange(10, 101, 10), numDepth)
# pset.addEphemeralConstant('learningrate', lambda: round(random.random(),3), float)

##
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalTrain(toolbox, individual, hof, trainData, trainLabel):
    if len(hof) != 0 and individual in hof:
        ind = 0
        while ind < len(hof):
            if individual == hof[ind]:
                accuracy, = hof[ind].fitness.values
                ind = len(hof)
            else: ind+=1
    else:
        try:
            func = toolbox.compile(expr=individual)
            output = np.asarray(func(trainData, trainLabel))
            y_predict  = np.argmax(output, axis=1)
            accuracy = 100*np.sum(y_predict == trainLabel) / len(trainLabel)
        except:
            accuracy=0
    return accuracy,

# genetic operator
toolbox.register("evaluate", evalTrain, toolbox, trainData=x_train, trainLabel=y_train)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))


def GPMain(randomSeeds):
    random.seed(randomSeeds)

    pop = toolbox.population(pop_size)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation, randomSeeds,
                               stats=mstats, halloffame=hof, verbose=True)
    return pop, log, hof


def evalTest(toolbox, individual, trainData, trainLabel, test, testL):
    x_train = np.concatenate((trainData, test), axis=0)
    func = toolbox.compile(expr=individual)
    output = np.asarray(func(x_train, trainLabel))
    print(output.shape)
    y_predict = np.argmax(output, axis=1)
    accuracy = 100*np.sum(y_predict==testL)/len(testL)
    return accuracy

if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    testResults = evalTest(toolbox, hof[0], x_train, y_train, x_test, y_test)

    testTime = time.process_time() - endTime
    print('testResults ', testResults)

    saveFile.saveAllResults(randomSeeds, dataSetName, hof, trainTime, testResults, log)


