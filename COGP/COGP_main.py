#python packages
import random
import time
import operator
#for using multiple CPU cores in fitness evaluation
from scoop import futures
import numpy as np
# deap package
import evalGP
import gp_restrict
import gp_tree
from deap import base, creator, tools, gp
# fitness function and function for testing
from FEVal_norm_fast import evalTest, feature_length
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
#COGP data types
from STGPdataType import Img, Img2, Vector, Int, Double, Filter # defined by author
import functionSet as fs

randomSeeds =3
dataSetName = 'f1'

x_train = np.load(dataSetName+'_train_data.npy')
y_train = np.load(dataSetName+'_train_label.npy')
x_test = np.load(dataSetName+'_test_data.npy')
y_test = np.load(dataSetName+'_test_label.npy')
print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)

#COGP parameters:
population=500
generation=50
cxProb=0.5
mutProb=0.49
elitismProb=0.01
totalRuns = 1
initialMinDepth=2
initialMaxDepth=6
maxDepth=8
##COGP tree structure, function set and terminal set
pset = gp_tree.PrimitiveSetTyped('MAIN',[Img], Vector, prefix='Image')
#Functions at Concatenation layer
pset.addPrimitive(fs.root_conVector2,[Vector,Vector],Vector,name='Root1')
pset.addPrimitive(fs.root_conVector2,[Img2, Img2],Vector,name='Root2')
pset.addPrimitive(fs.root_conVector3,[Img2, Img2, Img2],Vector,name='Root3')
pset.addPrimitive(fs.root_conVector4,[Img2, Img2, Img2, Img2],Vector,name='Root4')
#Filtering at a flexible layer. Use *F as the names of the functions to avoid the same names, which is not allowed in DEAP
pset.addPrimitive(fs.ZeromaxP,[Img2, Int, Int],Img2,name='ZMaxPF')
#Filtering at a flexible layer. Use *F as the names of the functions to avoid the same names, which is not allowed in DEAP
pset.addPrimitive(fs.mixconadd, [Img2, Double, Img2, Double], Img2, name='AddF')
pset.addPrimitive(fs.mixconsub, [Img2, Double, Img2, Double], Img2, name='SubF')
pset.addPrimitive(np.abs, [Img2], Img2, name='AbsF')
pset.addPrimitive(fs.sqrt, [Img2], Img2, name='SqrtF')
pset.addPrimitive(fs.relu, [Img2], Img2, name='ReluF')
pset.addPrimitive(fs.conv_filters, [Img2, Filter], Img2, name='ConvF')  #convolution operator
#Pooling functions at the Pooling layer.
pset.addPrimitive(fs.maxP,[Img2, Int, Int], Img2,name='MaxPF') # max-pooling operator
pset.addPrimitive(fs.maxP,[Img, Int, Int], Img2,name='MaxP') #max-pooling operator
#Filteing functions at the Filtering layer
pset.addPrimitive(fs.mixconadd, [Img, Double, Img, Double], Img, name='Add')
pset.addPrimitive(fs.mixconsub, [Img, Double, Img, Double], Img, name='Sub')
pset.addPrimitive(np.abs, [Img], Img, name='Abs')
pset.addPrimitive(fs.sqrt, [Img], Img, name='Sqrt')
pset.addPrimitive(fs.relu, [Img], Img, name='Relu')
pset.addPrimitive(fs.conv_filters, [Img, Filter], Img, name='Conv') #convolution operator
#Terminals
pset.renameArguments(ARG0='grey')  #the input image
pset.addEphemeralConstant('randomD',lambda:round(random.random(),3),Double) # parameters for the Add, Sub, AddF and SubF functions
pset.addEphemeralConstant('filters3',lambda:list(fs.random_filters(3)), Filter)  #3 * 3 filters
pset.addEphemeralConstant('filters5',lambda:list(fs.random_filters(5)), Filter) #5 * 5 filters
pset.addEphemeralConstant('filters7',lambda:list(fs.random_filters(7)), Filter) #7 * 7 filters
pset.addEphemeralConstant('kernelSize',lambda:random.randrange(2,5,2), Int) # kernel size for the max-pooling functions

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mapp", futures.map)

#Fitness evaluation
def evalTrain(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_tf = []
        for i in range(0, len(y_train)):
            train_tf.append(np.asarray(func(x_train[i, :, :])))
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
        lsvm= LinearSVC()
        accuracy = round(100*cross_val_score(lsvm, train_norm, y_train, cv=5).mean(),2)
    except:
        accuracy=0
    return accuracy,

toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament,tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=6)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

def GPMain(randomSeeds):
    random.seed(randomSeeds)
    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit,size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(randomSeeds, pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True)

    return pop,log, hof

if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    train_tf, test_tf, trainLabel, testL, testResults = evalTest(toolbox, hof[0], x_train, y_train, x_test, y_test)
    testTime = time.process_time() - endTime

    print('Best individual ', hof[0])
    print('Test results  ', testResults)
    print('Train time  ', trainTime)
    print('Test time  ', testTime)
    print('Train set shape  ', train_tf.shape)
    print('Test set shape  ', test_tf.shape)
    print('End')