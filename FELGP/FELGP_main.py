#python packages
import random
import time
import gp_restrict as gp_restrict
import algo_iegp as evalGP
import numpy as np
from deap import base, creator, tools, gp
import felgp_functions as felgp_fs
from strongGPDataType import Int1, Int2, Int3, Int4, Int5, Int6
from strongGPDataType import Float1, Float2, Float3
from strongGPDataType import Array1, Array2, Array3, Array4, Array5
# defined by author

randomSeeds = 12
dataSetName = 'f1'

x_train = np.load(dataSetName+'_train_data.npy')/255.0
y_train = np.load(dataSetName+'_train_label.npy')
x_test = np.load(dataSetName+'_test_data.npy')/255.0
y_test = np.load(dataSetName+'_test_label.npy')

print(x_train.shape,y_train.shape, x_test.shape,y_test.shape)

#parameters:
num_train = x_train.shape[0]
pop_size=2
generation=5
cxProb=0.8
mutProb=0.19
elitismProb=0.01
totalRuns = 1
initialMinDepth=2
initialMaxDepth=8
maxDepth=8

##GP
pset = gp.PrimitiveSetTyped('MAIN', [Array1, Array2], Array5, prefix = 'Image')
# Combination, use 'Combine' for increasing the depth of the GP tree
pset.addPrimitive(felgp_fs.combine, [Array5, Array5, Array5], Array5, name='Combine')
pset.addPrimitive(felgp_fs.combine, [Array4, Array4, Array4], Array5, name='Combine3')
pset.addPrimitive(felgp_fs.combine, [Array4, Array4, Array4, Array4, Array5], Array5, name='Combine5')
pset.addPrimitive(felgp_fs.combine, [Array4, Array4, Array4, Array4, Array4, Array4, Array4], Array5, name='Combine7')
#Classification
pset.addPrimitive(felgp_fs.linear_svm, [Array3, Array2, Int4], Array4, name='SVM')
pset.addPrimitive(felgp_fs.lr, [Array3, Array2, Int4], Array4, name='LR')
pset.addPrimitive(felgp_fs.randomforest, [Array3, Array2, Int5, Int6], Array4, name='RF')
pset.addPrimitive(felgp_fs.erandomforest, [Array3, Array2, Int5, Int6], Array4, name='ERF')
###Feature Concatenation
pset.addPrimitive(felgp_fs.FeaCon2, [Array3, Array3], Array3, name ='FeaCon2')
pset.addPrimitive(felgp_fs.FeaCon3, [Array3, Array3, Array3], Array3, name ='FeaCon3')
pset.addPrimitive(felgp_fs.FeaCon4, [Array3, Array3, Array3, Array3], Array3, name ='FeaCon4')
#Feature Extraction
pset.addPrimitive(felgp_fs.global_hog_small, [Array1], Array3, name = 'F_HOG')
pset.addPrimitive(felgp_fs.all_lbp, [Array1], Array3, name = 'F_uLBP')
pset.addPrimitive(felgp_fs.all_sift, [Array1], Array3, name = 'F_SIFT')
##Filtering and Pooling
pset.addPrimitive(felgp_fs.maxP, [Array1, Int3, Int3], Array1,name='MaxP')
pset.addPrimitive(felgp_fs.gau, [Array1, Int1], Array1, name='Gau')
pset.addPrimitive(felgp_fs.gauD, [Array1, Int1, Int2, Int2], Array1, name='GauD')
pset.addPrimitive(felgp_fs.gab, [Array1, Float1, Float2], Array1, name='Gabor')
pset.addPrimitive(felgp_fs.laplace, [Array1], Array1, name='Lap')
pset.addPrimitive(felgp_fs.gaussian_Laplace1, [Array1], Array1, name='LoG1')
pset.addPrimitive(felgp_fs.gaussian_Laplace2, [Array1], Array1, name='LoG2')
pset.addPrimitive(felgp_fs.sobelxy, [Array1], Array1, name='Sobel')
pset.addPrimitive(felgp_fs.sobelx, [Array1], Array1, name='SobelX')
pset.addPrimitive(felgp_fs.sobely, [Array1], Array1, name='SobelY')
pset.addPrimitive(felgp_fs.lbp, [Array1], Array1, name='LBP')
pset.addPrimitive(felgp_fs.hog_feature, [Array1], Array1, name='HoG')
pset.addPrimitive(felgp_fs.medianf, [Array1], Array1,name='Med')
pset.addPrimitive(felgp_fs.maxf, [Array1], Array1,name='Max')
pset.addPrimitive(felgp_fs.minf, [Array1], Array1,name='Min')
pset.addPrimitive(felgp_fs.meanf, [Array1], Array1,name='Mean')
pset.addPrimitive(felgp_fs.sqrt, [Array1], Array1, name='Sqrt')
pset.addPrimitive(felgp_fs.mixconadd, [Array1, Float3, Array1, Float3], Array1, name='W-Add')
pset.addPrimitive(felgp_fs.mixconsub, [Array1, Float3, Array1, Float3], Array1, name='W-Sub')
pset.addPrimitive(felgp_fs.relu, [Array1], Array1, name='Relu')
#Terminals
pset.renameArguments(ARG0='grey')
pset.addEphemeralConstant('Singma', lambda: random.randint(1, 4), Int1)
pset.addEphemeralConstant('Order', lambda: random.randint(0, 3), Int2)
pset.addEphemeralConstant('Theta', lambda: random.randint(0, 8), Float1)
pset.addEphemeralConstant('Frequency', lambda: random.randint(0, 5), Float2)
pset.addEphemeralConstant('n', lambda: round(random.random(), 3), Float3)
pset.addEphemeralConstant('KernelSize', lambda: random.randrange(2, 5, 2), Int3)
pset.addEphemeralConstant('C', lambda: random.randint(-2, 5), Int4)
pset.addEphemeralConstant('num_Tree', lambda: random.randrange(50, 501, 10), Int5)
pset.addEphemeralConstant('tree_Depth', lambda: random.randrange(10, 101, 10), Int6)
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
        output = np.asarray(func(x_train, y_train))
        y_predict  = np.argmax(output, axis=1)
        accuracy = 100*np.sum(y_predict == y_train) / len(y_train)
    except:
        accuracy = 0
    return accuracy,

toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament,tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
#toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
#toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

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

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,randomSeeds,
                    stats=mstats, halloffame=hof, verbose=True)

    return pop,log, hof

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

    testResults = evalTest(toolbox, hof[0], x_train, y_train,x_test, y_test)

    testTime = time.process_time() - endTime
    print('testResults ', testResults)

