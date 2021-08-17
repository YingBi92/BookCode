#python packages
import random
import time
import operator
import numpy as np
# deap package
import evalGP
import gp_restrict
from deap import base, creator, tools, gp
# fitness function
from fitnessEvaluation import evalAccuracy
from strongGPDataType import Img, Int1, Int2, Int3, Region, Double  # defined by author
import functionSet as fs

dataSetName='uiuc'
randomSeeds=12

x_train=np.load(dataSetName+'_train_data.npy')
y_train=np.load(dataSetName+'_train_label.npy')
x_validation=np.load(dataSetName+'_vali_data.npy')
y_validation=np.load(dataSetName+'_vali_label.npy')
x_test=np.load(dataSetName+'_test_data.npy')
y_test=np.load(dataSetName+'_test_label.npy')
#parameters:
population=100
generation=5
cxProb=0.8
mutProb=0.19
elitismProb=0.01
totalRuns = 1
initialMinDepth=2
initialMaxDepth=6
maxDepth=6
image_width, image_height = x_train[0].shape
##GP
pset = gp.PrimitiveSetTyped('MAIN',[Img], Double, prefix='Raw')
#Functions at the feature constructions tier
pset.addPrimitive(operator.sub, [Double, Double], Double, name='Sub')
# Functions at the feature extraction layer
pset.addPrimitive(np.std, [Region], Double, name='G_Std1')
pset.addPrimitive(np.std, [Region], Double, name='G_Std2')
pset.addPrimitive(np.std, [Region], Double, name='G_Std3')
pset.addPrimitive(fs.hist_equal, [Region], Region, name='Hist_Eq')
pset.addPrimitive(fs.gaussian_1, [Region], Region, name='Gau1')
pset.addPrimitive(fs.gaussian_11, [Region], Region, name='Gau11')
pset.addPrimitive(fs.gauGM, [Region], Region, name='GauXY')
pset.addPrimitive(fs.laplace, [Region], Region, name='Lap')
pset.addPrimitive(fs.sobelx, [Region], Region, name='Sobel_X')
pset.addPrimitive(fs.sobely, [Region], Region, name='Sobel_Y')
pset.addPrimitive(fs.gaussian_Laplace1, [Region], Region, name='LoG1')
pset.addPrimitive(fs.gaussian_Laplace2, [Region], Region, name='LoG2')
pset.addPrimitive(fs.lbp, [Region], Region, name='LBP')
pset.addPrimitive(fs.hog_feature, [Region], Region, name='HOG')
# Functions  at the region detection layer
pset.addPrimitive(fs.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
pset.addPrimitive(fs.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S1')
pset.addPrimitive(fs.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S2')
pset.addPrimitive(fs.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')
pset.addPrimitive(fs.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R1')
pset.addPrimitive(fs.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R2')
# Terminals
pset.renameArguments(ARG0='grey')
pset.addEphemeralConstant('randomDouble', lambda: round(random.random(), 2), float)
pset.addEphemeralConstant('X', lambda: random.randint(0, image_width-20), Int1)
pset.addEphemeralConstant('Y', lambda: random.randint(0, image_height-20), Int2)
pset.addEphemeralConstant('Size', lambda: random.randint(20, 70), Int3)
##
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evalAccuracy,toolbox,x_train=x_train,y_train=y_train)
toolbox.register("validation", evalAccuracy,toolbox,x_train=x_validation,y_train=y_validation)
toolbox.register("select", tools.selTournament,tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

def GPMain(randomSeeds):

    random.seed(randomSeeds)
    pop = toolbox.population(population)
    hof = tools.HallOfFame(1)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit,size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log, hof2= evalGP.eaSimple(pop, toolbox, cxProb, mutProb,elitismProb, generation,
                    stats=mstats, halloffame=hof, verbose=True)

    return pop,log, hof,hof2

if __name__ == "__main__":

    beginTime = time.process_time()
    pop, log, hof, hof2 = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    testResults = evalAccuracy(toolbox, hof2[0], x_test, y_test)
    testTime = time.process_time() - endTime

    print('Best individual ', hof[0])
    print('Test results  ', testResults)
    print('Train time  ', trainTime)
    print('Test time  ', testTime)
    print('End')
