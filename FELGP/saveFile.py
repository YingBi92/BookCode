import pickle

from deap import gp


def saveResults(fileName, *args, **kwargs):
    f = open(fileName, 'w')
    for i in args:
        f.writelines(str(i)+'\n')
    f.close()
    return



def saveLog (fileName, log):
   f = open(fileName, 'wb')
   pickle.dump(log, f)
   f.close()
   return
# def saveLog (fileName, log):
#    f=open(fileName, 'wb')
#    pickle.dump(log, f)
#    f.close()
 #   return


# def plotTree(pathName,individual):
#    nodes, edges, labels = gp.graph(individual)
#    g = pgv.AGraph()
#    g.add_nodes_from(nodes)
#    g.add_edges_from(edges)
#    g.layout(prog="dot")

#    for i in nodes:
#        n = g.get_node(i)
#        n.attr["label"] = labels[i]
#    g.draw(pathName)
#    return


def bestInd(toolbox, population, number):
    bestInd = []
    best = toolbox.selectElitism(population, k=number)
    for i in best:
        bestInd.append(i)
    return bestInd
        

def saveAllResults(randomSeeds, dataSetName, hof, trainTime, testResults, log):
    fileName1 = str(randomSeeds) + 'Results_on' + dataSetName + '.txt'
    saveLog(fileName1, log)
    fileName = str(randomSeeds) + 'Final_Result_son' + dataSetName + '.txt'
    saveResults(fileName, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                         'trainResults', hof[0].fitness,
                         'testResults', testResults, 'bestInd in training',
                         hof[0])

    return
