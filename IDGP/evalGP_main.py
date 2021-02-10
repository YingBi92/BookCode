import random
from deap import tools

def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb=cxpb/(cxpb+mutpb)
    # Apply crossover and mutation on the offspring
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            if (offspring[i - 1] == offspring[i]):
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        else:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, elitpb, ngen , stats=None,
             halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    fitnesses = toolbox.mapp(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    hof_store = tools.HallOfFame(5 * len(population))
    hof_store.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)
    for gen in range(1, ngen + 1):
        #Select the next generation individuals by elitism
        elitismNum=int(elitpb * len(population))
        population_for_eli=[toolbox.clone(ind) for ind in population]
        offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)
        offspring = toolbox.select(population, len(population)-elitismNum)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        for i in offspring:
            ind = 0
            while ind<len(hof_store):
                if i == hof_store[ind]:
                    i.fitness.values = hof_store[ind].fitness.values
                    ind = len(hof_store)
                else:
                    ind+=1

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.mapp(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring[0:0]=offspringE
            
        # Update the hall of fame with the generated
        if halloffame is not None:
            halloffame.update(offspring)
        cop_po = offspring.copy()
        hof_store.update(offspring)
        for i in hof_store:
            cop_po.append(i)
        population[:] = offspring
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)
        if verbose:
            print(logbook.stream)
    return population, logbook
