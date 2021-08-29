import random
from deap import tools
from collections import defaultdict


def pop_compare(ind1, ind2):
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    for idx, node in enumerate(ind1[1:],1):
        types1[node.ret].append(idx)
    for idx, node in enumerate(ind2[1:],1):
        types2[node.ret].append(idx)
    return types1==types2

def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param elitpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* beceause of its propention to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematicaly, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb=cxpb/(cxpb+mutpb)

    #num_cx=int(new_cxpb*len(offspring))
    #num_mu=len(offspring)-num_cx
    #print(new_cxpb, new_mutpb)
    # Apply crossover and mutation on the offspring
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            if (offspring[i - 1] == offspring[i]) or pop_compare(offspring[i - 1], offspring[i]):
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


def eaSimple(population, toolbox, cxpb, mutpb, elitpb, ngen, randomseed, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param etilpb: The probability of elitism
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            elitismNum
            offspringE=selectElitism(population,elitismNum)
            population = select(population, len(population)-elitismNum)
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            offspring=offspring+offspringE
            evaluate(offspring)
            population = offspring.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` and :meth::`toolbox.selectElitism`,
     aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    # Evaluate the individuals with an invalid fitness
    # invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # print(len(invalid_ind))

    for i in population:
        i.fitness.values = toolbox.evaluate(individual=i, hof=[])

    if halloffame is not None:
        halloffame.update(population)
    hof_store = tools.HallOfFame(5 * len(population))
    hof_store.update(population)
    cop_po = population
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):

        # Select the next generation individuals by elitism
        elitismNum = int(elitpb * len(population))
        population_for_eli = [toolbox.clone(ind) for ind in population]
        offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)

        # Select the next generation individuals for crossover and mutation
        offspring = toolbox.select(population, len(population) - elitismNum)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        # add offspring from elitism into current offspring
        # generate the next generation individuals

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # print(len(invalid_ind))
        for i in invalid_ind:
            i.fitness.values = toolbox.evaluate(individual=i, hof=cop_po)

        offspring[0:0] = offspringE

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
        # print(record)
        logbook.record(gen=gen, nevals=len(offspring), **record)
        # print(record)
        if verbose:
            print(logbook.stream)
    return population, logbook

