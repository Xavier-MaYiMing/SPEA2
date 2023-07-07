#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/18 09:44
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : SPEA2.py
# @Statement : Strength Pareto Evolutionary Algorithm 2 (SPEA2)
# @Reference : Zitzler E, Laumanns M, Thiele L. SPEA2: Improving the strength Pareto evolutionary algorithm[J]. TIK-Report, 2001, 103.
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def cal_obj(x):
    # ZDT3
    if np.any(x < 0) or np.any(x > 1):
        return [np.inf, np.inf]
    f1 = x[0]
    num1 = 0
    for i in range(1, len(x)):
        num1 += x[i]
    g = 1 + 9 * num1 / (len(x) - 1)
    f2 = g * (1 - np.sqrt(x[0] / g) - x[0] / g * np.sin(10 * np.pi * x[0]))
    return [f1, f2]


def selection(pop, F, pc, k=2):
    # tournament selection
    (npop, dim) = pop.shape
    nm = int(npop * pc)
    nm = nm if nm % 2 == 0 else nm + 1
    mating_pool = np.zeros((nm, dim))
    for i in range(nm):
        selections = np.random.choice(npop, k, replace=True)
        ind = selections[np.argmin(F[selections])]
        mating_pool[i] = pop[ind]
    return mating_pool


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, dim) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, dim))
    mu = np.random.random((nm, dim))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, eta_m):
    # polynomial mutation
    (npop, dim) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, dim)) < 1 / dim
    mu = np.random.random((npop, dim))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def fitness(objs, K):
    # calculate the fitness of each individual
    npop = objs.shape[0]
    S = np.zeros(npop, dtype=int)  # the strength value
    R = np.zeros(npop, dtype=int)  # the raw fitness
    dom = np.full((npop, npop), False)  # domination matrix
    for i in range(npop - 1):
        for j in range(i, npop):
            if dominates(objs[i], objs[j]):
                S[i] += 1
                dom[i, j] = True
            elif dominates(objs[j], objs[i]):
                S[j] += 1
                dom[j, i] = True
    for i in range(npop):
        R[i] = np.sum(S[dom[:, i]])
    sigma = squareform(pdist(objs, metric='seuclidean'), force='no', checks=True)
    sigma_K = np.sort(sigma)[:, K]  # the K-th shortest distance
    D = 1 / (sigma_K + 2)  # density
    F = R + D  # fitness
    return sigma, F


def main(npop, narch, iter, lb, ub, pc=0.8, eta_c=20, eta_m=20):
    """
    The main function
    :param npop: population size
    :param narch: archive size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param pc: crossover probability (default = 0.8)
    :param eta_c: spread factor distribution index (default = 20)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    t = 0   # iterator
    dim = len(lb)  # dimension
    pop = np.random.uniform(lb, ub, (npop, dim))  # population
    objs = np.array([cal_obj(pop[i]) for i in range(npop)])  # the objectives of population
    arch = np.random.uniform(lb, ub, (narch, dim))  # archive
    arch_objs = np.array([cal_obj(arch[i]) for i in range(narch)])  # the objectives of archive
    nall = npop + narch  # the total number of individuals
    K = round(np.sqrt(nall))  # the K-th nearest neighbor

    # Step 2. The main loop
    while True:

        if (t + 1) % 20 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        # Step 2.1. Fitness assignment
        all_pop = np.concatenate((pop, arch), axis=0)
        all_objs = np.concatenate((objs, arch_objs), axis=0)
        sigma, F = fitness(all_objs, K)
        index = np.where(F <= 1)[0]

        # Step 2.2. Environmental selection
        if len(index) <= narch:
            rank = np.argsort(F)
            arch = all_pop[rank[: narch]]
            arch_objs = all_objs[rank[: narch]]
            arch_F = F[rank[: narch]]
        else:
            arch = all_pop[index]
            arch_objs = all_objs[index]
            sigma = sigma[index][:, index]
            eye = np.arange(len(sigma))
            sigma[eye, eye] = np.inf
            arch_F = F[index]

            # the original truncation
            delete = np.full(len(index), False)
            while np.sum(delete) < len(index) - narch:
                remain = np.where(~delete)[0]
                temp = np.sort(sigma[remain][:, remain])
                delete[remain[np.argmin(temp[:, 0])]] = True
            remain = np.where(~delete)[0]
            arch = arch[remain]
            arch_objs = arch_objs[remain]
            arch_F = arch_F[remain]

            # an improved truncation method
            # k = 0
            # while len(arch) > narch:
            #     while k < nall - 1 and np.min(sigma[:, k]) == np.max(sigma[:, k]):
            #         k += 1
            #     ind = np.argmin(sigma[:, k])
            #     arch = np.delete(arch, ind, axis=0)
            #     arch_objs = np.delete(arch_objs, ind, axis=0)
            #     sigma = np.delete(sigma, ind, axis=0)
            #     arch_F = np.delete(arch_F, ind)

        # Step 2.3. Termination
        if t == iter:
            break
        t += 1

        # Step 2.4. Selection + crossover + mutation
        mating_pool = selection(arch, arch_F, pc)
        offspring1 = crossover(mating_pool, lb, ub, eta_c)
        mutants = selection(arch, arch_F, 1 - pc)
        offspring2 = mutation(mutants, lb, ub, eta_m)
        pop = np.concatenate((offspring1, offspring2), axis=0)
        objs = np.array([cal_obj(pop[i]) for i in range(npop)])

    # Step 3. Sort the results
    pf = arch_objs[np.where(arch_F <= 1)]
    plt.figure()
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    plt.scatter(x, y)
    plt.xlabel('objective 1')
    plt.ylabel('objective 2')
    plt.title('The Pareto front of ZDT3')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(100, 100, 300, np.array([0] * 10), np.array([1] * 10))
