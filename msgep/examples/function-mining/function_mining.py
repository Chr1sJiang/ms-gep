from typing import Any, Union

import numpy

from msgep.functions.linking import sum_linker
from msgep.functions.arithmetic import addition, subtraction, multiplication, \
    division, power, cos, sin, sqrt, ln, exp, f, F
from msgep.chromosome import StandardChromosome
import random
from time import time
import math
import matplotlib.pyplot as plt

from msgep.population import StandardPopulation

SAMPLE = []


class DataPoint:
    global SAMPLE
    SAMPLE_SIZE = 20
    RANGE_LOW, RANGE_HIGH = 0,100
    RANGE_SIZE = RANGE_HIGH - RANGE_LOW

    def __init__(self, x):
        self.x = float(x)
        # self.y = float(y)
        # self.x2 = float(x2)
        # self.b = float(b)
        self.z = math.log(x+math.sqrt(x**2+1))

    @staticmethod
    def populate():
        for _ in range(DataPoint.SAMPLE_SIZE):
            x = DataPoint.RANGE_LOW + random.random() * DataPoint.RANGE_SIZE
            # y = DataPoint.RANGE_LOW + random.random() * DataPoint.RANGE_SIZE
            # x2 = DataPoint.RANGE_LOW + (random.random() * DataPoint.RANGE_SIZE)
            # SAMPLE.append(DataPoint(x, y))
            SAMPLE.append(DataPoint(x))
            # print(SAMPLE)
        # print(SAMPLE)


class Regression(StandardChromosome):
    REWARD = 50.0
    tree_functions = addition, multiplication, sqrt,f,F
    tree_terminals = 'x'

    # fitness value
    def _fitness(self):
        total = 0
        for x in SAMPLE:
            try:
                # guess = self(x=x.x, y=x.y)
                guess = self(x=x.x)
                # guess = self(x=x.x)
                diff = min(1.0, abs((guess - x.z) / x.z))
                total += Regression.REWARD * (1 - diff)
            except ZeroDivisionError:
                return 0
        return total

    def _solved(self):
        return self.REWARD * 20 - self.fitness <= 0.000001

    def _rmse(self):
        total = 0
        for x in SAMPLE:
            try:
                # guess = self(x=x.x, y=x.y)
                guess = self(x=x.x)
                # diff = abs((guess - x.z) / x.z)
                diff = (guess - x.z) * (guess - x.z) / 20
                total += diff
            except ZeroDivisionError:
                return 0
        return math.sqrt(total)


if __name__ == "__main__":
    DataPoint.populate()
    # 变异概率
    n_generation = 30
    p_min = 0.01
    p_max = 0.1
    # r = (math.floor(random.random() * 100)) / 100
    G = 1000
    p_final = p_max
    count_change_mut = 0
    count_back = 0
    limit1 = 100
    limit2 = 50
    store = []
    store_p = []
    f_i = 0
    p = StandardPopulation(Regression, n_generation, 3, 6, Regression.tree_functions, Regression.tree_terminals,
                           sum_linker,
                           mutation=p_max, inversion=0.1)
    print(p)

    # if p.best.fitness < 9000:
    #     p.evolve(mutation=p_max)
    #     print(_, p.best, p.best.fitness, p.worst.fitness,p_max)
    # else:
    # print(p_final)
    start = time()
    for _ in range(G):
        if p.best.solved:
            break

        pre_fitness = math.floor(p.best.fitness)
        if limit1 + limit2 >= count_change_mut > limit1 and count_back <= limit2:
            # store_newest = store[len(store) - 1].returnB()
            # store_p_newest = store[len(store) - 1].returnA()
            # fora = p.returnA()
            # st = p.returnB()
            # f_real_avg = numpy.average(fora)
            # f_avg = (p.best.fitness + p.worst.fitness) / 2
            # f_store_b_avg = 0
            # if f_avg < f_real_avg < f_avg + 100:
            #     p.evolve(mutation=0.04)
            #     print(_, p.best, p.best.fitness, p.worst.fitness, 0.04, "random change")
            # elif f_avg + 100 <= f_real_avg:
            #     for i in range(n_generation):
            #         if fora[i] > f_real_avg and fora != p.best.fitness:
            #             rad_tmp = math.floor(random.random() * 29)
            #             st[i] = store_newest[rad_tmp]
            #             fora[i] = store_p_newest[rad_tmp]
            #             f_store_b_avg = numpy.average(fora)
            #         if f_avg < f_store_b_avg < f_avg + 100:
            #             break
            # p.evolve(mutation=0.04)
            p.population = p.newPopulation(store, p, n_generation, _)
            p.evolve(mutation=p_final)
            # print(_, p.best, p.best.fitness, p.worst.fitness, p_final, "random change")
            print(_, p.best.fitness, p.best.rmse)
            # print(p.return_h(p, n_generation))

            # print(_, p.best, p.best.fitness, p.worst.fitness, 0.04, "random change")
            # else:
            #     for i in range(n_generation):
            #         if fora[i] < f_real_avg:
            #             rad_tmp = math.floor(random.random() * 29)
            #             store_b[i] = store_newest[rad_tmp]
            #             fora[i] = store_p_newest[rad_tmp]
            #             f_store_b_avg = numpy.average(fora)
            #         if f_avg < f_real_avg < f_avg + 100:
            #             break
            #         p.evolve(mutation=0.04)
            #         print(_, p.best, p.best.fitness, p.worst.fitness, 0.04, "random change")

            now_fitness = math.floor(p.best.fitness)
            if pre_fitness == now_fitness:
                count_change_mut = count_change_mut + 1
                count_back = count_back + 1
            else:
                count_change_mut = 0
                count_back = 0
                store.append(p)
            p_final = p.return_p_final(p, n_generation, p_min, p_max, _, G)
            # if count_back > limit2:
            #     p_final = p.return_p_final_normal(p_min, p_max, _, G)

        elif count_back > limit2:
            # para_limit2 = math.floor(count_back / limit2)
            # if para_limit2 > 3:
            #     p = store[len(store) - 4]
            #     p.evolve(mutation=p_final)
            #     now_fitness = math.floor(p.best.fitness)
            #     print(_, p.best, p.best.fitness, p.worst.fitness, p_final, "Furthest，local optimum")
            #     if pre_fitness == now_fitness:
            #         count_back = count_back + 1
            #     else:
            #         store.append(p)
            #         count_change_mut = 0
            #         count_back = 0
            # else:
            #     p = store[len(store) - para_limit2 - 1]
            #     p.evolve(mutation=p_final)
            #     now_fitness = math.floor(p.best.fitness)
            #     print(_, p.best, p.best.fitness, p.worst.fitness, p_final, "still")
            #     if pre_fitness == now_fitness:
            #         count_back = count_back + 1
            #     else:
            #         store.append(p)
            #         count_change_mut = 0
            #         count_back = 0
            p.population = p.newPopulationLast(store, p, n_generation, count_back, limit2)
            p.evolve(mutation=0.03)
            # print(_, p.best, p.best.fitness, p.worst.fitness, p_final, "still")
            print(_, p.best.fitness, p.best.rmse)
            # print(p.return_h(p, n_generation))
            # p_final = p.return_p_final(p, n_generation, p_min, p_max, _, G)
            now_fitness = math.floor(p.best.fitness)
            if pre_fitness == now_fitness:
                count_back = count_back + 1
            else:
                store.append(p)
                count_change_mut = 0
                count_back = 0
            p_final = p.return_p_final(p, n_generation, p_min, p_max, _, G)

        else:
            p.evolve(mutation=p_final)
            now_fitness = math.floor(p.best.fitness)
            # print(_, p.best, p.best.fitness, p.worst.fitness, p_final, "none")
            # print(p.return_h(p, n_generation))
            print(_, p.best.fitness, p.best.rmse)
            if pre_fitness == now_fitness:
                count_change_mut = count_change_mut + 1
            else:
                count_change_mut = 0
                store.append(p)
            p_final = p.return_p_final(p, n_generation, p_min, p_max, _, G)
            # if count_change_mut == limit1:
            #     p_final = p.return_p_final(p, n_generation, p_min, p_max, _, G)

end = time()

if p.best.solved:
    print()
    print('SOLVED:', p.best)
# plt.plot(result)
# plt.show()
print('Took %.3fms per cycle' % (1000 * (end - start) / p.generation))
