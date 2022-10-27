from .exceptions import InitializationError, ImplementationError
from .selection import RouletteWheelSelection
from .utils import argmax, argmin
import numpy
import random
import math


class StandardPopulation:
    """
    Handles a population of uniform chromosomes...
    """
    default_rates = dict(
        replication=0,
        mutation=0.05,  
        inversion=0.1,
        IS_transposition=0.1,
        RIS_transposition=0.1,
        gene_transposition=0.1,
        one_point_recombination=0,
        two_point_recombination=0,
        gene_recombination=0
    )

    def __init__(self, chromosomecls, population_size, chromosome_genes, genes_head, tree_functions, tree_terminals,
                 linking_function, selection_strategy=RouletteWheelSelection(), chromosomes=None, prefer_functions=0,
                 **kwargs):
        self.current_round = 0
        self.solved = False
        self.population_size = population_size
        self.selection_strategy = selection_strategy
        if chromosomes is not None:
            if len(chromosomes) != population_size:
                raise InitializationError("The given chromosomes' lenght is not the population size")
            self.population = chromosomes
        else:
            self.population = [chromosomecls(gene_number=chromosome_genes, genes_head=genes_head,
                                             tree_functions=tree_functions, tree_terminals=tree_terminals,
                                             linking_function=linking_function, prefer_functions=prefer_functions)
                               for _ in range(population_size)]
        for attr, default in StandardPopulation.default_rates.items():
            self.__dict__[attr] = default if not attr in kwargs else kwargs[attr]

    def evaluate(self):
        """
        评估所有染色体适应度
        """
        if not hasattr(self, "evaluation_round") or self.evaluation_round != self.current_round:
            # PERFORMANCE: 防止多次评估
            global a, b
            a = []
            b = []
            self.evaluation_round = self.current_round
            for chromosome in self.population:
                chromosome.fitness = chromosome._fitness()
                chromosome.rmse = chromosome._rmse()
                chromosome.solved = chromosome._solved()
                a.append(chromosome.fitness)
                b.append(chromosome)

    def newPopulation(self, store, p, n_generation, _):
        store_newest = store[len(store) - 1].returnB()
        store_p_newest = store[len(store) - 1].returnA()
        w = 88
        fora = p.returnA()
        f_real_avg = numpy.average(fora)
        f_avg = (p.best.fitness + p.worst.fitness) / 2
        f_store_b_avg = 0
        if f_avg < f_real_avg < f_avg + w:
            self.population = self.population
        elif f_avg + w <= f_real_avg:
            for i in range(n_generation):
                if fora[i] > f_real_avg and fora[i] != p.best.fitness and fora[i] != p.worst.fitness:
                    rad_tmp = math.floor(random.random() * (n_generation - 1))
                    self.population[i] = store_newest[rad_tmp]
                    fora[i] = store_p_newest[rad_tmp]
                    f_store_b_avg = numpy.average(fora)
                if f_avg < f_store_b_avg < f_avg + w:
                    break
        else:
            for i in range(n_generation):
                if fora[i] < f_real_avg and fora[i] != p.best.fitness:
                    rad_tmp = math.floor(random.random() * (n_generation - 1))
                    self.population[i] = store_newest[rad_tmp]
                    fora[i] = store_p_newest[rad_tmp]
                    f_store_b_avg = numpy.average(fora)
                if f_avg < f_store_b_avg < f_avg + w:
                    break
        return self.population

    # def newPopulation(self, p, n_generation, p_new, n_generation_new):
    #     h_limit = 2
    #     fora = p.returnA()
    #     new = p_new.returnB()
    #     print(p_new)
    #     h = p.return_h(p, n_generation)
    #     if h >= h_limit:
    #         self.population = self.population
    #     else:
    #         for i in range(n_generation):
    #             if fora[i] != p.best.fitness:
    #                 rad_tmp = math.floor(random.random() * (n_generation_new - 1))
    #                 self.population[i] = new[rad_tmp]
    #             h_new = self.return_h(self, n_generation)
    #             if n_generation % 10 == 0:
    #                 if h_new >= h_limit:
    #                     break
    #     return self.population

    def newPopulationLast(self, store, p, n_generation, count_back, limit2):
        para_limit2 = math.floor(count_back / limit2)
        fora = p.returnA()
        if para_limit2 > len(store):
            store_newest = store[0].returnB()
            for i in range(n_generation):
                if fora[i] != p.best.fitness:
                    self.population[i] = store_newest[i]
        else:
            store_newest = store[len(store) - para_limit2].returnB()
            for i in range(n_generation):
                if fora[i] != p.best.fitness:
                    self.population[i] = store_newest[i]
        return self.population

    # def return_p_final(self, p, n_generation, p_min, p_max, _, G):
    #     self.population_size = n_generation
    #     s_max = math.log(1 / n_generation)
    #     r = 0.5
    #     f_minus = (1 + 0.001) * p.best.fitness - (1 - 0.001) * p.worst.fitness
    #     fora = p.returnA()
    #     s = 0
    #     for i_ in range(n_generation):
    #         count3 = 0
    #         area_low = f_minus * i_ / n_generation
    #         area_up = f_minus * (i_ + 1) / n_generation
    #         s_i = 0
    #         for i__ in range(n_generation):
    #             if (1 - 0.001) * p.worst.fitness + area_low < fora[i__] <= (1 - 0.001) * p.worst.fitness + area_up:
    #                 count3 = count3 + 1
    #         p_i = count3 / n_generation
    #         if p_i != 0:
    #             s_i = - p_i * math.log(p_i)
    #         s += s_i
    #     global h
    #     h = s
    #     p_final = p_min + (p_max - p_min) * math.exp(r * s * s_max * _ / G)
    #     return p_final

    def return_p_final(self, p, n_generation, p_min, p_max, _, G):
        self.population_size = n_generation
        r = -0.5
        h = p.return_h(p, n_generation)
        p_final = p_min + (p_max - p_min) * math.exp(r * h * _ / G)
        return p_final


    def return_h(self, p, n_generation):
        f_minus = (1 + 0.001) * p.best.fitness - (1 - 0.001) * p.worst.fitness
        fora = p.returnA()
        s = 0
        for i_ in range(n_generation):
            count3 = 0
            area_low = f_minus * i_ / n_generation
            area_up = f_minus * (i_ + 1) / n_generation
            s_i = 0
            for i__ in range(n_generation):
                if (1 - 0.001) * p.worst.fitness + area_low < fora[i__] <= (1 - 0.001) * p.worst.fitness + area_up:
                    count3 = count3 + 1
            p_i = count3 / n_generation
            if p_i != 0:
                s_i = - p_i * math.log(p_i)
            s += s_i
        return s

    def return_p_final_normal(self, p_min, p_max, _, G):
        r = -0.5
        p_final = p_min + (p_max - p_min) * 2.7 ** (r * _ / G)
        return p_final

    def cycle(self):
        # For retrocompatibility with PyGEP
        self.evolve()

    def returnA(self):
        return a

    def returnB(self):
        return b

    def evolve(self, selection_strategy=None, **kwargs):
        """
        处理一轮种群进化

        This consist of:
        - The evaluation of chomosomes
        - The selection phase
        - The evolution functions
        """
        # 1. 评估所有染色体
        self.evaluate()
        # 2. 选择阶段使用策略
        sstrategy = self.selection_strategy if selection_strategy is None else selection_strategy
        # wheel
        new_population = sstrategy.select(self.population)
        self.population = new_population
        # 3. Applying evolution functions
        for attr, _ in StandardPopulation.default_rates.items():
            # print(attr, _)
            # 变异率
            # rate = self.__dict__[attr] if not hasattr(kwargs, attr) else kwargs[attr]
            rate = _ if not attr in kwargs else kwargs[attr]
            # print(self.__dict__[attr])
            # print(rate)
            # 所有参数概率
            if rate > 0:
                if "action_%s" % attr in self.__class__.__dict__:
                    self.__getattribute__("action_%s" % attr)(rate=rate, rnd=self.current_round)
                    # print(rate)
                    # print(self.current_round)
                    # Perform evolution function with its rate
                else:
                    raise ImplementationError("The evolution function action_%s is not there!" % attr)
        # 种群进化成功
        self.current_round += 1

    @property
    def generation(self):
        return self.current_round

    @property
    def best(self):
        self.evaluate()
        return self.population[argmax(self.population, lambda x: x.fitness)]

    @property
    def worst(self):
        self.evaluate()
        return self.population[argmin(self.population, lambda x: x.fitness)]

    # @property
    # def every(self):
    #     self.evaluate()
    #     a = []
    #     a.append()

    def __repr__(self):
        """
        Returns a representation for the population suitable for printing.
        """
        rep = "[Size: %d | Round: %d | Best fitness: %f]\n" % (
            self.population_size, self.current_round, self.best.fitness)
        for order, chromosome in enumerate(self.population, 1):
            rep += "%d - %s [%d] (%f)\n" % (order, repr(chromosome), len(chromosome), chromosome.fitness)
        return rep

    def __len__(self):
        """
        返回种群中的个体数
        """
        return self.population_size

    def action_mutation(self, rate, rnd):
        """
        individual mutate...
        """
        f = []
        for chromosome in self.population[1:]:
            f.append(chromosome.fitness)
        f_max = max(f)
        f_min = min(f)
        for chromosome in self.population[1:]:
            fit = chromosome.fitness
            rate1 = 0.09 * ((rate - 0.01) / 0.09) ** ((2 * fit) / (f_max + f_min)) + 0.01
            # print(fit, rate1)
            chromosome.mutate(rate1, rnd)

    def action_inversion(self, rate, rnd):
        """
        反转染色体的基因头部...
        """
        for chromosome in self.population[1:]:
            chromosome.inversion(rate, rnd)

    def action_IS_transposition(self, rate, rnd):
        """
        在染色体的任意点转位一个随机序列
        """
        for chromosome in self.population[1:]:
            chromosome.IS_transposition(rate, rnd)

    def action_RIS_transposition(self, rate, rnd):
        """
        转置从根函数开始的随机序列...
        """
        for chromosome in self.population[1:]:
            chromosome.RIS_transposition(rate, rnd)

    def action_gene_transposition(self, rate, rnd):
        """
        Transposes the genes in some chromosomes
        """
        for chromosome in self.population[1:]:
            chromosome.gene_transposition(rate, rnd)
            
    def action_one_point_recombination(self, rate, rnd):
        """
        one_point_recombination in some chromosomes
        """
        for chromosome in self.population[1:]:
            chromosome.one_point_recombination(rate, rnd)
