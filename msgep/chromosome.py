from random import random
from math import floor
from .genes import Gene
from .exceptions import ImplementationError
from copy import deepcopy


class StandardChromosome:
    def __init__(self, gene_number, genes_head, tree_functions=None, tree_terminals=None, linking_function=None,
                 prefer_functions=0):
        self._cantchange = False
        self.modified_round = 0
        self.gene_number = gene_number
        self.genes_head = genes_head
        self.prefer_functions = prefer_functions
        self.linking_function = linking_function if linking_function is not None else self.__class__.linking_function
        self.tree_functions = tree_functions if tree_functions is not None else self.__class__.tree_functions
        self.tree_terminals = tree_terminals if tree_terminals is not None else self.__class__.tree_terminals
        self.genes = [Gene(head_length=genes_head, tree_functions=tree_functions, tree_terminals=tree_terminals,
                           prefer_functions=prefer_functions)
                      for _ in range(gene_number)]
        for gene in self.genes:
            gene.initialize()

    def _fitness(self):
        """
        用户实现的函数，应该返回染色体的适应值
        """
        raise ImplementationError("You should provide a _fitness function for the chromosomes")

    def _rmse(self):
        """
        用户实现的函数，应该返回染色体的适应值
        """
        raise ImplementationError("You should provide a _fitness function for the chromosomes")

    def _solved(self):
        """
        如果没有子类化，它总是返回False。
        如果返回True，则搜索将停止，因为找到了最佳染色体
        """
        return False

    def __call__(self, dictionary=None, **kwargs):
        if dictionary is None:
            dictionary = kwargs
        args = map(lambda g: g(dictionary), self.genes)
        return self.linking_function(*args)

    def __len__(self):
        """
        以树的形式返回实际染色体长度

        It also counts the linking function at the start.
        """
        return 1 + sum([len(gene) for gene in self.genes])

    def __repr__(self):
        """
        返回染色体的表现形式
        """
        return "|".join([repr(gene) for gene in self.genes])

    def mutate(self, rate, rnd):
        for gene in self.genes:
            if gene.mutate(rate):
                self.modified_round = rnd

    def inversion(self, rate, rnd):
        if random() <= rate:
            gene_to_invert = floor(self.gene_number * random())
            self.genes[gene_to_invert].inversion()
            self.modified_round = rnd
            return True

    def IS_transposition(self, rate, rnd):
        if random() <= rate:
            gene_to_transpose = floor(self.gene_number * random())
            self.genes[gene_to_transpose].IS_transposition()
            self.modified_round = rnd
            return True

    def RIS_transposition(self, rate, rnd):
        if random() <= rate:
            gene_to_transpose = floor(self.gene_number * random())
            if self.genes[gene_to_transpose].RIS_transposition():
                self.modified_round = rnd
                return True
        return False

    def gene_transposition(self, rate, rnd):
        if random() <= rate:
            gene_to_transpose = floor(self.gene_number * random())
            self.genes[0] = deepcopy(self.genes[gene_to_transpose])
            self.modified_round = rnd
            return True
        return False

    def o_recombination(self, rate, rnd):
        if random() <= rate:
            gene_to_o = floor(self.gene_number * random())
            self.genes[0] = deepcopy(self.genes[gene_to_o])
            self.modified_round = rnd
            return True
        return False

    def t_recombination(self, rate, rnd):
        if random() <= rate:
            gene_to_t = floor(self.gene_number * random())
            self.genes[0] = deepcopy(self.genes[gene_to_t])
            self.modified_round = rnd
            return True
        return False

    def gene_recombination(self, rate, rnd):
        if random() <= rate:
            gene_to_recombination = floor(self.gene_number * random())
            self.genes[0] = deepcopy(self.genes[gene_to_recombination])
            self.modified_round = rnd
            return True
        return False

    def one_point_recombination(self, rate, rnd):
        if random() <= rate:
            gene_to_o1 = floor(self.gene_number * random())
            gene_to_o2 = floor(self.gene_number * random())
            r = floor((self.genes_head * 2 + 1) * random())
            if gene_to_o1 != gene_to_o2:
                self.genes[gene_to_o1].one_point_recombination(self.genes[gene_to_o2],r)
                self.genes[gene_to_o2].one_point_recombination(self.genes[gene_to_o1],r)
                self.modified_round = rnd
                return True
        return False
    
    @property
    def modified(self):
        return self.modified_round

    def __cmp__(self, other):
        # Returns -1 if self < other, 0 if self == other, 1 if self > other
        if self.fitness < other.fitness:
            return -1
        elif self.fitness > other.fitness:
            return 1
        elif self.modified > other.modified:
            return 1
        elif self.modified < other.modified:
            return -1
        else:
            return 0

    def __eq__(self, other):
        return self.__cmp__(other) == 0

    def __neq__(self, other):
        return self.__cmp__(other) != 0

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __le__(self, other):
        return self.__cmp__(other) <= 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

    def __ge__(self, other):
        return self.__cmp__(other) >= 0
