from msgep.decorators import symbol
import math
import numpy


@symbol("+")
def addition(i, j):
    return i + j


@symbol("-")
def subtraction(i, j):
    return i - j


@symbol("*")
def multiplication(i, j):
    return i * j


@symbol("/")
def division(i, j):
    return i / j


@symbol("^")
def power(i, j):
    return numpy.power(i, j)


@symbol("sqrt")
def sqrt(i):
    return numpy.sqrt(i)


@symbol("sin")
def sin(i):
    return numpy.sin(i)


@symbol("cos")
def cos(i):
    return numpy.cos(i)


@symbol("exp")
def exp(i):
    return numpy.exp(i)


@symbol("ln")
def ln(i):
    return numpy.log(i)


@symbol("f")
def f(i):
    return -i


@symbol("F")
def F(i):
    return 1 / i
