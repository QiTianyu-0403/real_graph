import random
import math
import matplotlib.pyplot as plt
import numpy as np
from models import DCSBM
from get_func import p_random_simple,get_weight,get_ave_degree


# code
def geneEncoding(pop_size, chrom_length):
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)
    return pop[1:]  # ???


# decode
def decodechrom(pop, chrom_length):
    temp = []
    for i in range(len(pop)):
        t = 0
        for j in range(chrom_length):
            t += pop[i][j] * (math.pow(2, chrom_length - 1 - j))
        temp.append(t)
    return temp


# calculation
def calobjValue(pop, chrom_length, min_value, max_value, degree, p, size):
    temp1 = []
    obj_value = []
    temp1 = decodechrom(pop, chrom_length)
    real_x=[]
    print('The progress of generating a population:|    ', end='')
    for i in range(len(temp1)):
        x = min_value + temp1[i] * (max_value - min_value) / (math.pow(2, chrom_length) - 1)
        real_x.append(x)

        weight = get_weight(degree,x)
        G = DCSBM(sizes=size, p=p, theta=weight, sparse=True)
        degree_value = get_ave_degree(G)
        e=list(G.nodes)
        G.remove_nodes_from(e)

        obj_value.append(degree_value)
        print('\b\b\b\b',end='')
        percent = (i + 1)*100 / len(temp1)
        print('# ', end='')
        if percent < 10:
            print(' ', end='')
        print('%d'%percent, end='')
        print('%', end='')
    print('|')
    print('x:', real_x)
    return obj_value


# fit function
def calfitValue(obj_value, c_min, real_degree):
    fit_value = []
    for i in range(len(obj_value)):
        '''
        if obj_value[i] < 0:
            temp = c_min - obj_value[i]
        else:
            temp = c_min
        fit_value.append(temp)
        '''
        temp = c_min - math.fabs(obj_value[i] - real_degree)
        if temp < 0:
            temp = 0

        fit_value.append(temp)
    return fit_value


# find the best solution coding
def best(pop, fit_value):
    px = len(pop)
    best_individual = []
    best_fit = fit_value[0]
    for i in range(1, px):
        if fit_value[i] > best_fit:
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]


# sum
def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total


# Cumulative probability
def cumsum(fit_value):
    for i in range(len(fit_value) - 2):
        t = 0
        j = 0
        while (j <= i):
            t += fit_value[j]
            j += 1
        fit_value[i] = t
        fit_value[len(fit_value) - 1] = 1


# Select operator
def selection(pop, fit_value):
    newfit_value = []
    # sum of fitting
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):
        newfit_value.append(fit_value[i] / total_fit)
    # Cumulative probability
    cumsum(newfit_value)
    ms = []
    pop_len = len(pop)
    for i in range(pop_len):
        ms.append(random.random())
    ms.sort()
    fitin = 0
    newin = 0
    newpop = pop
    # Roulette algorithm
    while newin < pop_len:
        if (ms[newin] < newfit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin += 1
        else:
            fitin += 1
    pop = newpop


# copulation
def crossover(pop, pc):
    pop_len = len(pop)
    for i in range(pop_len - 1):
        if random.random() < pc:
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i + 1][cpoint:len(pop[i])])
            temp2.extend(pop[i + 1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1
            pop[i + 1] = temp2


#  variation
def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])

    for i in range(px):
        if random.random() < pm:
            mpoint = random.randint(0, py - 1)
            if pop[i][mpoint] == 1:
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1


# binary TO decimalism (2 to 10)
def b2d(b, min_value, max_value, chrom_length):
    t = 0
    for j in range(len(b)):
        t += b[j] * (math.pow(2, len(b) - 1 - j))
    t = min_value + t * max_value / (math.pow(2, chrom_length) - 1)
    return t


# draw the plt
def plot_optimal(results, c_min):
    best_fun = 0
    best_individual = 0
    for item in results:
        if item[0] > best_fun:
            best_fun = item[0]
            best_individual = item[1]
    best_fun = c_min - best_fun
    x = np.arange(0.0, 10.0, 0.001)
    y = []
    for item in x:
        y.append(10 * math.sin(5 * item) + 7 * math.cos(4 * item))
    plt.plot(x, y)
    plt.plot(best_individual, best_fun, 'r*')


# get the c_min
def get_c_min(min_value, max_value, degree, p, size, real_degree):
    weight = get_weight(degree, max_value)
    G = DCSBM(sizes=size, p=p, theta=weight, sparse=True)
    degree_value1 = get_ave_degree(G)
    e = list(G.nodes)
    G.remove_nodes_from(e)
    if min_value != 0:
        weight = get_weight(degree, min_value)
        G = DCSBM(sizes=size, p=p, theta=weight, sparse=True)
        degree_value2 = get_ave_degree(G)
        e = list(G.nodes)
        G.remove_nodes_from(e)
        if degree_value1 > degree_value2:
            return math.fabs(degree_value1 - real_degree)
        else:
            return math.fabs(degree_value2 - real_degree)
    else:
        return math.fabs(degree_value1 - real_degree)


# get the degree_expect
def degree_expect(degree, p, x, size):
    sum = 0

    for i in range(0, 5):

        weight = get_weight(degree, x)
        G = DCSBM(sizes=size, p=p, theta=weight, sparse=True)
        degree_ave = get_ave_degree(G)
        e = list(G.nodes)
        G.remove_nodes_from(e)

        sum += degree_ave
    print('degree_expect:', sum/5)
    return sum/5


# find the border
def test_border(degree, p, size, real_degree):
    left = 0.
    right = 60.
    print('Ready to test the border---------------')

    while(1):
        print('left = ', left, 'right = ', right)
        degree_test = degree_expect(degree, p, right, size)
        if degree_test - real_degree > 0.5:
            left = right
            right += 60
            continue
        elif degree_test - real_degree < -0.5:
            middle = (left + right)/2
            print('middle = ', middle)
            degree_test = degree_expect(degree, p, middle, size)
            if degree_test - real_degree > 0.5:
                left = middle
                break
            elif degree_test - real_degree < -0.5:
                right = middle
                break
            else:
                left = middle - 15
                right = middle + 15
                break
        else:
            left = right - 15
            right = right + 15
            break
    print('min_value = ', left, 'max_value = ', right)
    return left, right


# main
def genetic_func(degree, p, size, real_degree):
    pop_size = 25  # population size
    stop_generation = 10  # Termination of algebra
    chrom_length = 10  # Chromosome length
    pc = 0.6  # crossover probability
    pm = 0.001  # mutation probability
    results = [[]]  # Store the optimal solution for each generation
    fit_value = []  # Individual fitness
    fit_mean = []  # Average fitness

    pop = geneEncoding(pop_size, chrom_length)
    min_value, max_value = test_border(degree, p, size, real_degree) # The left/right end of the domain
    print('the last:',min_value,max_value)

    c_min = get_c_min(min_value, max_value, degree, p, size, real_degree)
    print('c_min:',c_min)


    for i in range(stop_generation):
        print('run-time:', i, '-----------------------')

        obj_value = calobjValue(pop, chrom_length, min_value, max_value, degree, p, size)
        print('degree_value:',obj_value)
        fit_value = calfitValue(obj_value, c_min, real_degree)
        best_individual, best_fit = best(pop, fit_value)
        print('best_fit:', best_fit)
        results.append([best_fit, b2d(best_individual, min_value, max_value,
                                      chrom_length)])
        selection(pop, fit_value)
        crossover(pop, pc)
        mutation(pop, pm)

    results = results[1:]
    # results.sort()

    X = []
    Y = []
    for i in range(stop_generation):
        X.append(i)
        Y.append(-results[i][0] + c_min)
    print('Y:',Y)
    '''
    plt.plot(X, Y)
    plt.show()
    plot_optimal(results, c_min)
    plt.show()
    '''