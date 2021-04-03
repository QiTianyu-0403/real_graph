import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import collections
from matplotlib.pyplot import MultipleLocator
import networkx as nx
import numpy as np
import math
from scipy.optimize import curve_fit

'''test the degree'''
def draw_old(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("value Histogram")
    plt.ylabel("Count")
    plt.xlabel("value")
    ax.set_xticks([d for d in deg])
    ax.set_xticklabels(deg)

    plt.show()



'''#draw the degree'''
def draw_degree(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    degree = nx.degree_histogram(G)
    x = range(len(degree))
    y = [z / float(sum(degree)) for z in degree]

    # fig = plt.figure()
    fig, ax1 = plt.subplots()
    # plt.bar([d for d in deg], cnt, width=0.50, color="b",label='wwww')

    plt.title("Degree Distribution of OGB-arxiv", fontsize='xx-large')

    # bar graph
    ax1.bar([d for d in deg], cnt, width=0.50, color="b", label="Number of nodes in degree distribution")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Degree")
    x_minor_locator_ax1 = MultipleLocator(2)
    y_minor_locator_ax1 = MultipleLocator(500)
    ax1.xaxis.set_minor_locator(x_minor_locator_ax1)
    ax1.yaxis.set_minor_locator(y_minor_locator_ax1)
    ax1.set_xticks([d * 20 for d in range(0, 50)])
    ax1.set_xticklabels([d * 20 for d in range(0, 50)])

    ax2 = ax1.twinx()

    # plot graph
    ax2.plot(x, y, 'r', label="Degree distribution probability")
    ax2.set_ylabel('Probability')
    y_minor_locator_ax2 = MultipleLocator(0.01)
    ax2.yaxis.set_minor_locator(y_minor_locator_ax2)
    ax2.set_ylim(-0.01, 0.3)

    # Legend location
    ax1.legend(bbox_to_anchor=(1, 0.98))
    ax2.legend(bbox_to_anchor=(1, 0.88))

    # X-axis limit
    plt.xlim(-5, 250)

    #plt.text(150,0.35,'Number of nodes:10000')
    #plt.text(150,0.3,'Number of blocks:2')

    plt.savefig('picture.png')
    plt.show()

'''#draw a random Power law image'''
def draw(a):
    fig, ax = plt.subplots()

    plt.bar(range(1,50), a, width=0.80, color="b")
    plt.title("powerlaw")

    ax.set_xticks([d for d in range(1,50)])
    ax.set_xticklabels(a)

    plt.show()

'''# draw the degree of feature'''
def draw_data_x(data):

    p = data.x.sum(axis=0)
    p = p.numpy().tolist()
    num=[]
    for i in range(0,int(max(p)+1)):
        num.append(0)
    for i in range(0,1433):
        num[int(p[i])]+=1
    print(max(p))

    fig, ax = plt.subplots()
    plt.bar([i for i in range(0,len(num))], num, width=0.80, color="b")


    ax.set_xticks([d for d in range(0,len(num))])
    ax.set_xticklabels(range(0,len(num)))

    plt.show()

'''# fit function'''
def func1(x, a, b):
    y = a / (x**b)
    return y
def func2(x,a,b):
    y=a*x+b
    return y

'''# draw the exponential fting'''
def draw_degree_fit_power(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    x_minor_locator= MultipleLocator(20)
    y_minor_locator = MultipleLocator(500)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)

    plt.scatter([d for d in deg], cnt, alpha=0.5 ,s=50, c="b", marker='^',label='Number of nodes in degree distribution')

    x=[]
    y=[]
    for i in range(0,len(deg)):
        if deg[i] != 0:
            x.append(deg[i])
            y.append(cnt[i])
    popt, pcov = curve_fit(func1, x, y)
    y_pred2 = [func1(i, popt[0],popt[1]) for i in x]

    plt.plot([d for d in x],y_pred2,c='#000000',linestyle='-.',label='Fitted curve (Power-law)')

    plt.title("Degree distribution fitting of OGB-arxiv", fontsize='xx-large')
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.legend()
    #plt.xscale("log")
    #plt.yscale("log")
    plt.xlim(-50, 2000)
     #plt.ylim(1, 1000)
   # plt.text(20,30,'Number of nodes:1000')
   # plt.text(20,10,'Number of blocks:2')

    plt.savefig('picture.png')
    plt.show()

'''# draw the straight-line fitting'''
def draw_degree_fit_line(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    x_minor_locator= MultipleLocator(10)
    y_minor_locator = MultipleLocator(10)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)

    x=[]
    y=[]
    for i in range(0,len(deg)):
        if deg[i] > 0:
            x.append(math.log(deg[i],10))
            y.append(math.log(cnt[i],10))


    popt, pcov = curve_fit(func2, x, y)
    y_pred2 = [func2(i, popt[0],popt[1]) for i in x]


    plt.plot([d for d in x],y_pred2,c='#000000',linestyle='-.',label='Fitted curve (log)')
    plt.scatter([d for d in x], y, alpha=0.5 ,s=50, c="b", marker='^',label='Number of nodes in degree distribution')

    plt.title("Degree distribution fitting of OGB-arxiv", fontsize='xx-large')
    plt.xlabel("lg(Degree)")
    plt.ylabel("lg(Count)")
    plt.legend()

   # plt.xlim(-0.3, 2.5)
    plt.ylim(-0.5, 5)

    plt.savefig('picture.png')
    plt.show()


'''#get the average degree'''
def get_ave_degree(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    degree_sum = 0
    for node in range(0,len(deg)):
        degree_sum = deg[node] * cnt[node] + degree_sum
    degree_average = degree_sum / len(degree_sequence)
    print('the average degree:%f'%degree_average)
    return degree_average

def get_data_ave_degree(degree):
    a = degree.tolist()
    print('the average degree of real graph:%f'%np.mean(a))

def draw_network(G):
    nx.draw(G, pos=nx.layout.spring_layout(G), node_color='b', edge_color='#000000',width=0.3,style='solid', with_labels=True, font_size=1, node_size=10)
    plt.show()

