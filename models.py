
import random

import networkx as nx
import itertools
import math
from networkx.utils import py_random_state


def data_to_G(data):
    G=nx.Graph()
    nodes_num = len(data.adj)
    print("nodes num:%d" %nodes_num)
    print(data.edge_index)

    point = []
    for i in range(nodes_num):
        point.append(i)
    G.add_nodes_from(point)

    edge_row = data.edge_index.shape[0]
    edge_col = data.edge_index.shape[1]

    print("edge_row num:%d" % edge_row)
    print("edge_col num:%d" % edge_col)
    edges = []
    for i in range(edge_col):
        if data.edge_index[0][i].item() <= data.edge_index[1][i].item():
            edges.append((data.edge_index[0][i].item(),data.edge_index[1][i].item()))
    #print(edges)
    G.add_edges_from(edges)
    return G



# Generate the block probability matrix
def p_random(block):
    p = []
    for i in range(block):
        p.append([])
        for j in range(block):
            if i == j:
                p[i].append(random.uniform(0.2, 0.4))
            elif i < j:
                p[i].append(random.uniform(0.02, 0.1))
            else:
                p[i].append(p[j][i])
    return p


# Generate the block probability matrix (Value of pin and pout is the only)
def p_random_simple(block):
    p = []
    pin=random.uniform(0.2, 0.3)
    pout=random.uniform(0.05, 0.1)
    print('pin---:%f' % pin)
    print('pout---:%f' % pout)
    for i in range(block):
        p.append([])
        for j in range(block):
            if i == j:
                p[i].append(pin)
            else:
                p[i].append(pout)
    return p

# generate the Zipf function
def zipf(r,C,alpha):
    return C*r**(-alpha)


# generate SBM
@py_random_state(3)
def SBM(sizes, p, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True):
    if len(sizes) != len(p):
        raise nx.NetworkXException("'sizes' and 'p' do not match.")
    # Check for probability symmetry (undirected) and shape (directed)
    for row in p:
        if len(p) != len(row):
            raise nx.NetworkXException("'p' must be a square matrix.")
    if not directed:
        p_transpose = [list(i) for i in zip(*p)]
        for i in zip(p, p_transpose):
            for j in zip(i[0], i[1]):
                if abs(j[0] - j[1]) > 1e-08:
                    raise nx.NetworkXException("'p' must be symmetric.")
    # Check for probability range
    for row in p:
        for prob in row:
            if prob < 0 or prob > 1:
                raise nx.NetworkXException("Entries of 'p' not in [0,1].")
    # Check for nodelist consistency
    if nodelist is not None:
        if len(nodelist) != sum(sizes):
            raise nx.NetworkXException("'nodelist' and 'sizes' do not match.")
        if len(nodelist) != len(set(nodelist)):
            raise nx.NetworkXException("nodelist contains duplicate.")
    else:
        nodelist = range(0, sum(sizes))

    # Setup the graph conditionally to the directed switch.
    block_range = range(len(sizes))
    if directed:
        g = nx.DiGraph()
        block_iter = itertools.product(block_range, block_range)
    else:
        g = nx.Graph()
        block_iter = itertools.combinations_with_replacement(block_range, 2)
    # Split nodelist in a partition (list of sets).
    size_cumsum = [sum(sizes[0:x]) for x in range(0, len(sizes) + 1)]
    g.graph["partition"] = [
        set(nodelist[size_cumsum[x] : size_cumsum[x + 1]])
        for x in range(0, len(size_cumsum) - 1)
    ]
    # Setup nodes and graph name
    for block_id, nodes in enumerate(g.graph["partition"]):
        for node in nodes:
            g.add_node(node, block=block_id)

    g.name = "stochastic_block_model"

    # Test for edge existence
    parts = g.graph["partition"]
    for i, j in block_iter:
        if i == j:
            if directed:
                if selfloops:
                    edges = itertools.product(parts[i], parts[i])
                else:
                    edges = itertools.permutations(parts[i], 2)
            else:
                edges = itertools.combinations(parts[i], 2)
                if selfloops:
                    edges = itertools.chain(edges, zip(parts[i], parts[i]))
            for e in edges:
                if seed.random() < p[i][j]:
                    g.add_edge(*e)
        else:
            edges = itertools.product(parts[i], parts[j])
        if sparse:
            if p[i][j] == 1:  # Test edges cases p_ij = 0 or 1
                for e in edges:
                    g.add_edge(*e)
            elif p[i][j] > 0:
                while True:
                    try:
                        logrand = math.log(seed.random())
                        skip = math.floor(logrand / math.log(1 - p[i][j]))
                        # consume "skip" edges
                        next(itertools.islice(edges, skip, skip), None)
                        e = next(edges)
                        g.add_edge(*e)  # __safe
                    except StopIteration:
                        break
        else:
            for e in edges:
                if seed.random() < p[i][j]:
                    g.add_edge(*e)  # __safe
    return g


def get_weight(degree,pin,pout):
    normal=1
    weight=[]
    print(len(degree))
    for i in degree:
        weight.append(i[0]/140)
    random.shuffle(weight)
    print(weight)
    test = random.sample(weight, 2)
    ave_degree = 0.25 * (test[0] * pin + test[1] * pout) * len(degree)
    print('ave_degree:%f' % ave_degree)
    '''
    for ran in range(0,20):
        test = random.sample(weight,2)
        ave_degree=0.25*(test[0]*pin+test[1]*pout)*len(degree)
        print('ave_degree:%f' % ave_degree)
        if ave_degree>5 or ave_degree<2:
            normal+=10
            for i in degree:
                weight.append(i[0]/normal)
            random.shuffle(weight)
            print('normal:%d'%normal)
        else:
            break
    '''
    return weight

# generate DCSBM
@py_random_state(4)
def DCSBM(sizes, p, theta, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True):
    if len(sizes) != len(p):
        raise nx.NetworkXException("'sizes' and 'p' do not match.")
    # Check for probability symmetry (undirected) and shape (directed)
    for row in p:
        if len(p) != len(row):
            raise nx.NetworkXException("'p' must be a square matrix.")
    if not directed:
        p_transpose = [list(i) for i in zip(*p)]
        for i in zip(p, p_transpose):
            for j in zip(i[0], i[1]):
                if abs(j[0] - j[1]) > 1e-08:
                    raise nx.NetworkXException("'p' must be symmetric.")
    # Check for probability range
    for row in p:
        for prob in row:
            if prob < 0 or prob > 1:
                raise nx.NetworkXException("Entries of 'p' not in [0,1].")
    # Check for nodelist consistency
    if nodelist is not None:
        if len(nodelist) != sum(sizes):
            raise nx.NetworkXException("'nodelist' and 'sizes' do not match.")
        if len(nodelist) != len(set(nodelist)):
            raise nx.NetworkXException("nodelist contains duplicate.")
    else:
        nodelist = range(0, sum(sizes))

    # Setup the graph conditionally to the directed switch.
    block_range = range(len(sizes))
    if directed:
        g = nx.DiGraph()
        block_iter = itertools.product(block_range, block_range)
    else:
        g = nx.Graph()
        block_iter = itertools.combinations_with_replacement(block_range, 2)
    # Split nodelist in a partition (list of sets).
    size_cumsum = [sum(sizes[0:x]) for x in range(0, len(sizes) + 1)]
    g.graph["partition"] = [
        set(nodelist[size_cumsum[x] : size_cumsum[x + 1]])
        for x in range(0, len(size_cumsum) - 1)
    ]
    # Setup nodes and graph name
    for block_id, nodes in enumerate(g.graph["partition"]):
        for node in nodes:
            g.add_node(node, block=block_id)

    g.name = "stochastic_block_model"

    # Test for edge existence
    parts = g.graph["partition"]
    print(parts[0])
    for i, j in block_iter:
        if i == j:
            if directed:
                if selfloops:
                    edges = itertools.product(parts[i], parts[i])
                else:
                    edges = itertools.permutations(parts[i], 2)
            else:
                edges = itertools.combinations(parts[i], 2)
                if selfloops:
                    edges = itertools.chain(edges, zip(parts[i], parts[i]))
            for e in edges:
                if seed.random() < p[i][j]*(theta[e[0]]*theta[e[1]]):
                    g.add_edge(*e)
        else:
            edges = itertools.product(parts[i], parts[j])
        '''
        if sparse:
            if p[i][j] == 1:  # Test edges cases p_ij = 0 or 1
                for e in edges:
                    g.add_edge(*e)
            elif p[i][j] > 0:
                while True:
                    try:
                        logrand = math.log(seed.random())
                        skip = math.floor(logrand / math.log(1 - p[i][j]))
                        # consume "skip" edges
                        next(itertools.islice(edges, skip, skip), None)
                        e = next(edges)
                        g.add_edge(*e)  # __safe
                    except StopIteration:
                        break
        else:
        '''
        for e in edges:
            if seed.random() < p[i][j]*(theta[e[0]]*theta[e[1]]):
                g.add_edge(*e)  # __safe

    return g

