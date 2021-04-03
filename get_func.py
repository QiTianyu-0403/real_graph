import random
import math


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


def get_weight(degree):
    weight=[]
    for i in degree:
        weight.append(i[0]/50)
    random.shuffle(weight)
    print(weight)

    #ave_degree = 0.25 * (test[0] * pin + test[1] * pout) * len(degree)
    #print('ave_degree:%f' % ave_degree)
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


def statistic_nodes(data):
    class_nodes = data.y.tolist()
    print(class_nodes)
    dict = {x: class_nodes.count(x) for x in set(class_nodes)}
    print(dict)
    return dict,class_nodes


def statistic_edges(data):
    dict,class_nodes = statistic_nodes(data)
    print(data.edge_index)

    edges_num=[]
    for i in range(0,len(dict)):
        edges_num.append([])
        for j in range(0,len(dict)):
            edges_num[i].append(0)

    #edge_row = data.edge_index.shape[0]
    edge_col = data.edge_index.shape[1]

    for i in range(edge_col):
        if data.edge_index[0][i].item() <= data.edge_index[1][i].item():
            edges_num[class_nodes[data.edge_index[0][i].item()]][class_nodes[data.edge_index[1][i].item()]]+=1
    print(edges_num)

    for i in range(0,len(edges_num)):
        for j in range(0,len(edges_num[i])):
            if i < j:
                edges_num[i][j] += edges_num[j][i]
                edges_num[j][i] = edges_num[i][j]/2
                edges_num[i][j] = edges_num[j][i]
    print(edges_num)

    for i in range(0,len(edges_num)):
        for j in range(0,len(edges_num[i])):
            edges_num[i][j] = edges_num[i][j]/(dict[i]*dict[j])
    print(edges_num)
    return edges_num,dict


