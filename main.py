from models import data_to_G,SBM,DCSBM
from get_func import p_random,p_random_simple,zipf,get_weight,statistic_nodes,statistic_edges
from draw import draw_degree,draw,draw_data_x,draw_degree_fit_power,\
                    draw_degree_fit_line,draw_old,get_ave_degree,draw_network,get_data_ave_degree
from graph import init

if __name__ == '__main__':
    data,degree = init()

    #P,dict = statistic_edges(data)
    #G2 = DCSBM(sizes=[dict[0], dict[1],dict[2],dict[3],dict[4],dict[5],dict[6]], p=P, theta=get_weight(degree), sparse=True)
    #G2 = SBM(sizes=[dict[0], dict[1],dict[2],dict[3],dict[4],dict[5],dict[6]], p=P, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True)
    #G=data_to_G(data)
    #draw_old(G2)
    #draw_degree_fit_line(G)

    '''
    *******   Generate DCSBM from random sequence   ************
    '''
    '''
    power=[]
    for i in range(1,10001):
        power.append(zipf(i,450,1.3))

    random.shuffle(power)
    G=DCSBM(sizes=[500,500], p=p_random_simple(2), theta=power, sparse=True)
    draw_degree_fit_power(G)
    '''

    '''
    *******   Generate DCSBM according to the real graph distribution  ************
    '''
    get_data_ave_degree(degree)
    p=p_random_simple(2)
    weight = get_weight(degree)
    G2=DCSBM(sizes=[1354, 1354], p=p,theta=weight, sparse=True)
    #e=list(G2.nodes)
    #G2.remove_nodes_from(e)
    get_ave_degree(G2)
    draw_old(G2)

    print('hello')