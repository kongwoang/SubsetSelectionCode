import sys
import numpy as np
import datetime
from math import ceil,exp
from random import random,randint
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import copy
import algorithms as algos

class ObjectiveMC(object):
    def __init__(self, data, budget):
        self.data = data
        self.budget=budget
    
    def InitDVC(self, n, q, outdegree_file=None):
        self.n = n
        self.cost = [0] * self.n
        sum=0
        for i in range(self.n):
            tempElemetn = [i]
            tempElemetn.extend(self.data[i])
            tempValue = len(list(set(tempElemetn)))-q
            if tempValue > 0:
                self.cost[i] = tempValue + 1
            else:
                self.cost[i] = 1
                sum+=1
        a=1

        
  
    def Position(self, s):
        return np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

    def FS(self, s):
        pos = self.Position(s)
        tempSet = []
        for j in pos:
            tempSet.extend(self.data[j])
        tempSet.extend(pos)
        tempSet = list(set(tempSet))
        
        tempSum = len(tempSet)

        return tempSum

    def CS(self, s):
        pos = self.Position(s)
        tempSum = 0.0
        for item in pos:
            tempSum += self.cost[item] 
        return tempSum

    def max_subset_size(self):
        self.dp = [[0] * (ceil(self.budget) + 1) for _ in range(self.n + 1)]
        
        for i in range(1, self.n+1):
            cost = self.cost[i-1]
            for j in range(ceil(self.budget) + 1):
                self.dp[i][j] = self.dp[i-1][j]
                if j >= cost:
                    self.dp[i][j] = max(self.dp[i][j], self.dp[i-1][round(j-cost)] + 1)
        
        return self.dp[self.n][ceil(self.budget)]


def GetDVCData(fileName,n):# node number start from 0
    node_neighbor = []
    i = 0
    file = open(fileName)
    lines = file.readlines()
    index=0
    while i < n:
        currentLine = []
        lines=lines[index:]
        for j in range(len(lines)):
            items = lines[j].split()
            if int(items[0]) == int(i+1):
                currentLine.append(int(int(items[1])-1))
            else:
                index=j
                break
        node_neighbor.append(currentLine)
        i += 1
    file.close()
    return node_neighbor

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main(args):
    # save args settings
    adjacency_file="outdegree/"+args.adjacency_file
    algo=args.algo
    budget=args.budget
    times=args.times
    budget=args.budget
    T=args.T
    print(adjacency_file,budget,algo,T)

    #problem
    data = GetDVCData(adjacency_file,args.n)
    myObject = ObjectiveMC(data, budget)
    myObject.InitDVC(args.n, args.q)

    K_B=myObject.max_subset_size()
    GreedyEvaualte=myObject.n * K_B
    print(K_B)


    if algo=="sto_EVO_SMC" :
        res_file='Result-q='+str(args.q)+'/'+ args.adjacency_file+ '_'+ algo + '_'+ str(args.epsilon) +'_' + str(args.prob) +'_' + str(budget) 
    else:
        res_file='Result-q='+str(args.q)+'/'+ args.adjacency_file+ '_'+ algo +'_' + str(budget) 

    if not os.path.exists(res_file):
        os.makedirs(res_file)

    if algo=="GGA":
        algos.GGA(res_file,times,myObject)

    elif algo=="greedy_max":
        algos.greedy_max(res_file,times,myObject)
    
    elif algo=="one_guess_greedy_plus":
        algos.one_guess_greedy_plus(res_file,times,myObject)

    elif algo=="POMC":
        algos.POMC(res_file,times,T,GreedyEvaualte,myObject)
    elif algo=="EAMC":
        algos.EAMC(res_file,times,T,GreedyEvaualte,myObject)
    elif algo=="FPOMC":
        algos.FPOMC(res_file,times,T,GreedyEvaualte,myObject)
    elif algo=="EVO_SMC":
        algos.sto_EVO_SMC(res_file,times,T,GreedyEvaualte,myObject)
    elif algo=="sto_EVO_SMC":
        algos.sto_EVO_SMC(res_file,times,T,GreedyEvaualte,myObject,args.epsilon,args.prob)
    elif algo=="EPOMC":
        algos.EPOMC(res_file,times,T,GreedyEvaualte,myObject)

    elif algo=="PPOMC":
        algos.P_POMC(res_file,times,T,GreedyEvaualte,myObject)

    else:
        print("no suitable algo")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-adjacency_file',type=str,default= "congress.edgelist-new.txt")
    argparser.add_argument('-q',type=int, default=5)
    argparser.add_argument('-n',type=int, default=475)

    argparser.add_argument('-T', type=int, default=20)
    argparser.add_argument('-budget',type=float, default=500)
    argparser.add_argument('-times', type=int, default=3)
    argparser.add_argument('-algo',type=str, default="PPOMC")

    argparser.add_argument('-prob',type=float, default=0.5,help="sto_evo_smc_p")
    argparser.add_argument('-epsilon',type=float, default=0.1,help="sto_evo_smc_epsilon")


    args = argparser.parse_args()
    main(args)