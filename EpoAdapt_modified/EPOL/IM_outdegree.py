import sys
import numpy as np
import datetime
from math import ceil,exp
from random import randint,choice,randrange
import random
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import copy
import algorithms as algos
import time

# --- NumPy 2.0 compatibility shim for removed np.mat ---
if not hasattr(np, "mat"):
    # fall back to asmatrix (still present) or emulate it
    try:
        np.mat = np.asmatrix
    except AttributeError:
        # very defensive: emulate asmatrix behavior (2D guarantee)
        def _asmatrix(a, dtype=None):
            return np.array(a, dtype=dtype, copy=False, ndmin=2)
        np.asmatrix = _asmatrix
        np.mat = np.asmatrix
# --------------------------------------------------------


#import pickle5 as pickle
class ObjectiveIM(object):
    def __init__(self, weightMatrix,nodeNum,outdegree_file,budget):
        self.weightMatrix = weightMatrix
        self.n=nodeNum
        self.budget=budget
        self.solution = []
        self.allNodes=np.ones((1, self.n))
        self.cost=[0]*self.n

        dataFile=open(outdegree_file)
        dataLine=dataFile.readlines()
        items=dataLine[0].split()
        eps=[0]*self.n
        for i in range(self.n):
            eps[i]=float(items[i])
        dataFile.close()
        sum=0
        for i in range(self.n):
            outDegree=(self.weightMatrix[i,:]>0).sum()
            self.cost[i]=1.0+(1+abs(eps[i]))*outDegree
            sum+=self.cost[i]
        #print(sum)
 
    def FinalActiveNodes(self):  # solution is the numpy matrix 1*n
        activeNodes = np.zeros((1, self.n)) + self.solution
        cActive = np.zeros((1, self.n)) + self.solution# currently active nodes
        tempNum = int(cActive.sum(axis=1)[0, 0])
        while tempNum > 0:
            nActive = self.allNodes - activeNodes
            randMatirx = np.random.rand(tempNum, self.n)#uniformly random matrix between 0 and 1
            z = sum(randMatirx < self.weightMatrix[cActive.nonzero()[-1], :]) > 0 #cActive.nonzero()[-1] is the nonzero index
            cActive = np.multiply(nActive, z) #sum is the sum of each column,the new active node
            activeNodes = (cActive + activeNodes) > 0
            tempNum = int(cActive.sum(axis=1)[0, 0])
        return activeNodes.sum(axis=1)[0, 0]

    def Position(self, s):
        return np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

    def FS(self, solution, realEvaluate=False):  # simulate 500 times
        simulate_times = 10000 if realEvaluate else 500
        self.solution = solution
        val = 0
        for i in range(simulate_times):
            val += self.FinalActiveNodes()
        return val / (simulate_times*1.0)

    def CS(self,s):
        tempSum=0
        pos=self.Position(s)
        for item in pos:
            tempSum=tempSum+self.cost[item]
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
            
def ReadData(p,filePath):
    dataFile=open(filePath)
    maxNode=0
    while True:
        line=dataFile.readline()
        if not line:
            break
        items=line.split()
        if len(items)>0:
            start=int(items[0])
            end=int(items[1])
            if start>maxNode:
                maxNode=start
            if end>maxNode:
                maxNode=end
    dataFile.close()
    maxNode=maxNode

    data = np.mat(np.zeros([maxNode, maxNode], dtype=float))

    dataFile = open(filePath)
    while True:
        line = dataFile.readline()
        if not line:
            break
        items = line.split()
        if len(items)>0:
            data[int(items[0])-1,int(items[1])-1]=p
    dataFile.close()
    return data

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
    adjacency_file="outdegree/"+args.adjacency_file
    outdegree_file="outdegree/"+args.outdegree_file
    algo=args.algo
    probability=args.probability
    budget=args.budget
    T=args.T
    times=args.times
    print(adjacency_file,budget,algo, T)

    #problem
    weightMatrix=ReadData(probability,adjacency_file)
    nodeNum=np.shape(weightMatrix)[0]
    myObject=ObjectiveIM(weightMatrix,nodeNum,outdegree_file,budget)

    K_B=myObject.max_subset_size()
    print(K_B)
    GreedyEvaualte=myObject.n * K_B


    if algo=="sto_EVO_SMC" :
        res_file='Result_01/'+ args.adjacency_file+ '_'+ algo + '_'+ str(args.epsilon) +'_' + str(args.prob) +'_' + str(budget) 
    else:
        res_file='Result_01/'+ args.adjacency_file+ '_'+ algo + '_' + str(probability) +'_' + str(budget) 

    if not os.path.exists(res_file):
        os.makedirs(res_file)

    if algo=="GGA":
        algos.GGA(res_file,times,myObject)

    elif algo=="greedy_max":
        algos.greedy_max(res_file,times,myObject)
    
    elif algo=="one_guess_greedy_plus":
        algos.one_guess_greedy_plus(res_file,times,myObject)

    elif algo=="POMC":
        start_time = time.perf_counter()   # High-resolution timer
        algos.POMC(res_file,times,T,GreedyEvaualte,myObject)
        end_time = time.perf_counter()
        print("→ POMC returned; exiting main()")
        print("--------------------------------------")
        print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    elif algo=="EAMC":
        start_time = time.perf_counter()
        algos.EAMC(res_file,times,T,GreedyEvaualte,myObject)
        end_time = time.perf_counter()
        print("→ EAMC returned; exiting main()")
        print("--------------------------------------")
        print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    elif algo=="FPOMC":
        start_time = time.perf_counter()
        algos.FPOMC(res_file,times,T,GreedyEvaualte,myObject)
        end_time = time.perf_counter()
        print("→ FPOMC returned; exiting main()")
        print("--------------------------------------")
        print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    elif algo=="EVO_SMC":
        algos.sto_EVO_SMC(res_file,times,T,GreedyEvaualte,myObject)
    elif algo=="sto_EVO_SMC":
        algos.sto_EVO_SMC(res_file,times,T,GreedyEvaualte,myObject,args.epsilon,args.prob)
        
    elif algo=="EPOMC":
        start_time = time.perf_counter()   # High-resolution timer
        algos.EPOMC(res_file,times,T,GreedyEvaualte,myObject)
        end_time = time.perf_counter()
        print("→ EPOMC returned; exiting main()")
        print("--------------------------------------")
        print(f"Elapsed time: {end_time - start_time:.4f} seconds")

    elif algo=="PPOMC":
        algos.P_POMC(res_file,times,T,GreedyEvaualte,myObject)

    else:
        print("no suitable algo")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-adjacency_file',type=str,default= "graph200-01.txt")
    argparser.add_argument('-outdegree_file',type=str,default= "graph200_eps.txt")
    argparser.add_argument('-probability',type=float, default=0.1)
    argparser.add_argument('-T', type=int, default=20)

    argparser.add_argument('-budget',type=float, default=100)

    argparser.add_argument('-algo',type=str, default="PPOMC")
    argparser.add_argument('-prob',type=float, default=0.5,help="sto_evo_smc_p")
    argparser.add_argument('-epsilon',type=float, default=0.1,help="sto_evo_smc_epsilon")
    argparser.add_argument('-times', type=int, default=0)

    args = argparser.parse_args()
    main(args)
