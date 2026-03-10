import numpy as np
from math import ceil
import argparse
import os
import  time
import epo_adapt_opt_mut as algos
#*-------------------------------------------------------------------------------------------------------------------------
class ObjectiveIM(object):
    def __init__(self, weightMatrix,nodeNum,outdegree_file,budget):
        self.weightMatrix = weightMatrix
        self.n=nodeNum
        self.budget=budget

        self.solution = np.zeros((1, self.n), dtype='int8')
        self.allNodes=np.ones((1, self.n))
        self.cost=[0]*self.n

        dataFile=open(outdegree_file)
        dataLine=dataFile.readlines()
        items=dataLine[0].split()
        eps=[0]*self.n
        for i in range(self.n):
            eps[i]=float(items[i]) #type: ignore
        dataFile.close()
        sum=0
        for i in range(self.n):
            outDegree=(self.weightMatrix[i,:]>0).sum()
            self.cost[i]=1.0+(1+abs(eps[i]))*outDegree
            sum+=self.cost[i]
    #*-------------------------------------------------------------------------------------------------------------------------
    def FinalActiveNodes(self):
        """Optimized Independent Cascade simulation with early termination"""
        # Convert to simple arrays for faster operations
        activeNodes = np.zeros(self.n, dtype=bool)
        solution_indices = np.where(self.solution.ravel() == 1)[0]
        activeNodes[solution_indices] = True
        
        newly_active = solution_indices.copy()
        max_iterations = 50  # Prevent infinite loops
        iteration = 0
        
        while len(newly_active) > 0 and iteration < max_iterations:
            iteration += 1
            next_active = []
            
            # For each newly active node
            for u in newly_active:
                # Get neighbors and their activation probabilities
                neighbors = np.where(self.weightMatrix[u, :] > 0)[1]  # Get column indices where weight > 0
                probs = self.weightMatrix[u, neighbors].A1  # Convert to 1D array
                
                # Vectorized random activation
                rand_vals = np.random.random(len(neighbors))
                activated = neighbors[rand_vals < probs]
                
                # Add newly activated nodes
                for v in activated:
                    if not activeNodes[v]:
                        activeNodes[v] = True
                        next_active.append(v)
            
            newly_active = np.array(next_active)
        
        return int(np.sum(activeNodes))
    #*-------------------------------------------------------------------------------------------------------------------------
    def Position(self, s):
        a = np.array(s)
        row = a.ravel()
        return np.where(row == 1)[0]
    #*-------------------------------------------------------------------------------------------------------------------------
    def FS(self, solution):  # simulate 50 times
        simulate_times = 50
        self.solution = solution
        val = 0
        for _ in range(simulate_times):
            val += self.FinalActiveNodes()
        return val / (simulate_times*1.0)
    #*-------------------------------------------------------------------------------------------------------------------------
    def CS(self,s):
        tempSum=0
        pos=self.Position(s)
        for item in pos:
            tempSum=tempSum+self.cost[item]
        return tempSum
    #*-------------------------------------------------------------------------------------------------------------------------
    def max_subset_size(self):
        self.dp = [[0] * (ceil(self.budget) + 1) for _ in range(self.n + 1)]
        
        for i in range(1, self.n+1):
            cost = self.cost[i-1]
            for j in range(ceil(self.budget) + 1):          
                self.dp[i][j] = self.dp[i-1][j]
                if j >= cost:
                    self.dp[i][j] = max(self.dp[i][j], self.dp[i-1][round(j-cost)] + 1)
        
        return self.dp[self.n][ceil(self.budget)]
#*-------------------------------------------------------------------------------------------------------------------------           
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

    data = np.asmatrix(np.zeros([maxNode, maxNode]))
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
#*-------------------------------------------------------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#*-------------------------------------------------------------------------------------------------------------------------    
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
    GreedyEvaluate=myObject.n * K_B

    res_file='Results/'+ args.adjacency_file+ '_'+ algo + '_' + 'prob' + '_' + str(probability) + '_' + str(budget) 

    if not os.path.exists(res_file):
        os.makedirs(res_file, exist_ok=True)

    if algo=="EPOAdapt":
        start_time = time.perf_counter()   # High-resolution timer
        algos.EPOAdapt(res_file,times,T,GreedyEvaluate,myObject)
        end_time = time.perf_counter()
        print("→ EPOAdapt returned; exiting main()")
        print("--------------------------------------")
        print(f"Elapsed time: {end_time - start_time:.4f} seconds")

    else:
        print("no suitable algo")
#*-------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-adjacency_file',type=str,default= "graph200-01.txt")
    argparser.add_argument('-outdegree_file',type=str,default= "graph200_eps.txt")
    argparser.add_argument('-probability',type=float, default=0.1)
    argparser.add_argument('-T', type=int, default=20)

    argparser.add_argument('-budget',type=float, default=100)

    argparser.add_argument('-algo',type=str, default="EPOAdapt")
    argparser.add_argument('-prob',type=float, default=0.5)
    argparser.add_argument('-epsilon',type=float, default=0.1)
    argparser.add_argument('-times', type=int, default=0)

    args = argparser.parse_args()
    main(args)
