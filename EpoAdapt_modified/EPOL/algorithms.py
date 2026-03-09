import sys
import numpy as np
import datetime
from math import ceil,exp
import math
from random import randint,choice,randrange,random
import random
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import copy
from scipy.sparse import lil_matrix
from functools import lru_cache
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def Position(s):
    return np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

def save_result(res_file, times, tempmax1, cost, budget, result, cpu_time_used, wall_time_used):
    log = open(res_file+'/result_'+str(times)+'.txt', 'a')
    log.write("value = "+str(tempmax1) + " cpu_time_used =" +str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + " cost = "+str(cost)+" budget = "+str(budget))
    log.write("\n")
    for item in Position(result):
        log.write(str(item))
        log.write(" ")
    log.write("\n")
    log.close()

def GGA(res_file,times,problem):
    start_wall = time.time()  # Start wall-clock time measurement
    start_cpu = time.process_time()  # Start CPU time measurement

    n=problem.n
    budget=problem.budget
    result = np.mat(np.zeros((1, n)), 'int8')
    V_pi = [1] * n
    selectedIndex = 0
    while sum(V_pi) > 0:
        f = problem.FS(result)
        c = problem.CS(result)
        maxVolume = -1
        for j in range(0, n):
            if V_pi[j] == 1:
                result[0, j] = 1
                cv = problem.CS(result)
                if cv > budget:
                    result[0, j] = 0
                    V_pi[j] = 0
                    continue
                fv = problem.FS(result)
                tempVolume = 1.0 * (fv - f) / (cv - c)
                if tempVolume > maxVolume:
                    maxVolume = tempVolume
                    selectedIndex = j
                result[0, j] = 0
        result[0, selectedIndex] = 1
        V_pi[selectedIndex] = 0

    tempMax = 0.
    tempresult = np.mat(np.zeros((1, n)), 'int8')
    selectedSingleton=0
    for i in range(n):
        if problem.cost[i] <= budget:
            tempresult[0, i] = 1
            tempVolume = problem.FS(tempresult)
            if tempVolume > tempMax:
                tempMax = tempVolume
                selectedSingleton=i
            tempresult[0, i] = 0
    tempresult[0,selectedSingleton]=1

    tempmax1 = problem.FS(result)
    if tempmax1 < tempMax:
        tempmax1=tempMax
        result=tempresult

    end_cpu = time.process_time()  
    end_wall = time.time()

    cpu_time_used = end_cpu - start_cpu
    wall_time_used = end_wall - start_wall


    save_result(res_file,times,tempmax1, problem.CS(result), problem.budget, result, cpu_time_used, wall_time_used)


def best_S_add_Item(groundSet, result_plus_singleItem,problem,f):
    n=problem.n
    budget=problem.budget
    maxVolume = -1
    max_f = -1
    selectedIndex = -1 

    for j in range(0, n):
        if groundSet[j] == 1:
            result_plus_singleItem[0,j]=1
            c_value=problem.CS(result_plus_singleItem)
            if c_value > budget:
                result_plus_singleItem[0, j] = 0
                groundSet[j] = 0
                continue
            f_value = problem.FS(result_plus_singleItem)
            tempVolume = f_value -f
            if tempVolume > maxVolume:
                maxVolume = tempVolume
                selectedIndex = j
                max_f = f_value
            result_plus_singleItem[0, j] = 0
            
    result_plus_singleItem[0, selectedIndex] = 1
    
    return result_plus_singleItem, max_f

def greedy_max(res_file,times,problem):
    start_wall = time.time()  # Start wall-clock time measurement
    start_cpu = time.process_time()  # Start CPU time measurement

    n=problem.n
    budget=problem.budget
    groundSet=[1]*n
    
    result = np.mat(np.zeros((1, n)), 'int8')
    best_solution = np.mat(np.zeros((1, n)), 'int8')
    selectedIndex = -1
    f_best=-1
    max_f=-1

    while sum(groundSet) > 0:
        
        f = problem.FS(result)
        c = problem.CS(result)

        result_plus_singleItem, value_1 = best_S_add_Item(groundSet,result.copy(),problem,f)
        if value_1 > f_best:
            best_solution=result_plus_singleItem
            f_best=value_1

        maxVolume = -1
    

        for j in range(0, n):
            if groundSet[j] == 1:
                result[0, j] = 1
                cv = problem.CS(result)
                if cv > budget:
                    result[0, j] = 0
                    groundSet[j] = 0
                    continue
                fv = problem.FS(result)
                tempVolume = 1.0 * (fv - f) / (cv - c)
                if tempVolume > maxVolume:
                    maxVolume = tempVolume
                    selectedIndex = j
                    max_f = fv
                result[0, j] = 0
        if selectedIndex != -1:
            result[0, selectedIndex] = 1
            groundSet[selectedIndex] = 0

        
    if max_f > f_best:
        best_solution = result
        f_best = max_f
    
    end_cpu = time.process_time() 
    end_wall = time.time() 

    cpu_time_used = end_cpu - start_cpu
    wall_time_used = end_wall - start_wall

    save_result(res_file,times,f_best, problem.CS(best_solution), budget, best_solution, cpu_time_used, wall_time_used)

def solution_plus_singleItem(solution,index):
    s=solution.copy() 
    s[0,index]=1
    return s

def process_item(j, problem, budget):
    n=problem.n
    zero_solution = np.mat(np.zeros((1, n)), 'int8')
    zero_solution[0, j] = 1
    c_single_item = problem.CS(zero_solution)
    if c_single_item > budget:
        zero_solution[0, j] = 0
        return None
    f_single_item = problem.FS(zero_solution)

    def new_func_f(x):
        return problem.FS(solution_plus_singleItem(x, j)) - f_single_item

    def new_func_c(x):
        return problem.CS(solution_plus_singleItem(x, j)) - c_single_item

    V_pi = [1] * n
    V_pi[j] = 0
    solution, solution_value = greedy_plus(V_pi, budget - c_single_item, new_func_f, new_func_c)
    solution[0, j] = 1
    f_value = solution_value + f_single_item


    zero_solution[0, j] = 0
    return j, f_value, solution

def greedy_plus(V, budget, func_f, func_c):
    groundSet=V.copy()
    n=len(groundSet)    
    result = np.mat(np.zeros((1, n)), 'int8')
    best_solution = np.mat(np.zeros((1, n)), 'int8')
    f_best=-1
    max_f=-1

    while sum(groundSet) > 0:
        f = func_f(result)
        c = func_c(result)

        result_plus_singleItem = result.copy()

        maxVolume = -1
        value_1 = -1
        selectedIndex = -1 

        for j in range(0, n):
            if groundSet[j] == 1:
                result_plus_singleItem[0,j]=1
                c_value=func_c(result_plus_singleItem)
                if c_value > budget:
                    result_plus_singleItem[0, j] = 0
                    groundSet[j] = 0
                    continue
                f_value = func_f(result_plus_singleItem)
                tempVolume = f_value -f
                if tempVolume > maxVolume:
                    maxVolume = tempVolume
                    selectedIndex = j
                    value_1 = f_value
                result_plus_singleItem[0, j] = 0
                
        result_plus_singleItem[0, selectedIndex] = 1
        #return result_plus_singleItem, max_f

        if value_1 > f_best:
            best_solution=result_plus_singleItem
            f_best=value_1

        maxVolume = -1
        selectedIndex = -1 
    

        for j in range(0, n):
            if groundSet[j] == 1:
                result[0, j] = 1
                cv = func_c(result)
                if cv > budget:
                    result[0, j] = 0
                    groundSet[j] = 0
                    continue
                fv = func_f(result)
                tempVolume = 1.0 * (fv - f) / (cv - c)
                if tempVolume > maxVolume:
                    maxVolume = tempVolume
                    selectedIndex = j
                    max_f = fv
                result[0, j] = 0
        if selectedIndex != -1:
            result[0, selectedIndex] = 1
            groundSet[selectedIndex] = 0

        
    if max_f > f_best:
        best_solution = result
        f_best = max_f

    return best_solution, f_best
    
def one_guess_greedy_plus(res_file,times,problem):
    start_wall = time.time()  # Start wall-clock time measurement
    start_cpu = time.process_time()  # Start CPU time measurement

    budget=problem.budget
    n=problem.n
    best_solution = np.mat(np.zeros((1, n)), 'int8')
    V_pi = [1] * n
    f_best=problem.FS(best_solution)

    num_of_processors = 100

    with ProcessPoolExecutor(max_workers=num_of_processors) as executor:
        futures = [executor.submit(process_item, j, problem, budget) for j in range(n)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                j, f_value, solution = result
                if f_value > f_best:
                    f_best = f_value
                    best_solution = solution

    end_cpu = time.process_time()  
    end_wall = time.time()  

    cpu_time_used = end_cpu - start_cpu
    wall_time_used = end_wall - start_wall
    
    save_result(res_file,times,f_best, problem.CS(best_solution), budget, best_solution, cpu_time_used, wall_time_used)

def mutation(n, s):
    rand_rate = 1.0 / (n)  
    change = np.random.binomial(1, rand_rate, n)
    return np.abs(s - change)

def POMC(res_file, times, T, GreedyEvaualte,problem):
    start_wall = time.time() 
    start_cpu = time.process_time() 

    n=problem.n
    budget=problem.budget

    population = np.mat(np.zeros([1, n], 'int8'))  # initiate the population
    fitness = np.mat(np.zeros([1, 2]))
    popSize = 1
    iter = 0
    nn = GreedyEvaualte
    totalTime=T* GreedyEvaualte

    best_value = 0
    best_times = 0
    
    pbar=tqdm(range(totalTime), position = 0, leave = True)
    current_progress = 0
    
    while  current_progress < totalTime:
        if iter >= nn:
            iter = 0
            resultIndex = -1
            maxValue = float("-inf")
            for p in range(0, popSize):
                if fitness[p, 1] <= budget and fitness[p, 0] > maxValue:
                    maxValue = fitness[p, 0]
                    resultIndex = p


            best_times = best_times + 1 if fitness[resultIndex,0] == best_value else  0
       
            if best_times >= 10:
                break

            else:
                best_value =  fitness[resultIndex,0]

            end_cpu = time.process_time() 
            end_wall = time.time()  

            cpu_time_used = end_cpu - start_cpu
            wall_time_used = end_wall - start_wall

            log = open(res_file+'/result_'+str(times)+'.txt', 'a')
            log.write("value = "+str(fitness[resultIndex,0]) + " cpu_time_used =" +str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + " cost = "+str(fitness[resultIndex,1])+" budget = "+str(budget)+ " population = "+str(popSize))
            log.write("\n")
               
            for item in Position(population[resultIndex,:]):
                log.write(str(item))
                log.write(" ")
            log.write("\n")
            log.close()
                
        s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
        offSpring = mutation(n,s)  # every bit will be flipped with probability 1/n
        offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
        offSpringFit[0, 1] = problem.CS(offSpring)
        if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > budget:
            continue
        offSpringFit[0, 0] = problem.FS(offSpring)
        hasBetter = False
        for i in range(0, popSize):
            if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                    fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                hasBetter = True
                break
        if hasBetter == False:  # there is no better individual than offSpring
            Q = []
            for j in range(0, popSize):
                if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                    continue
                else:
                    Q.append(j)
            # Q.sort()
            fitness = np.vstack((offSpringFit, fitness[Q, :]))  # update fitness
            population = np.vstack((offSpring, population[Q, :]))  # update population


        
        popSize = np.shape(fitness)[0]

        iter += 1
        current_progress+=1
        pbar.update(1)

    iter = 0
    resultIndex = -1
    maxValue = float("-inf")
    for p in range(0, popSize):
        if fitness[p, 1] <= budget and fitness[p, 0] > maxValue:
            maxValue = fitness[p, 0]
            resultIndex = p

    end_cpu = time.process_time()  
    end_wall = time.time() 

    cpu_time_used = end_cpu - start_cpu
    wall_time_used = end_wall - start_wall

    log = open(res_file+'/result_'+str(times)+'.txt', 'a')
    log.write("value = "+str(fitness[resultIndex,0]) + " cpu_time_used =" +str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + " cost = "+str(fitness[resultIndex,1])+" budget = "+str(budget)+ " population = "+str(popSize))
    log.write("\n")
        
    for item in Position(population[resultIndex,:]):
        log.write(str(item))
        log.write(" ")
    log.write("\n")
    log.close()
    
    pbar.close() 
    return population[resultIndex,:], fitness[resultIndex,0], fitness[resultIndex,1]
     
     
@lru_cache(maxsize=None) 
def GS(alpha, fitness, cost,budget):
    if fitness > 0:
        return 1.0 * fitness / (1.0 - (1.0 / exp(alpha * cost / budget)))
    else:
        return 0

def EAMC(res_file, times, T, GreedyEvaualte,problem):  ##just consider cost is less B
    start_wall = time.time()  # Start wall-clock time measurement
    start_cpu = time.process_time()  # Start CPU time measurement


    n = problem.n
    budget=problem.budget

    population_u = np.mat(np.zeros([1, n], 'int8'))  
    population_v = np.mat(np.zeros([1, n], 'int8'))

    fitness_u = np.mat(np.zeros([1, 3]))
    fitness_v = np.mat(np.zeros([1, 3]))

    best_value = 0
    best_times = 0

    iter = 0
    nn = GreedyEvaualte
    totalTime=T* GreedyEvaualte

    pbar=tqdm(range(totalTime), position = 0, leave = True)
    current_progress = 0
    
    while  current_progress < totalTime:
        if iter >= nn:
            iter = 0
            resultIndex = -1
            maxValue = float("-inf")
            fitness=np.vstack((fitness_v,fitness_u))
      
            population=np.vstack((population_v, population_u))
            popSize=population.shape[0]
            for p in range(0, popSize):
                if fitness[p, 1] <= budget and fitness[p, 0] > maxValue:
                    maxValue = fitness[p, 0]
                    resultIndex = p

            if fitness[resultIndex,0] == best_value:
                best_times += 1
            else:
                best_times = 0

            if best_times >= 10:
                break

            else:
                best_value =  fitness[resultIndex,0]
          
            end_cpu = time.process_time() 
            end_wall = time.time()  

            cpu_time_used = end_cpu - start_cpu
            wall_time_used = end_wall - start_wall
                   
            log = open(res_file+'/result_'+str(times)+'.txt', 'a')
            #log.write(" value = "+str(fitness[resultIndex,0])+" cost = "+str(fitness[resultIndex,1])+ " budget = " + str(budget) + " population = "+str(popSize))
            log.write("value = "+str(fitness[resultIndex,0]) + " cpu_time_used =" +str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + " cost = "+str(fitness[resultIndex,1])+" budget = "+str(budget)+ " population = "+str(popSize))
            
            log.write("\n")
            for item in Position(population[resultIndex,:]):
                log.write(str(item))
                log.write(" ")
            log.write("\n")
            log.close()


        population = np.vstack((population_u, population_v))  # update fitness
        popSize=population.shape[0]
        s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly


        offSpring = mutation(n,s)  # every bit will be flipped with probability 1/n
        offSpringFit = np.mat(np.zeros([1, 3])) #f, c, g
        offSpringFit[0, 1] = problem.CS(offSpring)

        if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > budget:
            continue

        offSpringFit[0, 0] = problem.FS(offSpring)
        offSpringFit[0, 2] = GS(1.0, offSpringFit[0,0],offSpringFit[0,1], budget)
        
        index = offSpring[0, :].sum()

        popSize_v=population_v.shape[0]
        no_solution=True

        for i in range(popSize_v):
            if population_v[i,:].sum() == index:
                no_solution=False
                if fitness_v[i, 0]<offSpringFit[0, 0]:
                    population_v[i, :]=offSpring
                    fitness_v[i, :]=offSpringFit
                break
        if no_solution:
            population_v = np.vstack((population_v, offSpring))
            fitness_v = np.vstack((fitness_v, offSpringFit))


        popSize_u=population_u.shape[0]
        no_solution=True

        for i in range(popSize_u):
            if population_u[i,:].sum() == index :
                no_solution=False
                if fitness_u[i, 2]<offSpringFit[0, 2]:
                    population_u[i, :]=offSpring
                    fitness_u[i, :]=offSpringFit
                break
        if no_solution:
            population_u = np.vstack((population_u,offSpring))
            fitness_u = np.vstack((fitness_u,offSpringFit))

        iter += 1
        current_progress+=1
        pbar.update(1)

    resultIndex = -1
    maxValue = float("-inf")
    fitness=np.vstack((fitness_v,fitness_u))

    population=np.vstack((population_v, population_u))
    popSize=population.shape[0]
    for p in range(0, popSize):
        if fitness[p, 1] <= budget and fitness[p, 0] > maxValue:
            maxValue = fitness[p, 0]
            resultIndex = p
    
    end_cpu = time.process_time()  
    end_wall = time.time() 

    cpu_time_used = end_cpu - start_cpu
    wall_time_used = end_wall - start_wall
            
    log = open(res_file+'/result_'+str(times)+'.txt', 'a')
    log.write("value = "+str(fitness[resultIndex,0]) + " cpu_time_used =" +str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + " cost = "+str(fitness[resultIndex,1])+" budget = "+str(budget)+ " population = "+str(popSize))
    
    log.write("\n")
    for item in Position(population[resultIndex,:]):
        log.write(str(item))
        log.write(" ")
    log.write("\n")
    log.close()
    pbar.close() 
       
def h(z, x_f,x_c):
    C = 100000
    if x_c > z['c_value']:
        return (x_f - z['f_value']) / ( x_c - z['c_value'])
    else:
        return (x_f - z['f_value']) * C + z['c_value'] -  x_c
    
def select(problem,P,fitness,refer_points):
    # Filter out empty sets
    non_empty_P=[]
    p_vs_fitness=[]
    for index, p in enumerate(P):
        if np.shape(p)[0] > 0:
            non_empty_P.append(p)
            p_vs_fitness.append(fitness[index])


    i = randint(0, len(non_empty_P)-1)
    P_i = non_empty_P[i]

    #Find the nearest reference point less than i
    list=[k for k in range(i) if refer_points[k]]
    #If the selected subpopulation is i=0, there is no reference point, and the returned solution is naturally zero.
    if not list:
        return P_i[randint(0, np.shape(P_i)[0] - 1):,]
    
    k = max(list)
    z = refer_points[k] 

    # Finding the S set
    # Initialize the maximum h value and the corresponding solution set
    max_h_value = float('-inf')
    S = []

    for index, x in enumerate(P_i):
        # Calculate the h value of the current solution
        current_h_value = h(z,p_vs_fitness[i][index,0], p_vs_fitness[i][index,1])
        
        # Check whether the h value of the current solution is the new maximum value
        if current_h_value > max_h_value:
            # Update the maximum h value
            max_h_value = current_h_value
            # Reset the solution set because a new maximum h value has been found
            S = [x]
        elif current_h_value == max_h_value:
            # If the h value of the current solution is the same as the known maximum h value, it is added to the solution set
            S.append(x)

    if 'point' in refer_points[i]:
        flag=False
        for matrix2 in S:
            if np.array_equal(refer_points[i]['point'], matrix2):
                flag=True
        if flag:
            s = refer_points[i]['point'] 

        else: 
            s=S[randint(0, len(S)-1)]

    else: 
        s=S[randint(0, len(S)-1)]

    if np.random.rand() < 0.5:
        x = s 
    else:
        x=P_i[randint(0, np.shape(P_i)[0]-1),:]
    

    return x

def local_search(problem,x):
    sum=0
    y = copy.deepcopy(x['point'])
    y_f = problem.FS(y)
    y_c = problem.CS(y)
    sum += 1
    for i in range(problem.n):
        if x['point'][0, i] == 0:
            s = copy.deepcopy(x['point'])
            s[0, i] = 1
            s_c=problem.CS(s)
            if s_c <= problem.budget:
                s_f = problem.FS(s)
                sum += 1
                if h(x, s_f,s_c) >= h(x, y_f, y_c):
                    y = s
                    y_f = s_f 
                    y_c = s_c
                    
    return y, y_f, y_c, sum

def FPOMC(res_file, times, T, GreedyEvaualte, problem):
    start_wall = time.time()  # Start wall-clock time measurement
    start_cpu = time.process_time()  # Start CPU time measurement


    n=problem.n

    subP_0 = np.mat(np.zeros((1, n), dtype='int8'))
    P = [[] for _ in range(n+1)]
    P[0]=subP_0

    subFitness_0=np.mat(np.zeros([1, 2]))
    fitness = [[] for _ in range(n+1)]
    fitness[0]= subFitness_0

    refer_points = [[] for _ in range(n+1)]
    refer_points[0]={'point':np.mat(np.zeros((1, n)), dtype='int8'),'f_value':0,'c_value':0}

    iter1 = 0
    best_times = 0
    best_value = 0

    nn = GreedyEvaualte
    totalTime=T* GreedyEvaualte
    
    pbar=tqdm(range(totalTime), position = 0, leave = True)
    current_progress = 0

    while  current_progress < totalTime:
        if iter1 >= nn:
            iter1 = 0
            resultIndex = -1
            subpopulation = -1
            maxValue = float("-inf")
            popSize_sum=0
            for P_index in range(n+1):
                popSize=np.shape(P[P_index])[0]
                popSize_sum+=popSize
                for p in range(0, popSize):
                    if fitness[P_index][p, 1] <= problem.budget and fitness[P_index][p, 0] > maxValue:
                        maxValue = fitness[P_index][p, 0]
                        resultIndex = p
                        subpopulation = P_index

            if fitness[subpopulation][resultIndex,0] == best_value:
                best_times += 1

            if best_times >= 10:
                break

            else:
                best_value =  fitness[subpopulation][resultIndex,0]
            
            end_cpu = time.process_time()  
            end_wall = time.time()  

            cpu_time_used = end_cpu - start_cpu
            wall_time_used = end_wall - start_wall
  

            log = open(res_file+'/result_'+str(times)+'.txt', 'a')
            log.write(" value = "+str(fitness[subpopulation][resultIndex,0]) + " cpu_time_used =" +str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + " cost = "+str(fitness[subpopulation][resultIndex,1])+ " budget = " + str(problem.budget) + " population = "+str(popSize_sum))
            #log.write("value = "+str(fitness[resultIndex,0]) + " cpu_time_used =" +str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + " cost = "+str(fitness[resultIndex,1])+" budget = "+str(budget)+ " population = "+str(popSize))
            
            log.write("\n")
            for item in Position(P[subpopulation][resultIndex,:]):
                log.write(str(item))
                log.write(" ")
            log.write("\n")
            log.close()          
            
        x = select(problem,P,fitness,refer_points)
        x_prime = mutation(n,x)  # every bit will be flipped with probability 1/n
        
        offSpringFit = np.mat(np.zeros([1, 2])) 
        offSpringFit[0, 1] = problem.CS(x_prime)
        if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > problem.budget:
            continue
        offSpringFit[0, 0] = problem.FS(x_prime)
        
        j = np.sum(x_prime)
        popSize_j = np.shape(P[j])[0]
        hasBetter = False
        for i in range(0, popSize_j):
            if (fitness[j][i, 0] > offSpringFit[0, 0] and fitness[j][i, 1] <= offSpringFit[0, 1]) or (
                    fitness[j][i, 0] >= offSpringFit[0, 0] and fitness[j][i, 1] < offSpringFit[0, 1]):
                hasBetter = True
                break

        if hasBetter == False:  # there is no better individual than offSpring
            Q = []
            for q in range(0, popSize_j):
                if offSpringFit[0, 0] >= fitness[j][q, 0] and offSpringFit[0, 1] <=fitness[j][q, 1]:
                    continue
                else:
                    Q.append(q)
        

            if np.shape(fitness[j])[0]==0:
                fitness[j] = offSpringFit
                P[j] = x_prime
            else:
                fitness[j] = np.vstack((offSpringFit, fitness[j][Q,:]))  # update fitness
                P[j] = np.vstack((x_prime, P[j][Q, :]))  # update population

            if not refer_points[j]:
                refer_points[j]={'point':x_prime,'f_value': offSpringFit[0, 0],'c_value': offSpringFit[0, 1]}
            else:
                k = max([k for k in range(j) if refer_points[k]])
                z = refer_points[k]
                value=h(z, offSpringFit[0, 0], offSpringFit[0, 1])
                if value >= h(z, refer_points[j]['f_value'], refer_points[j]['c_value']):
                    _,ff,cc,tt=local_search(problem,z)
                    # Record the number of evaluations used in LS
                    progress_increment = tt
                    iter1 += tt
                    current_progress+=tt
                    pbar.update(progress_increment)

                    if value >= h(z,ff,cc):
                        refer_points[j]={'point':x_prime,'f_value': offSpringFit[0, 0],'c_value': offSpringFit[0, 1]}
                        y,ff,cc,tt = local_search(problem,refer_points[j])
                            # Record the number of evaluations used in LS
                        progress_increment = tt
                        iter1 += tt
                        current_progress+=tt
                        pbar.update(progress_increment)

                        offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
                        offSpringFit[0, 1] = cc 

                        if offSpringFit[0, 1] <= problem.budget:
                            offSpringFit[0, 0] = ff
                            popSize_j_plus_1 = np.shape(P[j+1])[0] 

                            hasBetter = False
                            for i in range(0, popSize_j_plus_1):
                                if (fitness[j+1][i, 0] > offSpringFit[0, 0] and fitness[j+1][i, 1] <= offSpringFit[0, 1]) or (
                                        fitness[j+1][i, 0] >= offSpringFit[0, 0] and fitness[j+1][i, 1] < offSpringFit[0, 1]):
                                    hasBetter = True
                                    break
                            if hasBetter == False:  # there is no better individual than offSpring
                                Q = []
                                for q in range(0, popSize_j_plus_1):
                                    if offSpringFit[0, 0] >= fitness[j+1][q, 0] and offSpringFit[0, 1] <= fitness[j+1][q, 1]:
                                        continue
                                    else:
                                        Q.append(q)

                                if np.shape(fitness[j+1])[0]==0:
                                    fitness[j+1] = offSpringFit
                                    P[j+1] = y
                                else:
                                    fitness[j+1] = np.vstack((offSpringFit, fitness[j+1][Q,:]))  # update fitness
                                    P[j+1] = np.vstack((y, P[j+1][Q, :]))  # update population

        iter1 += 1
        current_progress+=1
        pbar.update(1)

    resultIndex = -1
    subpopulation = -1
    maxValue = float("-inf")
    popSize_sum=0
    for P_index in range(n+1):
        popSize=np.shape(P[P_index])[0]
        popSize_sum+=popSize
        for p in range(0, popSize):
            if fitness[P_index][p, 1] <= problem.budget and fitness[P_index][p, 0] > maxValue:
                maxValue = fitness[P_index][p, 0]
                resultIndex = p
                subpopulation = P_index

    
    end_cpu = time.process_time() 
    end_wall = time.time()  

    cpu_time_used = end_cpu - start_cpu
    wall_time_used = end_wall - start_wall


    log = open(res_file+'/result_'+str(times)+'.txt', 'a')
    log.write(" value = "+str(fitness[subpopulation][resultIndex,0]) + " cpu_time_used =" +str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + " cost = "+str(fitness[subpopulation][resultIndex,1])+ " budget = " + str(problem.budget) + " population = "+str(popSize_sum))
    #log.write("value = "+str(fitness[resultIndex,0]) + " cpu_time_used =" +str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + " cost = "+str(fitness[resultIndex,1])+" budget = "+str(budget)+ " population = "+str(popSize))
    
    log.write("\n")
    for item in Position(P[subpopulation][resultIndex,:]):
        log.write(str(item))
        log.write(" ")
    log.write("\n")
    log.close()     
    pbar.close() 
        
def sto_EVO_SMC(res_file, times, T, GreedyEvaualte,problem, epsilon=1e-10, prob=0):
    start_wall = time.time()  # Start wall-clock time measurement
    start_cpu = time.process_time()  # Start CPU time measurement


    n=problem.n
    budget=problem.budget

    population_F = np.mat(np.zeros([1, n], 'int8'))  
    population_G = np.mat(np.zeros([1, n], 'int8'))
    population_G_prime = np.mat(np.zeros([1, n], 'int8'))

    fitness_F = np.mat(np.zeros([1, 3]))
    fitness_G = np.mat(np.zeros([1, 3]))
    fitness_G_prime = np.mat(np.zeros([1, 3]))

    best_value = 0
    best_times = 0
    iter = 0
    nn = GreedyEvaualte
    totalTime=T* GreedyEvaualte
    
    pbar=tqdm(range(totalTime), position = 0, leave = True)
    current_progress = 0
    ell = 1
    w = 0
    H = ceil(2 * exp(1) * n * math.log(1/epsilon))

    while  current_progress < totalTime:
        if iter >= nn:
            iter = 0
            resultIndex = -1
            maxValue = float("-inf")
            fitness=np.vstack((fitness_F,fitness_G))
            fitness=np.vstack((fitness,fitness_G_prime))

            population=np.vstack((population_F, population_G))
            population=np.vstack((population,population_G_prime))
            popSize=population.shape[0]

            for p in range(0, popSize):
                if fitness[p, 1] <= budget and fitness[p, 0] > maxValue:
                    maxValue = fitness[p, 0]
                    resultIndex = p

            if fitness[resultIndex,0] == best_value:
                best_times += 1
            else:
                best_times = 0

            if best_times >= 10:
                break

            else:
                best_value =  fitness[resultIndex,0]
            
            end_cpu = time.process_time()  
            end_wall = time.time() 

            cpu_time_used = end_cpu - start_cpu
            wall_time_used = end_wall - start_wall
  

            log = open(res_file+'/result_'+str(times)+'.txt', 'a')
            log.write("value = "+str(fitness[resultIndex,0]) + " cpu_time_used =" +str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + " cost = "+str(fitness[resultIndex,1])+" budget = "+str(budget)+ " population = "+str(popSize))
            
            log.write("\n")
            for item in Position(population[resultIndex,:]):
                log.write(str(item))
                log.write(" ")
            log.write("\n")
            log.close()

        if(random.random()<prob):
            ell += 1
            popSize_G=population_G.shape[0]
            no_solution=True
            for i in range(popSize_G):
                if population_G[i,:].sum() == w:
                    no_solution = False
                    s = population_G[i,:]
            if no_solution:
                s = np.mat(np.zeros([1, n], 'int8'))
            
            if(ell >= H):
                ell = 0
                w +=1
        
        else:
            population = np.vstack((population_F, population_G))  # update fitness
            popSize=population.shape[0]
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly

        offSpring = mutation(n,s)  # every bit will be flipped with probability 1/n
        offSpringFit = np.mat(np.zeros([1, 3])) #f, c, g
        offSpringFit[0, 1] = problem.CS(offSpring)

        if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > budget:
            continue

        offSpringFit[0, 0] = problem.FS(offSpring)
        index = offSpring[0, :].sum()
        offSpringFit[0, 2] = 0 if index == 0 else 1.0 * offSpringFit[0, 0] / offSpringFit[0, 1]
    

        index = offSpring[0, :].sum()

        popSize_F=population_F.shape[0]
        no_solution=True

        for i in range(popSize_F):
            if population_F[i,:].sum() == index:
                no_solution=False
                if fitness_F[i, 0]<offSpringFit[0, 0]:
                    population_F[i,:]=offSpring
                    fitness_F[i, :]=offSpringFit
        if no_solution:
            population_F = np.vstack((population_F,offSpring))
            fitness_F = np.vstack((fitness_F,offSpringFit))
        
        popSize_G=population_G.shape[0]
        no_solution=True

        for i in range(popSize_G):
            if population_G[i,:].sum() == index :
                no_solution=False
                if fitness_G[i, 2]<offSpringFit[0, 2]:
                    population_G[i,:]=offSpring
                    fitness_G[i, :]=offSpringFit

                    Add_evaluation, solution_G_prime, G_value, G_cost = G_plus_greedy(population_G[i,:],problem)
                    current_progress+=Add_evaluation
                    iter += Add_evaluation
                    pbar.update(Add_evaluation)

                    popSize_G_prime=population_G_prime.shape[0]
                    no_solutionIn_Gprime=True
                    for j in range(popSize_G_prime):
                        if population_G_prime[j,:].sum() == index:
                            no_solutionIn_Gprime=False
                            if fitness_G_prime[j, 0]<G_value:
                                population_G_prime[j,:]=solution_G_prime
                                fitness_G_prime[j, 0]=G_value
                                fitness_G_prime[j, 1]=G_cost
                    if no_solutionIn_Gprime:
                        newFitness_f = np.mat(np.zeros([1, 3])) 
                        newFitness_f[0,0]=G_value
                        newFitness_f[0,1]=G_cost
                        newFitness_f[0,2]=0.0
                        population_G_prime = np.vstack((population_G_prime,solution_G_prime))
                        fitness_G_prime = np.vstack((fitness_G_prime,newFitness_f))

        if no_solution:
            population_G = np.vstack((population_G,offSpring))
            fitness_G = np.vstack((fitness_G,offSpringFit))



        iter += 1
        current_progress+=1
        pbar.update(1)


    resultIndex = -1
    maxValue = float("-inf")
    fitness=np.vstack((fitness_F,fitness_G))
    fitness=np.vstack((fitness,fitness_G_prime))

    population=np.vstack((population_F, population_G))
    population=np.vstack((population,population_G_prime))
    popSize=population.shape[0]

    for p in range(0, popSize):
        if fitness[p, 1] <= budget and fitness[p, 0] > maxValue:
            maxValue = fitness[p, 0]
            resultIndex = p

    
    end_cpu = time.process_time()  
    end_wall = time.time() 

    cpu_time_used = end_cpu - start_cpu
    wall_time_used = end_wall - start_wall


    log = open(res_file+'/result_'+str(times)+'.txt', 'a')
    #log.write(" value = "+str(fitness[resultIndex,0])+" cost = "+str(fitness[resultIndex,1])+ " budget = " + str(budget) + " population = "+str(popSize))
    log.write("value = "+str(fitness[resultIndex,0]) + " cpu_time_used =" +str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + " cost = "+str(fitness[resultIndex,1])+" budget = "+str(budget)+ " population = "+str(popSize))
    
    log.write("\n")
    for item in Position(population[resultIndex,:]):
        log.write(str(item))
        log.write(" ")
    log.write("\n")
    log.close()
    
    pbar.close() 

def G_plus_greedy(solution,problem):
    n=problem.n
    budget=problem.budget

    result = solution.copy()

    maxVolume=-1
    cor_cost=-1
    count=0
    selectedIndex = 0

    for j in range(0, n):
        if result[0,j]==0:
            result[0,j]=1
            cv = problem.CS(result)
            if cv > budget:
                result[0, j] = 0
                continue
            fv = problem.FS(result)
            count+=1
            tempVolume = fv 
            if tempVolume > maxVolume:
                maxVolume = tempVolume
                cor_cost = cv
                selectedIndex = j
            result[0, j] = 0
    result[0, selectedIndex] = 1

    return count,result,maxVolume,cv

def process_subPOMC(res_file, times, T, GreedyEvaualte, problem, index, value, cost) :
    n=problem.n
    budget=problem.budget

    if cost > budget:
        return None

    def new_func_f(x):
        return problem.FS(solution_plus_singleItem(x, index)) - value

    def new_func_c(x):
        return problem.CS(solution_plus_singleItem(x, index)) - cost
    
    solution, solution_value, solution_cost = sub_POMC(res_file, times, T, GreedyEvaualte, n, index, budget - cost, new_func_f, new_func_c, value, cost)
    solution[0, index] = 1
    f_value = solution_value + value
    c_cost = solution_cost + cost

    return index, solution, f_value, c_cost
 
def EPOMC(res_file, times, T, GreedyEvaualte, problem):
    start_wall = time.time()  # Start wall-clock time measurement
    start_cpu = time.process_time()  # Start CPU time measurement

    n=problem.n
    budget=problem.budget

    list_single_f=[]
    zero_solution = np.mat(np.zeros([1, n], 'int8'))
    best_solution = np.mat(np.zeros([1, n], 'int8'))
    f_best=problem.FS(best_solution)
    c_best=problem.CS(best_solution)
    id=None
     
    K_B=int(GreedyEvaualte/problem.n)
     
    for i in range(n):
        zero_solution[0, i] = 1

        value = problem.FS(zero_solution)
        cost = problem.CS(zero_solution)
        if cost< budget:
            list_single_f.append((i, value,cost))


        zero_solution[0,i]=0
  
    list_single_f = sorted(list_single_f, key=lambda x: x[1], reverse=True)
    list_single_f = list_single_f[:K_B]


    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_subPOMC, res_file, times, T, GreedyEvaualte, problem, index, value, cost) 
                        for index, value, cost in list_single_f]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                solution_id,solution, f_value, c_cost = result
                if f_value > f_best:
                    f_best = f_value
                    c_best = c_cost
                    best_solution = solution
                    id=solution_id
                    
                    end_cpu = time.process_time() 
                    end_wall = time.time() 

                    cpu_time_used = end_cpu - start_cpu
                    wall_time_used = end_wall - start_wall

                            
                    log = open(res_file+'/result_' + str(times)+'.txt', 'a')
                    log.write("value = "+str(f_best) +  " single_item = " + str(solution_id)+
                    " cpu_time_used =" + str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + 
                    " cost = "+str(c_best) +" budget = "+str(budget) )
                    log.write("\n")
                    for item in Position(solution):
                        log.write(str(item))
                        log.write(" ")
                    log.write("\n")
                    log.close()
    
def sub_POMC(res_file, times, T, GreedyEvaualte, n, index, new_budget, new_func_f, new_func_c, value, cost):
    start_wall = time.time()  # Start wall-clock time measurement
    start_cpu = time.process_time()  # Start CPU time measurement
    
    budget=new_budget
    population = np.mat(np.zeros([1, n], 'int8'))  # initiate the population
    fitness = np.mat(np.zeros([1, 2]))
    popSize = 1
    iter = 0
    best_times = 0
    best_value = 0
    nn = GreedyEvaualte
    totalTime=T* GreedyEvaualte
    
    pbar=tqdm(range(totalTime), position = 0, leave = True)
    current_progress = 0
    
    while  current_progress < totalTime:
        if iter >= nn:
            iter = 0
            resultIndex = -1
            maxValue = float("-inf")
            for p in range(0, popSize):
                if fitness[p, 1] <= budget and fitness[p, 0] > maxValue:
                    maxValue = fitness[p, 0]
                    resultIndex = p

            sub_file=res_file+ '/times'+str(times)
            if not os.path.exists(sub_file):
                os.makedirs(sub_file)

            if fitness[resultIndex,0] == best_value:
                best_times += 1
            else:
                best_times = 0

            if best_times >= 10:
                break

            else:
                best_value =  fitness[resultIndex,0]

            end_cpu = time.process_time()  
            end_wall = time.time() 

            cpu_time_used = end_cpu - start_cpu
            wall_time_used = end_wall - start_wall
                    
            log = open(sub_file+'/subPOMC_index_' + str(index)+'.txt', 'a')
            log.write("value = "+str(fitness[resultIndex,0]) + " total_value = "+str(fitness[resultIndex,0]+value) +  
            " cpu_time_used =" + str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + 
            " cost = "+str(fitness[resultIndex,1]) + " total_cost = " + str(fitness[resultIndex,1]+cost) +
            " budget = "+str(budget) +  " total_budget = " + str(budget+cost) + " population = "+str(popSize))
            log.write("\n")
            for item in Position(population[resultIndex,:]):
                log.write(str(item))
                log.write(" ")
            log.write("\n")
            log.close()
                
        s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
        offSpring = mutation(n,s)  # every bit will be flipped with probability 1/n
        offSpring[0, index]=0 # different from POMC
        
        offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
        offSpringFit[0, 1] = new_func_c(offSpring)
        if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > budget:
            continue
        offSpringFit[0, 0] = new_func_f(offSpring)
        hasBetter = False
        for i in range(0, popSize):
            if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                    fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                hasBetter = True
                break
        if hasBetter == False:  # there is no better individual than offSpring
            Q = []
            for j in range(0, popSize):
                if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                    continue
                else:
                    Q.append(j)
            # Q.sort()
            fitness = np.vstack((offSpringFit, fitness[Q, :]))  # update fitness
            population = np.vstack((offSpring, population[Q, :]))  # update population
        
        popSize = np.shape(fitness)[0]

        iter += 1
        current_progress+=1
        pbar.update(1)

    resultIndex = -1
    maxValue = float("-inf")
    for p in range(0, popSize):
        if fitness[p, 1] <= budget and fitness[p, 0] > maxValue:
            maxValue = fitness[p, 0]
            resultIndex = p

    sub_file=res_file+ '/times'+str(times)
    if not os.path.exists(sub_file):
        os.makedirs(sub_file)


    end_cpu = time.process_time()
    end_wall = time.time()  

    cpu_time_used = end_cpu - start_cpu
    wall_time_used = end_wall - start_wall

    log = open(sub_file+'/subPOMC_index_' + str(index)+'.txt', 'a')
    log.write("value = "+str(fitness[resultIndex,0]) + " total_value = "+str(fitness[resultIndex,0]+value) +  
    " cpu_time_used =" + str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + 
    " cost = "+str(fitness[resultIndex,1]) + " total_cost = " + str(fitness[resultIndex,1]+cost) +
    " budget = "+str(budget) +  " total_budget = " + str(budget+cost) + " population = "+str(popSize))
    log.write("\n")
    for item in Position(population[resultIndex,:]):
        log.write(str(item))
        log.write(" ")
    log.write("\n")
    log.close()
    pbar.close()

    return population[resultIndex,:], fitness[resultIndex,0], fitness[resultIndex,1]

def process_sub_P_POMC(res_file, times, T, GreedyEvaualte, n, index, new_budget, new_func_f, new_func_c, value, cost):
    solution, solution_value, solution_cost = sub_POMC(res_file, times, T, GreedyEvaualte, n, index, new_budget, new_func_f, new_func_c, value, cost)
    return index, solution, solution_value, solution_cost

def P_POMC(res_file, times, T, GreedyEvaualte, problem):
    start_wall = time.time()  # Start wall-clock time measurement
    start_cpu = time.process_time()  # Start CPU time measurement

    n=problem.n
    budget=problem.budget

    list_single_f=[]
    zero_solution = np.mat(np.zeros([1, n], 'int8'))
    best_solution = np.mat(np.zeros([1, n], 'int8'))
    f_best=0
    c_best=0
    id=None
     
    K_B=int(GreedyEvaualte/problem.n)
     
    for i in range(n):
        zero_solution[0, i] = 1

        value = problem.FS(zero_solution)
        cost = problem.CS(zero_solution)
        if cost< budget:
            list_single_f.append((i, value, cost))


        zero_solution[0,i]=0
  
    list_single_f = sorted(list_single_f, key=lambda x: x[1], reverse=True)
    list_single_f = list_single_f[:K_B]


    GreedyEvaualtes=[]
    for index in range(K_B):
        cost=list_single_f[index][2]
        new_K_B=problem.dp[n][int(budget-cost)]
        GreedyEvaualtes.append(n*new_K_B)
   
    num_of_processors=K_B
    with ProcessPoolExecutor(max_workers=num_of_processors) as executor:
        futures = [executor.submit(process_sub_P_POMC, res_file, times, T, GreedyEvaualtes[index], n, index, budget, problem.FS, problem.CS, 0, 0) 
                        for index in range(K_B)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                solution_id, solution, f_value, c_cost = result
                if f_value > f_best:
                    f_best = f_value
                    c_best = c_cost
                    best_solution = solution
                    id = solution_id
                                       
                    end_cpu = time.process_time() 
                    end_wall = time.time() 

                    cpu_time_used = end_cpu - start_cpu
                    wall_time_used = end_wall - start_wall

                            
                    log = open(res_file+'/result_' + str(times)+'.txt', 'a')
                    log.write("value = "+str(f_best) + " id = " +str(id) +
                    " cpu_time_used =" + str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + 
                    " cost = "+str(c_best) +" budget = "+str(budget) )
                    log.write("\n")
                    for item in Position(best_solution):
                        log.write(str(item))
                        log.write(" ")
                    log.write("\n")
                    log.close()
 