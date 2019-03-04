#Michael Groff

import numpy as np
import matplotlib.pyplot as plt
import time
import mlrose


if __name__=="__main__":
    print ("Knapsack Problem")
    bits = range(10,105,5)
    rhct = []
    rhco = []
    sat = []
    sao = []
    gat = []
    gao = []
    mmt = []
    mmo = []

    for i in bits:
        print(i)
        weights = list(np.random.randint(low =1, high = 20,size=i))
        values = list(np.random.randint(low =1, high = 5,size=i))
        fitness = mlrose.Knapsack(weights,values)
        popt = mlrose.DiscreteOpt(length=i,fitness_fn = fitness)

        t = time.clock()
        state , opt = mlrose.random_hill_climb(problem=popt, max_attempts=20)
        s = time.clock()
        rhct.append(s-t)
        rhco.append(opt)

        t = time.clock()
        state , opt = mlrose.simulated_annealing(problem=popt, max_attempts=20)
        s = time.clock()
        sat.append(s-t)
        sao.append(opt)

        t = time.clock()
        state , opt = mlrose.genetic_alg(pop_size = i*20, problem=popt, max_attempts=20)
        s = time.clock()
        gat.append(s-t)
        gao.append(opt)

        t = time.clock()
        state , opt = mlrose.mimic(pop_size = i*20, problem=popt, max_attempts=20)
        s = time.clock()
        mmt.append(s-t)
        mmo.append(opt)

    plt.plot(bits,rhct,bits,sat,bits,gat,bits,mmt)
    plt.title("Knapsack Problem")
    plt.xlabel("Bit State Length")
    plt.ylabel("Time")
    plt.legend(["rhc", "sa","ga","mimic"])
    plt.show()

    plt.plot(bits,rhco,bits,sao,bits,gao,bits,mmo)
    plt.title("Knapsack Problem")
    plt.xlabel("Bit State Length")
    plt.ylabel("Optimum Score")
    plt.legend(["rhc", "sa","ga","mimic"])
    plt.show()
