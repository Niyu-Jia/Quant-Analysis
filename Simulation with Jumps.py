import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import kurtosis,skew


def lam(t,c,l0,l1,miu_u,kappa_u,Vt,T_L,epsilon_L):
    Nt = len(T_L)
    tmTn = t - np.array(T_L)
    Ut = c + np.sum(np.exp(-tmTn * kappa_u)*miu_u*np.array(epsilon_L[:Nt]))
    return [l0 + l1 * Vt + Ut,Ut]


def simulate(T = 1, r = 0.03, rho = 30, kappa1 =6, niu1 = 0.02,sigma1 = 0.25, miu1 =12.16, l0 = 0, l1 =33, c=0,kappa_u = 0.09, miu_u = 3, S0 = 100, V1_0 = 0.02,simSteps =100):
    # initialize simulation
    dt = T/simSteps
    t_L = [0]; S_L = [S0]; V_L = [V1_0]  # t list record current time, X list record cumulative return, update at every moving
    A_L = [0]  # record the compensator process, update at every moving
    T_L = []  # record the jump Time, update only when there is a jump
    E_L = np.random.exponential(scale=1,size=simSteps*5000)  # generated random variable for jumps, no need to uodate
    s_L = np.cumsum(E_L)
    epsilon_L = np.random.exponential(scale=1/rho,size=simSteps*5000)
    Lambda=[]
    U=[]
    
    while t_L[-1] < T:
        # renew the variable
        At = A_L[-1]
        t = t_L[-1]; St = S_L[-1]; Vt = V_L[-1]
        Nt = len(T_L)  # the number of jumps that actually happened

        lamt= lam(t,c,l0,l1,miu_u,kappa_u,Vt,T_L,epsilon_L)[0]
        Ut=lam(t,c,l0,l1,miu_u,kappa_u,Vt,T_L,epsilon_L)[1]
        
        Atemp = At + dt * lamt
        
        Lambda.append(lamt)
        U.append(Ut)
        
        if Atemp < s_L[Nt]:
            dSt = (r + 1/rho * lamt)*St * dt + np.sqrt(Vt) * St * np.random.normal(loc=0, scale= np.sqrt(dt))
            # dSt = (r)*St * dt + np.sqrt(Vt) * St * np.random.normal(loc=0, scale= np.sqrt(dt))

            dVt = kappa1 * (niu1 - Vt) * dt + sigma1 * np.sqrt(Vt) * np.random.normal(loc=0, scale= np.sqrt(dt))

            S_L.append(St + dSt); V_L.append(max(Vt + dVt,0)); t_L.append(t + dt)
            A_L.append(Atemp)

        else:
            Tk = t + (s_L[Nt] - At) * dt/(lamt * dt)
            dt_temp = Tk - t

            dSt = (r + 1/rho * lamt)*St * dt_temp + np.sqrt(Vt) * St * np.random.normal(loc=0, scale= np.sqrt(dt_temp))
            # dSt = (r)*St * dt_temp + np.sqrt(Vt) * St * np.random.normal(loc=0, scale= np.sqrt(dt_temp))

            dVt = kappa1 * (niu1 - Vt) * dt_temp + sigma1 * np.sqrt(Vt) * np.random.normal(loc=0, scale= np.sqrt(dt_temp))
            St_ = St + dSt; Vt_ = Vt + dVt
            Atemp = At + dt_temp * lamt

            S_L.append(St_ + St_ * (-epsilon_L[Nt])); V_L.append(max(Vt_ + miu1 * epsilon_L[Nt]**2, 0)); t_L.append(Tk)
            A_L.append(Atemp); T_L.append(Tk)

    return t_L, S_L, V_L,Lambda,U


if __name__ == "__main__":
    M = 10; r = 0.03; rho = 30; kappa1 =6; niu1 = 0.02
    sigma1 = 0.25; miu1 =12.16; l0 = 0; l1 =33; c=0
    kappa_u = 0.09; miu_u = 3; S0 = 100; V1_0 = 0.02

    ############### (a) Plot a sample path
    t_L, S_L, V_L,Lambda,Ut= simulate(10, 0.03, 30, 6, 0.02, 0.25, 12.16, 0, 33, 0,0.09,  3, 100,  0.02,100)

    plt.plot(t_L,S_L)
    plt.title('S Sample Path')
    plt.show()
    
    plt.plot(t_L,V_L)
    plt.title('V Sample Path')
    plt.show()
    
    plt.plot(t_L[1:],Lambda)
    plt.title('Lambda Sample Path')
    plt.show()
    
    plt.plot(t_L[1:],Ut)
    plt.title('Ut Sample Path')
    plt.show()
    

    ###############################################
    #(b) distribution
    
    rho_list=[10,30,100]
    M=100
    S1=np.zeros((3,100))
    
    for i in range(100):
        for k in range(3):
            rho=rho_list[k]
            S1_path= simulate(1, 0.03, rho, 6, 0.02, 0.25, 12.16, 0, 33, 0,0.09,  3, 100,  0.02,100)[1]
            S1[k][i]=S1_path[-1]
    
    plt.hist(S1[0],bins=30)
    plt.title('S distribution with rho=10')
    
    plt.hist(S1[1],bins=30)
    plt.title('S distribution with rho=30')
    
    plt.hist(S1[2],bins=30)
    plt.title('S distribution with rho=100')
    
    
     ###############################################
     #(c)
    import multiprocessing
    import timeit
    M=100
    K=100 
    
    def option(k):
        rho=rho_list[k]
        price_list1=[0]*800
        price_list2=[0]*800
        for j in range(800):
            S1=[simulate(1, 0.03, rho, 6, 0.02, 0.25, 12.16, 0, 33, 0,0.09,  3, 100,  0.02,100)[1][-1] for m in range(80)]
            S2=[simulate(1, 0.03, rho, 6, 0.02, 0.25, rho, 0, 33, 0,0.09,  3, 100,  0.02,100)[1][-1] for m in range(80)]
            price1=[max(0,m-K) for m in S1]
            price2=[max(0,m-K) for m in S2]
            
            price_list1[j]=np.mean(price1)
            price_list2[j]=np.mean(price2)
                    
        return price_list1,price_list2
    
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    k = range(3)
    
    start = timeit.default_timer()
    # method 1: map
    optionprice=pool.map(option,k)  
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    ###########################################################
    
    #(d)
    def option2(k):
        rho=rho_list[k]
        S1=[0]*100
        price_list=[0]*800
        for j in range(800):
            S1=[simulate(1, 0.03, rho, 6, 0.02, 0.25, rho, 0, 33, 0,0.09,  3, 100,  0.02,100)[1][-1] for m in range(100)]
            avg_S=np.mean(S1)
            price=max(0,avg_S-K)
            price_list[j]=price
        
        return price_list
    
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    k = range(3)
    
    start = timeit.default_timer()
    # method 1: map
    optionprice2=pool.map(option2,k)  # prints [0, 1, 4, 9, 16]
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    
    ###################################################
    #plot histogram 
    rho1=optionprice[0]
    rho2=optionprice[1]
    rho3=optionprice[2]
    
    for i in range(3):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data=optionprice[i][1]
        ax.hist(data, weights=np.zeros_like(data) + 1. / len(data),bins=35)
        plt.title("density histogram of rho=u_l="+str(rho_list[i]))
