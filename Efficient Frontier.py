# -*- coding: utf-8 -*-
# @Time   : 11/15/18 3:51 PM
# @Author : Niyu Jia
# @Email  : nyjia@bu.edu

# ============================================================

"""
Harry Markowitz won the Nobel Prize for his work on understanding the risk-return trade-off in investments,
and specifically the Efficient Frontier.
"""
import numpy as np
import pandas as pd

def calc_portfolio_return(e, w):
    """
    calculates and returns the portfolio return (as a float) for a portfolio of n >= 2 assets. The parameters are:
    e is a matrix of expected returns for the assets
    w is a matrix of portfolio weights of the assets, which sums to 1.
    """

    r=np.dot(e,w.T)

    return float(r)

def calc_portfolio_MSR(rf,v,r):
    """
    give the w of maximum sharp ration portfolio
    :param rf: risk free rate
    :param v: cov matrix
    :param r: expected return of assets
    :return: weights
    """

    number=len(v)
    excess_r=r-rf
    print(excess_r)
    I=np.array([1]*number).reshape(number,1)
    numerator=(np.dot(v.I,excess_r))
    deno=np.dot(np.dot(I.T,v.I),excess_r)
    w=numerator/deno
    return w





def calc_portfolio_stdev(v, w):
    """
    calculates and returns the portfolio standard deviation (as a float) for a portfolio of n >= 2 assets. The parameters are:
    v is a matrix of covariances among the assets
    w is a matrix of portfolio weights of the assets, which sums to 1.
    """

    
    weight_matirx=np.dot(w.T,w)
    #print(weight_matirx)

    #weighted cov matrix
    std=np.multiply(weight_matirx,v)
    #print(std)

    #calculate standard deviation
    std=np.sum(std)**0.5
    return std



def calc_global_min_variance_portfolio(v):
    """
    That is, this function will find the portfolio with the absolute minimum variance that can be composed of the selected assets,
    where v is the matrix of covariances among the assets.
    """
    number=len(v)
    
    # reshape the ones matrix
    I=np.array([1]*number).reshape(number,1)
    C=float(np.dot(np.dot(I.T,v.I),I))
    #print(C)
    w=(np.dot(I.T,v.I))/C

    #acquire the final minimun var
    s=w.sum()

    return w/s


def calc_min_variance_portfolio(e, v, r):
    """
    finds and returns the portfolio weights corresponding to the minimum variance portfolio for the required rate of return r.
    The parameters are: e is a matrix of expected returns for the assets
    v is a matrix of covariances among the assets.
    r is the required rate of return
    """

    number=len(v)

    I=np.array([1]*number).reshape(number,1)
    e=e.reshape(number,1)

    #fllowing are the matrix that are needed to calculate the min var 
    # portfolio according to the file

    A=float(np.dot(np.dot(I.T,v.I),e))
    B=float(np.dot(np.dot(e.T,v.I),e))
    C=float(np.dot(np.dot(I.T,v.I),I))
    D=B*C-A*A

    g=np.dot((B*I.T-A*e.T),v.I)/D
    h=np.dot((C*e.T-A*I.T),v.I)/D

    #the final asset w:
    w=g+r*h

    return w



def calc_efficient_portfolios_stdev(e, v, rs):
    """
    finds a series of minimum variance portfolios and returns their standard deviations.
    The parameters are: e is a matrix of expected returns for the assets
    v is a matrix of covariances among the assets.
    rs is a numpy.array of rates of return for which to calculate the corresponding minimum variance portfolio’s standard deviation
    """

    # create empty sigma list
    sigmas=[]

    #loop for every rate of return
    for r in rs:
        w=calc_min_variance_portfolio(e,v,r)#get min var
        sigma=calc_portfolio_stdev(v,w)#get std according to the min var
        sigmas.append(sigma)#append sigma to the list

        print("r = %1.4f, sigma = %1.4f  w=%s" %(r,sigma,w))#formated print out

    return sigmas


def get_stock_prices_from_csv_files(symbols):
    """obtain a pandas.DataFrame containing historical stock prices for several stocks. The parameter symbols will be a
    list of stock symbols, and the return value will be a pandas.DataFrame containing the monthly stock prices for each of those stock symbols, for the period of dates given in the CSV files.

    """
    count=0
    # for each stock symbol, create the appropriate file name to read:
    for symbol in symbols:

        fn = './%s-monthly.csv' % symbol
        #print(fn)
        info=pd.read_csv(fn,parse_dates=['Date'])


	#acquire only two column
        info=info.loc[:,["Date","Adj Close"]]
	#change the column name
        info.columns=["Date",symbol]
        info_col=info[symbol]

        if count==0:#for the first file,we need the date info
            df=info
        else:
            df=df.join(info_col)# for other file,we only need the price info
        count+=1# count accumulation

    # beautify and formatize the dataframe
    df=df.set_index(df["Date"])#change the index to date
    df=df.drop(["Date"],axis=1)
    df.index.name = None
    return df



def get_stock_returns_from_csv_files(symbols):
    """
    which will return a single pandas.DataFrame object containing the stock returns.
    """

    prices=get_stock_prices_from_csv_files(symbols)#get stcok price
    index=prices.index

    #create lag price dataframe
    lag_prices=prices.shift(1)

    #change it to ndarray
    data=np.array(prices)
    lag_data=np.array(lag_prices)

    #print(data[0:5],"\n",lag_data[0:5])

    # use two array to calculate return in one time
    r=(data-lag_data)/lag_data
    # change the result to dataframe
    df=pd.DataFrame(data=r,index=index,columns=symbols)


    return df



def get_covariance_matrix(returns):
    """
    generates a covariance matrix for the stock returns in returns. The parameter return will be a pandas.DataFrame object,
    i.e., the same type as from the get_stock_returns_from_csv_files(symbols) function above.
    """
    # drop the first index line
    returns=returns.drop(returns.index[0])
    #returns=returns.fillna(0)

    r=np.array(returns).T
    #r=returns.as_matrix(columns=returns.columns).T
    #print(r)
    
    # create cov matrix with bias
    cov=np.cov(r,bias=True)

    #convert it to the dataframe
    covdf=pd.DataFrame(data=cov,columns=returns.columns,index=returns.columns)

    return covdf


################################## test of function #################################
if __name__=="__main__":
    
    e = np.matrix([0.1, 0.11, 0.08])
    w = np.matrix([1, 1, 1]) / 3
    print("test of 1")
    print(calc_portfolio_return(e, w))

    w = np.matrix([0.3, 0.5, 0.2])
    v = np.matrix([[0.2, 0.1, 0.5], [0.1, 0.1, 0.2], [0.5, 0.2, 0.3]])
    print("test of 2")
    print(calc_portfolio_stdev(v, w))
    
    v = np.matrix([[0.09, -0.03, 0.084], [-0.03, 0.04, 0.012], [0.084, 0.012, 0.16]])
    w = calc_global_min_variance_portfolio(v)
    print("global sulution")
    print(w)
    
    v = np.matrix([[0.09, -0.03, 0.084], [-0.03, 0.04, 0.012], [0.084, 0.012, 0.16]])
    r=np.matrix([[0.1],[0.09],[0.16]])

    w=calc_portfolio_MSR(0.02,v,r)
    print(w)

    exp=np.matrix([[0.1],[0.09],[0.16]])
    msr_ret=np.dot(w.T,exp)
    print(msr_ret)
    

    e = np.matrix([0.1, 0.11, 0.08])
    v = np.matrix([[0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.15]])
    w = calc_min_variance_portfolio(e, v, 0.09)
    print("restriction solution")
    print(w)

    rs = np.linspace(0.07, 0.12, 10)

    sigmas = calc_efficient_portfolios_stdev(e, v, rs)
    print(sigmas)

    #print(get_stock_prices_from_csv_files(["AAPL","DIS","KO"]))
    symbols = ['AAPL', 'DIS', 'GOOG', 'KO', 'WMT']
    returns=get_stock_returns_from_csv_files(symbols)
    covar = get_covariance_matrix(returns)
    print(covar)
    
