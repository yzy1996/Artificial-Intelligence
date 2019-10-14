import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
from scipy import optimize

sampx = np.loadtxt("PA-1-data-text/polydata_data_sampx.txt")
sampy = np.loadtxt("PA-1-data-text/polydata_data_sampy.txt")
polyx = np.loadtxt("PA-1-data-text/polydata_data_polyx.txt")
polyy = np.loadtxt('PA-1-data-text/polydata_data_polyy.txt')

# 画散点图
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.scatter(sampx, sampy,s=10)
# plt.show()

def feature_transformation(x,k):
    X = []
    for n in x:
        a = []
        for i in range(k+1):
            a.append(math.pow(n,i))
        X.append(a)
    return np.mat(X)


def least_squares(x,y,k):
    X = np.mat(x).reshape(-1,k)
    Y = np.mat(y).reshape(-1,1)
    theta = np.dot(np.dot(inv(np.dot(X.T,X)),X.T),Y)
    return theta

def regularized_LS(x,y,k,l):
    X = np.mat(x).reshape(-1, k)
    Y = np.mat(y).reshape(-1, 1)
    n = np.mat(l*np.identity(k))
    theta = np.dot(np.dot(inv(np.dot(X.T,X) + n),X.T),Y)
    return theta

def L1_regularized_LS(x,y,k,l):
    X = np.mat(x).reshape(-1, k)
    Y = np.mat(y).reshape(-1, 1)
    theta = np.dot(inv(np.dot(X.T,X)),np.dot(X.T,Y)+l*np.identity(k))
    return theta

def robust_regression(x,y,k):
    X = np.mat(x).reshape(-1, k+1)
    Y = np.mat(y).reshape(-1, 1)
    n = Y.shape[0]
    f = np.zeros(k+1)
    one = np.ones(n)
    f = np.mat(np.append(f,one))
    a1 = np.hstack((-X,-np.mat(np.identity(n))))
    a2 = np.hstack((X,-np.mat(np.identity(n))))
    A = np.vstack((a1,a2))
    b = np.vstack((-Y,Y)).reshape(-1,1)
    bounds=(None)
    res = optimize.linprog(f,A,b,bounds)
    return res

def Bayesian_regression(x,y,k,alpha):
    X = np.mat(x).reshape(-1, k)
    Y = np.mat(y).reshape(-1, 1)
    sigma = np.mat(alpha*np.identity(k))
    Sigma = inv((1/alpha*np.identity(k)+ (1/sigma**2)*np.dot(X.T,X)))
    mu = (1/sigma**2)*Sigma*np.dot(X.T,Y)
    return Sigma,mu


if __name__ == '__main__':
    x = feature_transformation(sampx,5)
    y = sampy
    
    theta = robust_regression(x,y,5).x[0:6].reshape(-1,1)
    print(theta)
    # theta = least_squares(x,y,5)
    # theta = regularized_LS(x,y,5,0.01)

    x_input = feature_transformation(polyx,5)
    Y = x_input*theta
    # sigma, mu = Bayesian_regression(x,y,5,0.01)
    # x_input = feature_transformation(polyx,5)
    # theta = mu
    # mu_predict = np.dot(x_input,mu)
    # sigma_predicet = np.dot(np.dot(x_input,sigma),x_input.T)
    # Y = 1/np.sqrt(2*math.pi*sigma_predicet**2)*pow(math.e,(-1/2*sigma_predicet**2)*(x_input*theta-mu_predict)**2)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('BR estimation')
    ax1.plot(polyx,Y)
    ax1.scatter(sampx,sampy,s=10)
    plt.show()