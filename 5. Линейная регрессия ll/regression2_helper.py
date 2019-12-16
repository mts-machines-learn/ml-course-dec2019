import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn import datasets, linear_model
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ipywidgets import interact, IntSlider,  FloatSlider
from sklearn.datasets import load_boston    
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error


def load_all_data():
            
    font = {'family': 'Verdana', 'weight': 'normal'}
    rc('font', **font)
    plt.rcParams.update({'font.size': 22})
    boston_dataset = load_boston()
    y = boston_dataset['target']
    data = boston_dataset['data']
    return data, y

def load_small_data():
    data, y = load_all_data()
    return data[:, 5], data[:, -1], y 



def print_3d_table_with_data(X1, X2, y):
    l = len(X1)
    d = {"Cреднее количество комнат" : pd.Series(X1, index=range(0, l)), 
         "LSTAT %" : pd.Series(X2, index=range(0, l)),
         'Цена квартиры, $1000$' : pd.Series(y, index=range(0, l))}
    df = pd.DataFrame(d)
    print(df.head(17))

def plot_rm(X_room, y):
    plt.figure(figsize=(15, 9))
    plt.scatter(X_room, y, color="black")
    plt.xlabel('Cреднее количество комнат')
    plt.ylabel('Цена квартиры, 1000$')
    plt.grid()
    plt.show()

def plot_lstat(X_lstat, y):
    plt.figure(figsize=(15, 9))
    plt.scatter(X_lstat, y, color="black")
    plt.xlabel('LSTAT %')
    plt.ylabel('Цена квартиры, 1000$')
    plt.grid()
    plt.show()
    
    
def plot_new_3d_data(X1, X2, y):
    angles1 = IntSlider(min=0, max=180, step=1, value=45, description='Вертикальное')
    angles2 = IntSlider(min=0, max=180, step=1, value=45, description='Горизонтальное')

    @interact(angle1=angles1, angle2=angles2)
    def plot_new_data(angle1, angle2):
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X1, X2, y, color="black")

        ax.set_xlabel('Cреднее количество комнат')
        ax.set_ylabel('LSTAT %')
        ax.set_zlabel('Цена квартиры, $1000$')
        ax.view_init(angle1, angle2)

        plt.show()
        
        
def plot_new_3d_data_and_hyp(X1, X2, y):    
    angles1 = IntSlider(min=0, max=180, step=1, value=45, description='Вертикальное')
    angles2 = IntSlider(min=0, max=180, step=1, value=45, description='Горизонтальное')
    theta0 = IntSlider(min=-19, max=21, step=1, value=1)
    theta1 = IntSlider(min=-9, max=11, step=1, value=1)
    theta2 = IntSlider(min=-4, max=6, step=1, value=1)
    @interact(angle1=angles1, angle2=angles2,theta0=theta0, theta1=theta1, theta2=theta2)
    def plot_plane(angle1=angles1, angle2=angles2, theta0=theta0, theta1=theta1, theta2=theta2):
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X1, X2, y/1e6, color="black")

        ax.set_xlabel('Cреднее количество комнат')
        ax.set_ylabel('LSTAT %')
        ax.set_zlabel('Цена квартиры, $1000')
        xx, yy = np.meshgrid(np.linspace(2, 9, 100), np.linspace(0, 40, 100))
        z = np.zeros_like(xx)


        for i in range(len(xx)):
            for j in range(len(xx)):
                z[i, j] = theta0 + theta1*xx[i, j] + theta2*yy[i, j]
        z = z / 1e6

        ax.plot_wireframe(xx, yy, z)
        ax.view_init(angle1, angle2)
        plt.show()

def create_data(X1, X2):
    X_ones = np.ones_like(X1).reshape(-1,1)
    X = np.hstack((X_ones, X1.reshape(-1,1)))
    X = np.hstack((X, X2.reshape(-1,1)))
    return X

def gradient_descent(Theta, X, y, alpha, iters):        
    theta = Theta.copy()
    for i in range (iters):
        theta = theta - alpha * gradient_function(theta, X, y)
    return theta
        
def plot_new_3d_data_and_hyp_grad_des(X1, X2, y, theta0, theta1, theta2):  
    Theta = np.ones((3))
    X = create_data(X1, X2)
    angles1 = IntSlider(min=0, max=180, step=1, value=45, description='Вертикальное')
    angles2 = IntSlider(min=0, max=180, step=1, value=45, description='Горизонтальное')
    @interact(angle1=angles1, angle2=angles2)
    def plot_plane(angle1=angles1, angle2=angles2):
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X1, X2, y/1e6, color="black")

        ax.set_xlabel('Cреднее количество комнат')
        ax.set_ylabel('LSTAT %')
        ax.set_zlabel('Цена квартиры, $1000')
        xx, yy = np.meshgrid(np.linspace(2, 9, 100), np.linspace(0, 40, 100))
        z = np.zeros_like(xx)


        for i in range(len(xx)):
            for j in range(len(xx)):
                z[i, j] = theta0 + theta1*xx[i, j] + theta2*yy[i, j]
        z = z / 1e6

        ax.plot_wireframe(xx, yy, z)
        ax.view_init(angle1, angle2)
        plt.show()
        
        
        
#def plot_new_3d_data_and_hyp(X, y):    
#    angles1 = IntSlider(min=0, max=180, step=1, value=0, description='Вертикальное')
#    angles2 = IntSlider(min=0, max=180, step=1, value=90, description='Горизонтальное')
#    @interact(angle1=angles1, angle2=angles2)
#    def plot_plane(angle1=angles1, angle2=angles2):
#        fig = plt.figure(figsize=(15, 10))
#        ax = fig.add_subplot(111, projection='3d')
#
#        ax.scatter(X[:, 1], X[:, 2], y, color="black")
#
#        ax.set_xlabel('Cреднее количество комнат')
#        ax.set_ylabel('LSTAT %')
#        ax.set_zlabel('Цена квартиры, $1000')
#        
#        ax.view_init(angle1, angle2)
#        plt.show()

def linear_function(X, Theta):
    return np.dot(X, Theta) #X @ Theta


def MSE_Loss(X, Theta, y_true):
    y_pred = linear_function(X, Theta)
    return (np.sum((y_pred - y_true)**2))/(len(y_true))    

def gradient_function(Theta, X, y_true):
    grad = np.zeros_like(Theta)
    y_pred = linear_function(X, Theta)
    for j in range(Theta.shape[0]):       
        grad[j] = 2*np.mean((y_pred - y_true)* X[:,j])
    return grad
        
def plot_grad(X, Theta_init, y, a, k_min=0, k_max=3, skip=2):    
    iters_slider = IntSlider(min=0, max=500, step=1, value=1, description='Iter #')

    @interact(iters=iters_slider)
    def grad(iters):
        plt.figure(figsize=(10, 10))

        k1s = np.linspace(k_min, k_max, 100)
        k2s = np.linspace(k_min, k_max, 100)
        k1s, k2s = np.meshgrid(k1s, k2s)

        z = np.zeros_like(k1s)
        for i in range(len(k1s)):
            for j in range(len(k1s)):
                t = np.array([-10, k1s[i, j], k2s[i, j]])
                z[i, j] = MSE_Loss(X, t, y)

        lines = np.unique(np.round(z.ravel(), 3))
        lines.sort()
        ind = np.array([2**i for i in range(math.floor(math.log(len(lines), 2)) + 1)])

        plt.contour(k1s, k2s, z, lines[ind], cmap=cm.coolwarm)  # нарисовать указанные линии уровня

        Theta = Theta_init.copy()
        Thetas = [Theta.copy()]
        plt.xlabel("Значение $\\theta_1$")
        plt.ylabel("Значение $\\theta_2$")

        if iters > 0:
            for i in range(iters):
                g = a*gradient_function(Theta, X, y)
                Theta -= g
                Thetas.append(Theta.copy())
            Thetas = np.vstack(Thetas)

            plt.scatter(Thetas[:-1, 1], Thetas[:-1, 2], color='gray')
            plt.scatter(Thetas[iters-1, 1], Thetas[iters-1, 2], color='black')
            plt.text(k_max/2+k_min, k_max, "$\\alpha \\dfrac{Loss(\\Theta)}{\\theta_1} =  $" + f"{np.round(g[1], 5)}", va='top', ha='left')
            plt.text(k_max/2+k_min, k_max-k_max/6, "$\\alpha \\dfrac{Loss(\\Theta)}{\\theta_2} =  $" + f"{np.round(g[2], 5)}", va='top', ha='left')
        plt.show()

        
# ********************** Полином ****************************

def get_data_for_polynom():
    data = np.array([[ 5.000,  2.500], [16.600,  4.150],  [ 8.000, 12.450], [26.150,  7.950], [14.600,  9.350],
                    [40.400, 18.450], [21.200,  3.850], [23.500,  3.900], [38.200, 18.400], [25.050,  8.200],
                    [29.350, 12.700], [28.200, 10.150], [11.050, 10.850], [ 4.850, 13.350], [ 4.850, 12.400],
                    [ 3.850, 14.350], [27.850,  9.400], [29.550, 13.250], [29.150, 14.400], [15.400,  4.650],
                    [38.900, 18.600], [ 4.550, 15.600], [14.950,  2.550], [30.150, 15.600], [23.500,  6.900],
                    [31.150, 16.050], [32.500, 13.200], [ 5.200,  9.600], [33.050, 17.800], [12.950,  6.850],
                    [23.550,  5.700], [35.800, 18.850], [39.650, 18.550], [19.900,  2.200], [ 6.650, 12.200],
                    [16.700,  1.550], [ 3.550, 15.550], [34.450, 18.250], [ 2.600, 14.900], [27.800, 15.900]])
    
    X = data[:, 0]
    y = data[:, 1]
    
    return X, y



def get_more_data_for_polynom():
    data = np.array([[12.850,  8.050], [12.650,  5.150], [31.300, 17.250], [31.850, 17.600], [23.200,  1.250],
                     [ 9.400, 16.050], [35.150, 16.800], [28.100, 11.400], [10.550,  7.150], [11.800,  8.800],
                     [37.050, 17.750], [17.100,  3.000], [30.900,  9.450], [29.200, 15.150], [20.550,  2.800],
                     [18.200,  2.000], [ 5.900, 14.200], [14.550,  3.700]])
    X = data[:, 0]
    y = data[:, 1]
    
    return X, y



def plot_poly_data(X, y, X_test=None, y_test=None):
    font = {'family': 'Verdana', 'weight': 'normal'}
    rc('font', **font)
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 22})
    plt.scatter(X, y)
    if X_test is not None and y_test is not None:
        plt.scatter(X_test, y_test)
    
    plt.xlabel('Скорость')
    plt.ylabel('Расход топлива 1ой ступени')
    plt.grid()
    plt.show()
    
    
def visualize_prediction(X, y_train, y_pred):

    plt.figure(figsize=(10,10))
    plt.scatter(X, y_train)
    plt.xlabel('Скорость')
    plt.ylabel('Расход топлива 1ой ступени')
    plt.plot(X, y_pred, color='r')
    plt.grid()
    plt.show()
    
    
def plot_parabola():

    a_koef = FloatSlider(min = -5, max=10, step=0.5, value=1, description='a')
    b_koef = FloatSlider(min = -5, max=10, step=0.5, value=1, description='b')
    c_koef = FloatSlider(min = -5, max=10, step=0.5, value=1, description='c')
    
    @interact(a=a_koef, b=b_koef, c=c_koef)
    def interact_plot_parabol(a, b, c):
        x = np.linspace(-10, 10, num=200)
        y = a*x**2 + b*x + c


        plt.figure(figsize=(16,10))
        plt.plot(x, y, color='black')
        plt.xlim((-10,10))
        plt.ylim((-10,100))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.title("$y(X) = a X^2 + b X + c$")
        plt.show()
        
        
def plot_polynoms():
    x = np.linspace(-10, 20, 300)

    y3 = -1/6*x**3 + 9/6*x**2 - 3*x + 1
    y4 = 1/24*(x**4 - 16*x**3 + 72*x**2 - 96*x + 24)
    y5 = 1/120*(-x**5 + 25*x**4 - 200*x**3 + 600*x**2 - 600*x + 120)

    plt.figure(figsize=(16,10))
    plt.plot(x,y3, label="Полином 3 степени")
    plt.plot(x,y4, label="Полином 4 степени")
    plt.plot(x,y5, label="Полином 5 степени")
    # plt.xticks(ticks = x)
    plt.ylim((-25, 50))
    plt.xlim((-5, 15))
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
    

def plot_poly_results(X_poly_scaled, y_true, poly_transformer, scaler, regressor):

    x_axis_ticks = poly_transformer.transform((np.arange(0,41.3,0.01)).reshape(-1,1))
    x_axis_ticks = scaler.transform(x_axis_ticks)

    y_pred = regressor.predict(x_axis_ticks)

    plt.figure(figsize=(10,10))
    plt.scatter(X_poly_scaled[:, 1], y_true)
    plt.xlabel('Скорость')
    plt.ylabel('Расход топлива 1ой ступени')
    plt.ylim(min(y_true)-1, max(y_true)+1)
    plt.xlim(min(X_poly_scaled[:, 1]), max(X_poly_scaled[:, 1]))
    plt.grid()

    plt.plot(x_axis_ticks[:,1], y_pred, color='r')
    plt.show()
    
    
def interactive_polynom(X, y_true, X_test=None, y_test=None):  
    deg_slider = IntSlider(min=1, max=40, step=1, value=1, description='Степень полинома')
    
    @interact(deg=deg_slider)
    def p(deg):
        poly_transformer = PolynomialFeatures(deg)
        X_poly = poly_transformer.fit_transform(X.reshape(-1,1))

        scaler = StandardScaler()
        X_poly_scaled = scaler.fit_transform(X_poly)
        regressor = LinearRegression().fit(X_poly_scaled, y_true)
        y_d_pred = regressor.predict(X_poly_scaled)

        x_axis_ticks = poly_transformer.transform((np.arange(0, 41.3, 0.1)).reshape(-1,1))
        x_axis_ticks = scaler.transform(x_axis_ticks)
        y_pred = regressor.predict(x_axis_ticks)

        plt.figure(figsize=(10,10))
        plt.scatter(X, y_true)
        
        if X_test is not None and y_test is not None:
            plt.scatter(X_test, y_test)
            X_poly = poly_transformer.transform(X_test.reshape(-1,1))           
            X_poly_scaled = scaler.transform(X_poly)
            y_d_t_pred = regressor.predict(X_poly_scaled)
            plt.title(f'Ошибка на данных обучения = {mean_squared_error(y_d_pred, y_true):.2f}\nОшибка на новых данных = {mean_squared_error(y_d_t_pred, y_test):.2f}\n')
        else:
            plt.title(f'Ошибка = {mean_squared_error(y_d_pred, y_true):.2f}')
        
        plt.xlabel('Скорость')
        plt.ylabel('Расход топлива 1ой ступени')
        plt.ylim(0, 20)
        plt.grid()

        plt.plot(scaler.inverse_transform(x_axis_ticks)[:, 1], y_pred, color='r')
        plt.show()