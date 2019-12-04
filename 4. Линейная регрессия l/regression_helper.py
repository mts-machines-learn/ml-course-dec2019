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
import math

        
        
font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)


    
X_LIM = 0.6
Y_LIM = 0.6


seed = 6942069
np.random.seed(seed)

def get_data():
    
    X = np.random.randint(10, 50, size=(20, )) / 100
    X.sort()
    
    k = 0.5 
    b = .1

    y = np.round(k*X + b + .045*np.random.normal(size=X.shape), 2)
    
    return X, y


def create_base_plot():
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10, 5), dpi=300)
    plt.xlabel("Площадь квартиры,\nквадратные метры", fontsize=22)
    plt.ylabel("Цена квартиры,\nмлн рублей", fontsize=22)
    plt.ylim([0, X_LIM])
    plt.xlim([0, Y_LIM])
    plt.grid()


def plot_data(X, y):
    create_base_plot()
    plt.scatter(X, y,  color='black', marker="o", s=50)   
    plt.show()

    
def visualize_X(X):
    print(pd.DataFrame(X, columns=['Площадь квартиры, кв. метры']))
    
    
def visualize_y(y):
    print(pd.DataFrame(y, columns=['Цена квартиры, млн. руб.']))
    
    
def plot_data_and_hyp(X, y, k):    
    create_base_plot()
    length = len(X)
    plt.plot(np.linspace(0, 1, length), k*np.linspace(0, 1, length), label="k={0}".format(k), color='black')   
    plt.scatter(X, y,  color='black', marker="o", s=50)  
    plt.legend(loc="upper left")   
    plt.show()

    
def choose_slope(X, y):
    k_slider = FloatSlider(min=0, max=2, step=0.1, value=0.1)

    @interact(k=k_slider)
    def interact_plot_data_and_hyp(k):
        plot_data_and_hyp(X, y, k)
    

def plot_data_and_error(X, y):
    k_slider = FloatSlider(min=0, max=2, step=0.1, value=0.1)

    @interact(k=k_slider)
    def plot_data_and_hyp_with_error(k):    
        create_base_plot()
        length = len(X)
        plt.plot(np.linspace(0, 1, length), k*np.linspace(0, 1, length), label="k={0}".format(k), color='black')   
        plt.scatter(X, y,  color='black', marker="o", s=50)  
        plt.legend(loc="upper left") 
    
        for x_i, y_i in zip(X, y):
            plt.plot([x_i, x_i], [x_i * k, y_i], color='red')
            
        plt.scatter(X, y,  color='black', marker="o", s=50)  
        plt.show()
        
        
def J(k, X, y):
    return np.mean((y - k*X)**2)


def plot_data_and_J(X, y):
    plt.rcParams.update({'font.size': 20})
    
    
    k_slider = FloatSlider(min=0, max=2, step=0.1, value=0.1)

    @interact(k=k_slider)
    def plot_data_and_hyp_with_error(k):    
        fig, axis = plt.subplots(1, 2, figsize=(18, 6))
    
        c = 'black'
        length = len(X)
        
        axis[0].plot(np.linspace(0, 0.6, length),  k*np.linspace(0, 0.6, length), label="k={0}".format(k), color=c)
        axis[0].set_title("Полученая линейная функция")
        for x_i, y_i in zip(X, y):
            axis[0].plot([x_i, x_i], [x_i * k, y_i], color='red')
        axis[0].scatter(X, y,  color='black', marker="o", s=50)  
        axis[0].set_xlabel("Дальность квартиры от метро, метры")
        axis[0].set_ylabel("Цена квартиры, млн рублей")
        axis[0].set_xlim(0, X_LIM)
        axis[0].set_ylim(0, Y_LIM)
        axis[0].legend()    
        axis[0].grid()

        axis[1].set_title("Значение ошибки\nдля гипотезы", fontsize=24)
        axis[1].set_ylabel("Значение функции потерь", fontsize=20)
        axis[1].set_xlabel("Значение коэффициента $k$")
        axis[1].scatter(k, J(k, X, y),  marker="+", label="J={0}".format( round(J(k, X, y), 3)), s=50, color=c)
        axis[1].set_xlim([-1, 3])
        axis[1].set_ylim([0, 0.15])
        axis[1].legend()    
       

        # We change the fontsize of minor ticks label 
        axis[1].tick_params(axis='both', which='major', labelsize=20)
        axis[1].tick_params(axis='both', which='minor', labelsize=20)
        axis[1].grid()
        plt.show()   
    
    
def plot_all_J(X, y):
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(8, 5))
    plt.title("Функция ошибки")
    plt.ylabel("Значение функции потерь")
    plt.xlabel("Значение коэффициента $k$")
    k = np.linspace(-1, 2.65, 100)
    plt.plot(k, [J(tmp_k, X, y) for tmp_k in k], color='black')
    plt.grid()
    plt.show()


def derivation(x0):
    d_slider = FloatSlider(min=-1, max=1.5, step=0.1, value=1.5, description='$\Delta x$')

    def f(x):
        return x**2 + 1.5

    def der_f(x):
        return 2*x


    @interact(dx=d_slider)
    def interact_plot_data_and_hyp(dx):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

        plt.ylim([-1, 11])
        plt.xlim([-4, 4])
        plt.grid()
        length = 100
        x = np.linspace(-3.5, 3.5, length)
        
        tg_alingment = 'left' if x0 > 0 else 'right'
        tg_position = -4 if x0 > 0 else 4

        x1 = x0+dx
        y0 = f(x0)
        y1 = f(x1)

        begin = -11
        end = 11
        
        
        tg_value = (y1-y0)/(dx) if dx != 0 else der_f(x0)
        tg_text = "$tg( \\alpha ) = \dfrac{\Delta y}{\Delta x} = $" + "{0:.2}".format(tg_value) if x0 != 0 else \
        "$tg( \\alpha ) = $" + "{0:.2}".format(float(tg_value))

        if x1 != x0:
            k = (y1-y0)/(x1-x0)
            b = y0 - k*x0
            if x0 != 0:
                plt.text( -b/k, -1, "$\\alpha$", ha='left', va='bottom')
            plt.plot([begin, end], [k*begin + b, k*end + b], color='black', linestyle='dashed')
            
            vpos = 'top' if y1 > y0 else 'bottom'
            hpos = 'left' if x1 > x0 else 'right'
            if abs(x1 - x0) > 1.5 or abs(y1 - y0) > 1.5:
                plt.text( x0 + (x1 - x0)/2, y0, "$\Delta x$", ha='center', va=vpos)
                plt.text( x1, y0 + (y1 - y0)/2, "$\Delta y$", ha=hpos, va='center')
        else:
            plt.plot([begin, end], [der_f(x0)*(begin-x0) + f(x0), der_f(x0)*(end-x0) + f(x0)], 
                     color='black', linestyle='dashed')
            if x0 != 0:
                plt.text((der_f(x0)*x0 - f(x0) )/ der_f(x0), -1, "$\\alpha$", ha='left', va='bottom')
        
      
        if x1 >= x0:
            plt.text(x1+0.1, -1, "$x_0 + \Delta x$", ha='left', va='bottom', fontsize=19)
        else:
            plt.text(x1+0.1, 0, "$x_0 + \Delta x$", ha='left', va='bottom', fontsize=19)
            
        if abs(y1-y0) < 0.7:
            plt.text(-3.2, y1+0.1, "$f(x_0 + \Delta x)$", ha='left', va='top', fontsize=19)
        else:
            plt.text(-4, y1+0.1, "$f(x_0 + \Delta x)$", ha='left', va='top', fontsize=19)
            
        
        plt.text(tg_position, -1, tg_text, 
                     ha=tg_alingment, va='bottom')

        plt.plot(x, f(x), color='black')   

        plt.plot([-5, x0], [y0, y0], color='red', linestyle='dotted')
        plt.plot([-5, x1], [y1, y1], color='blue', linestyle='dotted')
        plt.plot([x0, x0], [-5, y0], color='red', linestyle='dotted')
        plt.plot([x1, x1], [-5, y0], color='blue', linestyle='dotted')

        plt.text(-4, y0-0.1, "$f(x_0)$", ha='left', va='top')
        plt.text(x0-0.1, -1, "$x_0$", ha='right', va='bottom')

        plt.scatter([x0, x1], [y0, y1], color='black', marker="o", s=50) 

        plt.plot([x0, x1], [y0, y0], color='black', linestyle='dashed')
        plt.plot([x1, x1], [y0, y1], color='black', linestyle='dashed')

        plt.show()

        
def plot_func_and_der(f, der_f, same=True):
    x_slider = FloatSlider(min=-2, max=2, step=0.1, value=2, description='$x$')
    
    @interact(x0=x_slider)
    def plot_data_and_hyp_with_error(x0):    
        fig, axis = plt.subplots(1, 2, figsize=(18, 6), dpi=300)
    
        for ax in axis:
            ax.set_xlim(-4, 4)
            ax.grid()
            
        if same:
            axis[0].set_ylim(-11, 11)
        else:
            axis[0].set_ylim(0, 11)
        axis[1].set_ylim(-11, 11)
            
        length = 1000
        x = np.linspace(-12, 12, length)
        
        axis[0].plot(x, f(x), color='black')
        axis[1].plot(x, der_f(x), color='black')
        
        axis[0].scatter(x0, f(x0), color='black')
        axis[1].scatter(x0, der_f(x0), color='black')
        
        axis[0].set_title("$f(x) = x^2 + 1.5$")
        axis[1].set_title("$f'(x) = 2x$")
        
        axis[0].text(-4, -11 if same else 0, f"f({x0}) = {f(x0):.2}", ha='left', va='bottom')
        axis[1].text(4, -11, f"f'({x0}) = {der_f(x0):.2}", ha='right', va='bottom')
        
        begin = -10
        end = 20
        if x0 != 0:
            axis[0].plot([begin, end], [der_f(x0)*(begin-x0) + f(x0), der_f(x0)*(end-x0) + f(x0)], 
                     color='black', linestyle='dashed')
        else:
            axis[0].plot([begin, end], [f(0), f(0)], 
                     color='black', linestyle='dashed')
    
        plt.show() 
    
def plot_simple_func_and_der(same=True):
    
    def f(x):
        return x**2 + 1.5
    
    def der_f(x):
        return 2*x
    
    plot_func_and_der(f, der_f, same=same)
    

    

def der_J(X, y, k):
    return np.mean((k*X - y)*X)*2

def plot_all_J_with_der(X, y):
    k_slider = FloatSlider(min=0, max=2, step=0.1, value=0.1)

    @interact(k=k_slider)
    def plot_all_J_with_der_inner(k):
        plt.figure(figsize=(8, 5))
        plt.title("Значение ошибки")
        plt.ylabel("Значение функции потерь")
        plt.xlabel("Значение коэффициента $k$")
        ks = np.linspace(-1, 2.65, 100)
        plt.plot(ks, [J(tmp_k, X, y) for tmp_k in ks], color='black')
        plt.scatter(k, J(k, X, y), s=50, marker='+', label="Текущее значение")
        
        k_0 = (k - J(k, X, y) / der_J(X, y, k))
        k_next = 5 if k > k_0 else -5
        
        plt.plot([ k_0, k_next], [0, J(k, X, y) + der_J(X, y, k)*(k_next - k)], label="Текущее значение")
        plt.legend()
        plt.grid()
        plt.xlim([-1, 3])
        plt.ylim([0, 0.15])
        plt.show() 


    
def lin_grad_trace(X, y, k_init, alpha, iters=20):
    plt.figure(figsize=(10, 5))
    plt.title("Значение ошибки")
    plt.ylabel("Значение функции потерь")
    plt.xlabel("Значение коэфициента")
    T = np.linspace(-1, 2.6, 100)
    plt.plot(T, [J(t, X, y) for t in T], color='black')
    plt.scatter(k_init, J(k_init, X, y), color='green', marker="o", label="Начальная значение $k$")
    k = k_init
    for i in range(iters):
        tmp_k = k
        k = k - alpha * der_J(X, y, k)
        plt.plot([tmp_k, k], [J(tmp_k, X, y), J(k, X, y)], color='black', linestyle='dashed')
        if i == 0:
            plt.scatter(k, J(k, X, y), color='blue', marker="o", label="Промежуточные значения $k$")
        else:
            plt.scatter(k, J(k, X, y), color='blue', marker="o")
    plt.scatter(k, J(k, X, y), color='red', marker="o", label="Конечное значение $k$")
    plt.ylim([0, 0.1])
    plt.xlim([-0.5, 2])
    plt.legend(loc='best')
    plt.text(-0.5, 0, s="$k$="+f"{k:.4}", va='bottom', ha='left')
    plt.show()    
    return k

      
    
def Traice(X, y):
    k_init = FloatSlider(min=-0.1, max=1.8, step=0.1, value=0, description='$k$ init:')
    alpha = FloatSlider(min=0.1, max=12, step=0.1, value=0.1, description='$\\alpha$:', readout_format='.5f',)   

    @interact(k_init=k_init, a=alpha)
    def plot_data_and_hyp_with_error(k_init, a):    
        k = lin_grad_trace(X, y, k_init=k_init, alpha=a, iters=8)   

 
        
def lin_grad(X, y, k_init, alpha, iters=20):
    plt.title("Значение ошибки")
    plt.ylabel("Значение средней квадратичной ошибки")
    plt.xlabel("Значение коэфициента")
    T = np.linspace(175000, 196000, 100)
    plt.scatter(T, [J(t, X, y) for t in T], color='black', marker="+")
    
    plt.scatter(k_init, J(k_init, X, y), color='Yellow', marker="o")
    k = k_init
    for i in range(iters):
        k = k - alpha * der_J(X, y, k)
        plt.scatter(k, J(k, X, y), color='blue', marker="o")
    plt.scatter(k, J(k, X, y), color='red', marker="o")
    plt.show()
    
    return k
    
    
def get_new_data():
    
    #200000*X1 - 50000 * X2 + 5000000 + np.random.rand(20)*500000 - 1e6
    X1 = np.array([40, 27, 47, 20, 42, 56, 54, 33, 44, 74, 53, 33, 51, 58, 59, 20, 31, 44, 39, 51])
    X2 = np.array([95, 64, 68, 41, 71, 20, 85, 93, 42, 44, 50, 72, 29, 68,  3, 93,  8, 79, 85, 58])

    y = np.array([  7400000.,   6600000.,  10200000.,   6400000.,   9200000.,
        14400000.,  10700000.,   6000000.,  10700000.,  16900000.,
        12400000.,   7500000.,  13000000.,  12200000.,  16000000.,
         3500000.,  10300000.,   9300000.,   8000000.,  11400000.]) 
    return X1, X2, y

def linearn_loss_function(k, b, X, y):
    
    N = X.shape[0]       # получаем размер вектора столбца
    # или N = len(X)
   
    J = np.sum(((k*X + b) - y)**2)/N
    # или J = np.mean((k*X - y)**2)/N
    return J


def print_3d_table_with_data(X1, X2, y):
    l = len(X1)
    d = {"Площадь квартиры" : pd.Series(X1, index=range(0, l)), 
         "Расстояние до центра" : pd.Series(X2, index=range(0, l)),
         'Цена квартиры' : pd.Series(y, index=range(0, l))}
    df = pd.DataFrame(d)
    print(df)

def plot_new_3d_data(X1, X2, y):
    angles1 = IntSlider(min=0, max=180, step=1, value=0, description='Вертикальное')
    angles2 = IntSlider(min=0, max=180, step=1, value=0, description='Горизонтальное')

    @interact(angle1=angles1, angle2=angles2)
    def plot_new_data(angle1, angle2):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X1, X2, y/1e6, color="black")

        ax.set_xlabel('Площадь квартиры, квадратные метры')
        ax.set_ylabel('Растояние до центра Москвы, км')
        ax.set_zlabel('Цена квартиры, млн рублей')
        ax.view_init(angle1, angle2)

        plt.show() 
    
    
def J_full(k0, k1, k2, X1, X2, y):
    return sum((y - k1*X1 + k2*X2 + k0)**2) / (len(y)*2)
    
    
def plot_loss_in_3d(X1, X2, y):    
    angles1 = IntSlider(min=0, max=180, step=1, value=0, description='Вертикальное')
    angles2 = IntSlider(min=0, max=180, step=1, value=0, description='Горизонтальное')

    @interact(angle1=angles1, angle2=angles2)
    def plot_loss(angle1, angle2):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.linspace(0, 500000, 20)
        Y = np.linspace(-200000, 0, 20)
        X, Y = np.meshgrid(X, Y)

        Z = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(len(X)):
                Z[i, j] = J_full(5000000, X[i, j], Y[i, j], X1, X2, y)

        ax.set_xlabel('Площадь квартиры, квадратные метры')
        ax.set_ylabel('Растояние до центра Москвы, км')
        ax.set_zlabel('Цена квартиры, млн рублей')
        
        surf = ax.plot_surface(X, Y, Z,   linewidth=0, antialiased=False, cmap=cm.coolwarm)
        ax.view_init(angle1, angle2)
        plt.show()
    

def lin_grad_full(X1, X2, y, alpha, iters=20, k0_init=5000000, k1_init=0, k2_init=0):    
    angles1 = IntSlider(min=0, max=180, step=1, value=0, description='Вертикальное')
    angles2 = IntSlider(min=0, max=180, step=1, value=0, description='Горизонтальное')
    
    k1 = None
    k2 = None
    
    X = np.linspace(0, 500000, 20)
    Y = np.linspace(-200000, 0, 20)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X)):
            Z[i, j] = J_full(k0_init, X[i, j], Y[i, j], X1, X2, y)
    
    k1 = k1_init
    k2 = k2_init
    k0 = k0_init
    
    ks1 = [k1]
    ks2 = [k2]
    Js = [J_full(k0_init, k1, k2, X1, X2, y)]
    
    for i in range(iters):
        y_h = k1*X1 + k2*X2 + k0  -   y 
        k1 = k1 - alpha * sum(y_h * X1) / len(X1)
        k2 = k2 - alpha * sum(y_h * X2) / len(X2)
        ks1.append(k1)
        ks2.append(k2)
        Js.append(J_full(k0_init, k1, k2, X1, X2, y))
    
    @interact(angle1=angles1, angle2=angles2)    
    def plot_trace(angle1, angle2):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.set_xlabel('Площадь квартиры, квадратные метры')
        ax.set_ylabel('Растояние до центра Москвы, км')
        ax.set_zlabel('Цена квартиры, млн рублей')
        surf = ax.plot_wireframe(X, Y, Z,  cmap=cm.coolwarm)

        ax.scatter(ks1[0], ks2[0], Js[0], c="yellow")
        ax.scatter(ks1[1:-1], ks2[1:-1], Js[1:-1], c="blue")
        ax.scatter(ks1[-1], ks2[-1], Js[-1], c="Red")
        ax.view_init(angle1, angle2)
        plt.show()
        
    return k1, k2


def plot_new_data_and_hyp(X1, X2, y, k0, k1, k2):    
    angles1 = IntSlider(min=0, max=180, step=1, value=0, description='Вертикальное')
    angles2 = IntSlider(min=0, max=180, step=1, value=0, description='Горизонтальное')

    @interact(angle1=angles1, angle2=angles2)
    def plot_plane(angle1=angles1, angle2=angles2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X1, X2, y/1e6, color="black")

        ax.set_xlabel('Площадь квартиры, квадратные метры')
        ax.set_ylabel('Растояние до центра Москвы, км')
        ax.set_zlabel('Цена квартиры, млн рублей')
        xx, yy = np.meshgrid(np.linspace(0, 100, 20), np.linspace(0, 100, 20))
        z = np.zeros_like(xx)


        for i in range(len(xx)):
            for j in range(len(xx)):
                z[i, j] = k0 + k1*xx[i, j] + k2*yy[i, j]
        z = z / 1e6

        ax.plot_wireframe(xx, yy, z)
        ax.view_init(angle1, angle2)
        plt.show()
        
        

def get_poly_data():
    X = np.array([-1.849, -1.83,  -1.45 , -1.353, -1.104, -0.939, -0.222, -0.184,  0.333,  0.476,
                  0.486,  0.49,   0.542,  0.606,  0.634,  1.238,  1.36 ,  1.423,  1.511,  1.608])
    Y = np.array([ 6.15819037,  6.51804501,  0.59873171,  0.51437109, -0.07674737,  0.3879844 ,
                  1.2872504 ,  1.41004733,  1.38665749,  1.15472048,  1.01244111,  0.54954359,
                  0.07217831,  0.94735526,  0.18606682,  0.63916156,  1.31780108,  1.0590866 ,
                 -0.37102745,  1.99443298] )
    return X, Y 

def get_more_poly_data():
    X = np.array([-1.965, -1.933, -1.859, -1.782, -1.768, -1.643, -1.551, -1.543, -1.464, -1.432,
                 -1.42 , -1.42 , -1.409, -1.376, -1.369, -1.329, -1.315, -1.289, -1.104, -1.1  , -1.083,
                 -1.081, -1.069, -0.998, -0.934, -0.883, -0.836, -0.75 , -0.613, -0.584, -0.563,
                 -0.543, -0.432, -0.423, -0.371, -0.341, -0.328, -0.279, -0.214, -0.205, -0.186,
                 -0.131, -0.124, -0.091, -0.045, -0.016,  0.018,  0.057,  0.065,  0.099,  0.146,
                  0.229,  0.334,  0.346,  0.396,  0.398,  0.401,  0.472,  0.473,  0.506,  0.556,
                  0.565,  0.574,  0.587,  0.594,  0.597,  0.6  ,  0.626,  0.687,  0.695,  0.722,
                  0.725,  0.736,  0.743,  0.777,  0.823,  0.876,  0.998,  1.004,  1.005,  1.01 ,
                  1.112,  1.128,  1.236,  1.297,  1.314,  1.373,  1.405,  1.405,  1.513,  1.56, 
                  1.653,  1.688,  1.751,  1.756,  1.779,  1.85 ,  1.888,  1.904,  1.985] )
    Y = np.array([  8.28069778e+00,   7.68792987e+00,   6.62338200e+00,   4.67159991e+00,
                   4.48986359e+00,   2.87246601e+00,   2.14268602e+00,   1.55230438e+00,
                   8.43245397e-01,   6.26970943e-01,   1.18822929e+00,   1.98231365e+00,
                   1.25411579e+00,   9.42109132e-01,   1.27080005e+00,   6.08174143e-01,
                   1.69690246e+00,  -1.03835948e+00,   1.10258193e+00,  -6.39231733e-01,
                  -3.57375740e-01,   1.04270411e+00,   6.76625186e-01,   6.26952847e-01,
                   5.10433801e-01,   9.06599172e-02,   4.73783759e-01,   5.84644900e-01,
                   1.26923965e+00,   1.09318452e+00,   9.39945861e-01,   7.91994388e-01,
                   4.44194282e-02,   5.04393392e-01,   1.18098395e+00,   7.31776640e-01,
                  -4.91896638e-01,   6.70739056e-01,   1.89928651e+00,   2.21360719e+00,
                   1.27267694e-01,   6.54522248e-01,   1.16080350e+00,   8.60793906e-01,
                   1.42976614e+00,   2.08905455e+00,   1.01114096e+00,   5.91699878e-01,
                   9.64180759e-01,   2.08232350e+00,   1.25535727e+00,   8.38448838e-01,
                   8.23409487e-01,   2.33468834e-01,   6.97237660e-01,   6.26998895e-02,
                   1.06245266e+00,   1.28162688e-01,   1.13627574e+00,   4.19477706e-01,
                  -2.42651741e-01,   6.92200635e-03,   1.60973931e+00,   7.35610458e-01,
                   7.49141385e-01,  -7.09630553e-01,   4.32460929e-01,  -1.24700010e-02,
                   3.38846618e-01,   4.92559440e-01,   9.38345226e-01,  -8.93473918e-01,
                   8.00097249e-02,   2.91594741e-01,   1.00399074e+00,  -1.30473915e-01,
                   5.54628333e-01,  -6.59439660e-01,  -3.87959273e-01,  -1.87875999e-01,
                  -7.08778644e-01,  -2.82940805e-02,  -8.66548790e-02,  -5.32108189e-02,
                   9.88552450e-01,  -7.42980423e-01,  -2.56286254e-01,   6.19908155e-01,
                   8.37526981e-01,   1.37451984e+00,   2.11560025e+00,   2.18365693e+00,
                   3.28490407e+00,   3.82202829e+00,   5.19455980e+00,   5.28177855e+00,
                   7.21271110e+00,   6.74401676e+00,   7.19116823e+00,   9.11559289e+00] )
    return X, Y 

def plot_more_poly_data(X, y, X1=None, y1=None):
    plt.grid()
    plt.scatter(X, y,  color='black', marker="+")   
    if X1 is not None:
        plt.scatter(X1, y1, color='red', marker="+")   
    plt.ylim([-1, 8])
    plt.show()

    
def plot_poly_data(X, y, X1=None, y1=None):    
    n_deg = IntSlider(min=1, max=19, step=1, value=1, description='Степень полинома')
    @interact(pol=n_deg)
    def plot_plane(pol):
        coefs = np.polyfit(X, y, pol)
        p = np.poly1d(coefs)
        x_plot = np.linspace(-2, 2, 1000)
        plt.plot(x_plot, p(x_plot), "k-") 
        if X1 is not None:
            plt.scatter(X1, y1, color='red', marker="+")   
            print("Ошибка на новых данных = {0}".format(round(sum( (y1 - p(X1))**2) / (len(y)*2), 4) ))
        plt.scatter(X, y,  color='black', marker="+")  
        plt.ylim([-1, 8])
        print("Ошибка равна = {0}".format(round(sum( (y - p(X))**2) / (len(y)*2), 4) ))
        plt.show()
    

def plot_linear_loss_in_3d(X, y):    
    angles1 = IntSlider(min=0, max=180, step=1, value=0, description='Вертикальное')
    angles2 = IntSlider(min=0, max=180, step=1, value=45, description='Горизонтальное')

    @interact(angle1=angles1, angle2=angles2)
    def plot_loss(angle1, angle2):
        fig = plt.figure(figsize=(15, 10))
        ax = fig.gca(projection='3d')

        # Make data.
        ks = np.linspace(0, 200000, 20)
        bs = np.linspace(-20000000, 20000000, 20)
        ks, bs = np.meshgrid(ks, bs)

        Z = np.zeros_like(ks)
        for i in range(len(ks)):
            for j in range(len(bs)):
                Z[i, j] = linearn_loss_function(ks[i, j], bs[i, j], X, y)

        ax.set_xlabel('Значение параметра $k$')
        ax.set_ylabel('Значение параметра $b$')
        ax.set_zlabel('Функция ошибки')
        
        surf = ax.plot_surface(ks, bs, Z,   linewidth=0, antialiased=False, cmap=cm.coolwarm)
        ax.view_init(angle1, angle2)
        plt.show()
        
        

def lin_grad_linear(X, y, alpha, iters=20, k_init=0, b_init=0):    
    angles1 = IntSlider(min=0, max=90, step=1, value=0, description='Вертикальное')
    angles2 = IntSlider(min=0, max=180, step=1, value=0, description='Горизонтальное')
    
    k = None
    b = None
    
    ks = np.linspace(0, 500000, 20)
    bs = np.linspace(-200, 200, 20)
    ks, bs = np.meshgrid(ks, bs)
    Z = np.zeros_like(ks)
    for i in range(len(ks)):
        for j in range(len(ks)):
            Z[i, j] = linearn_loss_function(ks[i, j], bs[i, j], X, y)
    
    k = k_init
    b = b_init
    
    all_k = [k]
    all_b = [b]
    Js = [linearn_loss_function(k, b, X, y)]
    
    for i in range(iters):
        y_h = k*X + b -   y 
        k = k - alpha * 2*sum(y_h * X) / len(X)
        b = b - alpha * 2*sum(y_h) / len(X)
        all_k.append(k)
        all_b.append(b)
        Js.append(linearn_loss_function(k, b, X, y))
    
    @interact(angle1=angles1, angle2=angles2)    
    def plot_trace(angle1, angle2):
        fig = plt.figure(figsize=(15, 10))
        ax = fig.gca(projection='3d')

        ax.set_xlabel('Значение коэффициента $k$')
        ax.set_ylabel('Значение коэффициента $b$')
        ax.set_zlabel('Функция ошибка')
        surf = ax.plot_wireframe(ks, bs, Z,  cmap=cm.coolwarm)

        ax.scatter(all_k[0], all_b[0], Js[0], c="yellow")
        ax.scatter(all_k[1:-1], all_b[1:-1], Js[1:-1], c="blue")
        ax.scatter(all_k[-1], all_b[-1], Js[-1], c="Red")
        ax.view_init(angle1, angle2)
        plt.show()