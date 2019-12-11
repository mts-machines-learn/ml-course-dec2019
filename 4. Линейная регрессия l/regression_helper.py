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

    y = np.round(k*X + b + .03*np.random.normal(size=X.shape), 2)
    
    return X, y


def create_base_plot():
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10, 5), dpi=300)
    plt.xlabel("Значение X", fontsize=22)
    plt.ylabel("Значение Y", fontsize=22)
    plt.ylim([0, Y_LIM])
    plt.xlim([0, X_LIM])
    plt.grid()


def plot_data(X, y):
    create_base_plot()
    plt.scatter(X, y,  color='black', marker="o", s=50)   
    plt.show()

    
def visualize_Xy(X, y):
    print(pd.DataFrame(np.column_stack([X, y]), columns=['Значение X', 'Значение Y']))
    
    
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
        plt.scatter(X, k*X,  color='black', marker="x", s=50)  
        plt.legend(loc="upper left") 
    
        for x_i, y_i in zip(X, y):
            plt.plot([x_i, x_i], [x_i * k, y_i], color='red')
            
        plt.scatter(X, y,  color='black', marker="o", s=50)  
        plt.show()

def f(X, k):
    return k*X

def error_on_sample(X, y, k):
    for i in range(X.shape[0]): 
        diff = f(X[i], k) - y[i]
        print(f"Разница на примере {i} равна {diff:.4}")
        
def quad_error_on_sample(X, y, k):
    for i in range(X.shape[0]): 
        diff_quad = (f(X[i], k) - y[i])**2
        print(f"Квадрат разницы на примере {i} равен {diff_quad:.4}")       


def plot_data_and_loss(X, y, with_der=False):
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
        
        axis[1].scatter(k, J(X, y, k),  marker="+", label="$Loss({0})={1}$".format(k, round(J(X, y, k), 3)), s=50, color=c)
        if with_der: 
            der_label= "$\dfrac{d Loss(" + str(k) + ")}{dk}$=" + str(round(der_J(X, y, k), 3))
            axis[1].text(-1, 0, s=der_label, color=c, ha='left', va='bottom')
        axis[1].set_xlim([-1, 3])
        axis[1].set_ylim([0, 0.15])
        axis[1].legend()    
       

        # We change the fontsize of minor ticks label 
        axis[1].tick_params(axis='both', which='major', labelsize=20)
        axis[1].tick_params(axis='both', which='minor', labelsize=20)
        axis[1].grid()
        plt.show()  
    
    
def plot_all_loss(X, y):
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10, 5))
    plt.title("Функция ошибки")
    plt.ylabel("Значение функции потерь")
    plt.xlabel("Значение коэффициента $k$")
    k = np.linspace(-1, 2.65, 100)
    plt.plot(k, [J(tmp_k, X, y) for tmp_k in k], color='black')
    plt.ylim([0, min([J(k[0], X, y), J(k[-1], X, y)])])
    plt.grid()
    plt.show()


def derivation(x0):
    
    d_slider = FloatSlider(min=-1, max=1.5, step=0.10001, value=-1 if x0 < 0 else 1.5, description='$\Delta x$', readout_format='.2f')

    def f(x):
        return x**2 + 1.5

    def der_f(x):
        return 2*x
    

    
    @interact(dx=d_slider)
    def interact_plot_data_and_hyp(dx):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        
        if d_slider.value == 0.0001:
            d_slider.readout_format='.4f'
        else:
            d_slider.readout_format='.2f'
        

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
        tg_text = "$tg( \\alpha ) = \dfrac{\Delta y}{\Delta x} = $" + "{0:.2f}".format(tg_value) if x0 != 0 else \
        "$tg( \\alpha ) = $" + "{0:.2f}".format(float(tg_value))

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

        plt.scatter(x0, y0, color='red', marker="o", s=50) 
        plt.scatter(x1, y1, color='blue', marker="o", s=50) 

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
    
def plot_simple_func_and_der(same=False):
    
    def f(x):
        return x**2 + 1.5
    
    def der_f(x):
        return 2*x
    
    plot_func_and_der(f, der_f, same=same)


def der_J(X, y, k):
    return 2*np.mean((k*X - y)*X)

def J(X, y, k):
    return np.mean((k*X - y)**2)


def plot_loss_and_der(X, y, same=True):
    k_slider = FloatSlider(min=-2, max=2, step=0.1, value=2, description='$k$')
    
    @interact(k0=k_slider)
    def plot_data_and_hyp_with_error(k0):    
        fig, axis = plt.subplots(1, 2, figsize=(18, 6), dpi=300)
    
        for ax in axis:
            ax.set_xlim(-5, 6.5)
            ax.grid()
            
        
        axis[0].set_ylim(-1, 5)
        axis[1].set_ylim(-1, 1)
            
        length = 1000
        ks = np.linspace(-12, 12, length)
        
        axis[0].plot(ks, [J(X, y, k) for k in ks], color='black')
        axis[1].plot(ks, [der_J(X, y, k) for k in ks], color='black')
        
        axis[0].scatter(k0, J(X, y, k0), color='black')
        axis[1].scatter(k0, der_J(X, y, k0), color='black')
        
        axis[0].set_title("$\dfrac{1}{N} \sum_{i=1}^{N} (kX_i - y_i)^2$")
        axis[1].set_title("$\dfrac{2}{N} \sum_{i=1}^{N} (kX_i - y_i)X_i$")

        begin = -15
        end = 25
        if k0 != 0:
            axis[0].plot([begin, end], [der_J(X, y, k0)*(begin-k0) + J(X, y, k0), der_J(X, y, k0)*(end-k0) + J(X, y, k0)], 
                     color='black', linestyle='dashed')
        else:
            axis[0].plot([begin, end], [J(X, y, k0), J(X, y, k0)], 
                     color='black', linestyle='dashed')
    
        plt.show()    

def interactive_gradient_descent(X, y, iters=10):
    k_init = FloatSlider(min=-0.1, max=1.8, step=0.1, value=0.1, description='$k$ init:')
    alpha = FloatSlider(min=0.1, max=12, step=0.5, value=0.5, description='$\\alpha$:', readout_format='.1f',)   
    iteration = IntSlider(min=0, max=50, step=1, value=0, description='Iteration #:')  

    @interact(k_init=k_init, a=alpha, it=iteration)
    def plot_data_and_hyp_with_error(k_init, a, it):    
        fig, axis = plt.subplots(1, 2, figsize=(18, 6), dpi=300)
        
        for ax in axis:
            ax.grid()
            
        axis[0].set_title("Значение ошибки")
        axis[0].set_ylabel("Значение функции потерь")
        axis[0].set_xlabel("Значение коэфициента")
        
        
        T = np.linspace(-1, 2.6, 100)
        axis[0].plot(T, [J(X, y, t) for t in T], color='black')
        axis[0].scatter(k_init, J(X, y, k_init), color='green', marker="o", label="Начальная значение $k$")
        k = float(k_init)
        for i in range(it):
            tmp_k = k
            k = k - a * der_J(X, y, k)
            axis[0].plot([tmp_k, k], [J(X, y, tmp_k), J(X, y, k)], color='black', linestyle='dashed')
            if it > 1:
                if i == 0:
                    axis[0].scatter(k, J(X, y, k), color='gray', marker="o", label="Промежуточные значения $k$")
                else:
                    axis[0].scatter(k, J(X, y, k), color='gray', marker="o")
                
        length = len(X)
        axis[1].plot(np.linspace(0, 1, length), k*np.linspace(0, 1, length), label="k={0:.3f}".format(k), color='black')   
        axis[1].scatter(X, y,  color='black', marker="o", s=50)         
        axis[1].legend(loc="upper left")  
        
        
        if it > 0: axis[0].scatter(k, J(X, y, k), color='red', marker="o", label="Конечное значение $k$")
        axis[0].set_ylim([0, 0.1])
        axis[0].set_xlim([-0.5, 2])
        
        axis[1].set_xlim([0, 0.6])
        axis[1].set_ylim([0, 0.6])
        axis[1].set_title("J({0:.3f}) = {1:.4f}".format(k, J(X, y, k)))
        
        axis[0].legend(loc='best')
        axis[0].text(-0.5, 0, s="$k$="+f"{k:.4}", va='bottom', ha='left')
        plt.show()        

# ***********************************************************************************************************************

def plot_data_and_hyp_with_bias(X, y, k, b):    
    create_base_plot()
    length = len(X)
    plt.plot(np.linspace(0, 1, length), k*np.linspace(0, 1, length) + b, label="k={0:.4f}, b={1:.4f}".format(k, b), color='black')   
    plt.scatter(X, y,  color='black', marker="o", s=50)  
    plt.legend(loc="upper left")   
    plt.show()

    
def choose_slope_with_bias(X, y):
    k_slider = FloatSlider(min=0, max=2, step=0.1, value=0.1)
    b_slider = FloatSlider(min=-0.5, max=0.5, step=0.01, value=0.1)

    @interact(k=k_slider, b=b_slider)
    def interact_plot_data_and_hyp(k, b):
        plot_data_and_hyp_with_bias(X, y, k, b)

def der_f_3d(x, y):
    return np.array([4.5*x + 7.5, 5*y])

def f_3d(x, y):
    return (1.5*x + 2.5)**2 + 2.5*y**2 + 0.5

def plot_func_in_3d():    
    angles1 = IntSlider(min=0, max=90, step=1, value=0, description='Вертикальное')
    angles2 = IntSlider(min=0, max=90, step=1, value=0, description='Горизонтальное')

    @interact(angle1=angles1, angle2=angles2)
    def plot_loss(angle1, angle2):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')

        # Make data.
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        x, y = np.meshgrid(x, y)

        Z = np.zeros_like(x)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i, j] = f_3d(x[i, j], y[i, j])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('$\phi(x, y)$')
        
        surf = ax.plot_surface(x, y, Z,   linewidth=0, antialiased=False, cmap=cm.coolwarm)
        ax.view_init(angle1, angle2)
        plt.show()
        
def plot_3d_func_with_grad(x0=0, y0=0, pos_neg_grad=None):    

    fig = plt.figure(figsize=(15, 15))


    if pos_neg_grad is not None:
        grad = der_f_3d(x0, y0)

        ha = 'right' 
        va = 'top' 

        plt.scatter(x0, y0)
        plt.text(x0, y0, "$(x_0, y_0)$", ha='center', va="bottom")

        # left text with point and derivative
        plt.text(-10, 9.5, "$x_0 = " + "{0} $".format(x0), ha='left', va="center")
        plt.text(-10, 9, "$y_0 = " + "{0} $".format(y0), ha='left', va="center")
        plt.text(-10, 8, "$\dfrac{\delta \phi(x, y)}{\delta x} = 4.5x + 7.5 = " + "{0} $".format(grad[0]), ha='left', va="center")
        plt.text(-10, 6.5, "$\dfrac{\delta \phi(x, y)}{\delta y} = 5y = " + "{0} $".format(grad[1]), ha='left', va="center")
        
        
        col = 'gray'
        
        if pos_neg_grad in 'positive':
            col = 'black'     
            plt.text(x0 + grad[0], y0 + grad[1], "$(x_0 + \dfrac{\delta \phi(x, y)}{\delta x}, y_0 + \dfrac{\delta \phi(x, y)}{\delta y})$", ha='center', va="bottom", color=col)                
            plt.text(x0 + grad[0]/2, y0, "$\dfrac{\delta \phi(x, y)}{\delta x}$", ha='center', va='top')
            plt.text(x0, y0 + grad[1]/2, r"$\dfrac{\delta \phi(x, y)}{\delta y}$", va='center', ha="right")

        plt.arrow(x0, y0, grad[0], grad[1], color=col, length_includes_head=True, head_width=0.1)
        plt.arrow(x0, y0, grad[0], 0, length_includes_head=True, head_width=0.1, color=col)
        plt.arrow(x0, y0, 0, grad[1], length_includes_head=True, head_width=0.1, color=col)
        plt.plot([x0,           x0 + grad[0]], [y0 + grad[1], y0 + grad[1]], color=col, linestyle='dashed')
        plt.plot([x0 + grad[0], x0 + grad[0]], [y0,           y0 + grad[1]], color=col, linestyle='dashed')
        
        
        
        if pos_neg_grad in 'negative':

            plt.arrow(x0, y0, -grad[0], 0, length_includes_head=True, head_width=0.1)
            plt.arrow(x0, y0, 0, -grad[1], length_includes_head=True, head_width=0.1)
            plt.text(x0 - grad[0]/2, y0, "$-\dfrac{\delta \phi(x, y)}{\delta x}$", ha='center', va='bottom')
            plt.text(x0, y0 - grad[1]/2, r"$-\dfrac{\delta \phi(x, y)}{\delta y}$", va='center', ha="left")
            plt.plot([x0,           x0 - grad[0]], [y0 - grad[1], y0 - grad[1]], color='black', linestyle='dashed')
            plt.plot([x0 - grad[0], x0 - grad[0]], [y0,           y0 - grad[1]], color='black', linestyle='dashed')
            plt.text(x0 - grad[0], y0 - grad[1], "$(x_0 - \dfrac{\delta \phi(x, y)}{\delta x}, y_0 - \dfrac{\delta \phi(x, y)}{\delta y})$", ha='center', va="top", color='black')

            plt.arrow(x0, y0, -grad[0], -grad[1], color='r', length_includes_head=True, head_width=0.1)

    # Make data.
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    x, y = np.meshgrid(x, y)

    Z = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = f_3d(x[i, j], y[i, j])

    plt.xlabel('Значение параметра $x$')
    plt.ylabel('Значение параметра $y$')

    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.title("$\phi(x, y) = (1.5x + 2.5)^2 + 2.5y^2 + 0.5$")

    lines = np.unique(Z.flatten())
    lines.sort()

    ind = np.array([2**i for i in range(12)])
    plt.contour(x, y, Z, lines[ind], cmap=cm.coolwarm)  # нарисовать указанные линии уровня
    plt.grid()
    plt.show()


def plot_3d_func_with_grad_interactive():    
    
    x0_slider = FloatSlider(min=-10, max=10, step=0.1, value=0)
    y0_slider = FloatSlider(min=-10, max=10, step=0.1, value=1)
    a_slider = FloatSlider(min=0.01, max=1, step=0.01, value=1, description='$\\alpha$')

    @interact(x0=x0_slider, y0=y0_slider, a=a_slider)
    def plot_top(x0, y0, a):    

        fig = plt.figure(figsize=(15, 15))


        grad = a*der_f_3d(x0, y0)
        plt.text(-10, 9.5, "$x_0 = " + "{0:.3} $".format(x0), ha='left', va="center")
        plt.text(-10, 9, "$y_0 = " + "{0:.3} $".format(y0), ha='left', va="center")
        plt.text(-10, 8, "$\\alpha \dfrac{\delta \phi(x, y)}{\delta x} = 4.5x + 7.5 = " + "{0:.3} $".format(grad[0]), ha='left', va="center")
        plt.text(-10, 6.5, "$\\alpha \dfrac{\delta \phi(x, y)}{\delta y} = 5y = " + "{0:.3} $".format(grad[1]), ha='left', va="center")
        
        ha_pos = 'right' if x0 < 0.5 else 'left'
        va_pos = 'top' if y0 > 0 else 'bottom'
        
        ha_neg = 'left' if x0 < 0.5 else 'right'
        va_neg = 'bottom' if y0 > 0 else 'top'

        plt.scatter(x0, y0)
        
        if grad[0] != 0:
            plt.arrow(x0, y0, grad[0], 0, length_includes_head=True, head_width=0.1, color='gray')
            plt.arrow(x0, y0, -grad[0], 0, length_includes_head=True, head_width=0.1, color='black')
            
            if abs(grad[0]) < 10:
                plt.text(x0 + grad[0]/2, y0, "{0:.3}".format(grad[0]), ha='center', va=va_pos, color='gray')
                plt.text(x0 - grad[0]/2, y0, "{0:.3}".format(-grad[0]), ha='center', va=va_neg, color='black')
        if grad[1] != 0:
            
            if abs(grad[1]) < 10:
                plt.arrow(x0, y0, 0, grad[1], length_includes_head=True, head_width=0.1, color='gray')
                plt.arrow(x0, y0, 0, -grad[1], length_includes_head=True, head_width=0.1, color='black')
                
                plt.text(x0, y0 + grad[1]/2, "{0:.3}".format(grad[1]), va='center', ha=ha_pos, color='gray')
                plt.text(x0, y0 - grad[1]/2, "{0:.3}".format(-grad[1]), va='center', ha=ha_neg, color='black')
            
        if grad[0] and grad[1]:
            plt.plot([x0,           x0 + grad[0]], [y0 + grad[1], y0 + grad[1]], color='gray', linestyle='dashed')
            plt.plot([x0 + grad[0], x0 + grad[0]], [y0,           y0 + grad[1]], color='gray', linestyle='dashed')
            plt.plot([x0,           x0 - grad[0]], [y0 - grad[1], y0 - grad[1]], color='black', linestyle='dashed')
            plt.plot([x0 - grad[0], x0 - grad[0]], [y0,           y0 - grad[1]], color='black', linestyle='dashed')
        plt.arrow(x0, y0, grad[0], grad[1], color='gray', length_includes_head=True, head_width=0.1)
        plt.arrow(x0, y0, -grad[0], -grad[1], color='black', length_includes_head=True, head_width=0.1)
        
        
        
        plt.arrow(x0, y0, -grad[0], -grad[1], color='r', length_includes_head=True, head_width=0.1)

        
        
        plt.text(x0, y0, "$(x_0, y_0)$", ha='center', va="bottom")
        if abs(grad[0]) < 10 and abs(grad[1]) < 10:
            plt.text(x0 - grad[0], y0 - grad[1], "$(x_0 - \dfrac{\delta \phi(x, y)}{\delta x}, y_0 - \dfrac{\delta \phi(x, y)}{\delta y})$", ha='center', va="top", color='black')
        
        # Make data.
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        x, y = np.meshgrid(x, y)

        Z = np.zeros_like(x)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i, j] = f_3d(x[i, j], y[i, j])

        plt.xlabel('Значение параметра $x$')
        plt.ylabel('Значение параметра $y$')
        
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])

        lines = np.unique(Z.flatten())
        lines.sort()

        ind = np.array([2**i for i in range(12)])
        plt.contour(x, y, Z, lines[ind], cmap=cm.coolwarm)  # нарисовать указанные линии уровня
        plt.grid()
        plt.title("$\phi(x, y) = (1.5x + 2.5)^2 + 2.5y^2 + 0.5$")
        plt.show()

        
        
def linearn_loss_function(X, y, k, b):  
    return np.mean((k*X  + b  - y)**2)
        
def plot_linear_loss_in_3d(X, y):    
    angles1 = IntSlider(min=0, max=90, step=1, value=90, description='Вертикальное')
    angles2 = IntSlider(min=0, max=180, step=1, value=90, description='Горизонтальное')

    @interact(angle1=angles1, angle2=angles2)
    def plot_loss(angle1, angle2):
        fig = plt.figure(figsize=(15, 10))
        ax = fig.gca(projection='3d')

        k_min = -2
        k_max = 3

        b_min = -3
        b_max = 3

        ks = np.linspace(-2, 3, 40)
        bs = np.linspace(-1, 1, 40)
        ks, bs = np.meshgrid(ks, bs)

        Z = np.zeros_like(ks)
        for i in range(len(ks)):
            for j in range(len(bs)):
                Z[i, j] = linearn_loss_function(X, y, ks[i, j], bs[i, j])

        ax.set_xlabel('Значение параметра $k$')
        ax.set_ylabel('Значение параметра $b$')
        ax.set_zlabel('Функция ошибки')
        
        surf = ax.plot_surface(ks, bs, Z,   linewidth=0, antialiased=False, cmap=cm.coolwarm)
        ax.view_init(angle1, angle2)
        
        ax.set_xlim([-2, 3])
        ax.set_ylim([-1, 1])
        
        plt.show()


def plot_linear_loss_in_3d_up(X, y):    
    
    plt.figure(figsize=(15, 10))

    # Make data.
    
    k_min = -1
    k_max = 1

    b_min = -0.3
    b_max = 0.7
    
    ks = np.linspace(k_min, k_max, 60)
    bs = np.linspace(b_min, b_max, 60)
    ks, bs = np.meshgrid(ks, bs)

    Z = np.zeros_like(ks)
    for i in range(len(ks)):
        for j in range(len(bs)):
            Z[i, j] = linearn_loss_function(X, y, ks[i, j], bs[i, j])
   # Z = 
    plt.xlabel('Значение параметра $k$')
    plt.ylabel('Значение параметра $b$')

    lines = np.unique(np.round(Z.ravel(), 3))
    lines.sort()
    print(len(lines))
    ind = np.array([2**i for i in range(10)])
    plt.contour(ks, bs, Z, lines[ind], cmap=cm.coolwarm)  # нарисовать указанные линии уровня
    
    plt.xlim([-1, 1])
    plt.ylim([-.3, .7])
    plt.show()

    
def gradient_function(X, y, k, b):    
    return np.array([2*np.mean( ((k * X + b) - y) * X), 2*np.mean( ((k * X + b) - y))])

def plot_gradient_descent_in_3d(X, y, iters=5, alpha=0.15):    
    
    plt.figure(figsize=(15, 10))


    k = 0
    b = 0.5
    
    grad = alpha*gradient_function(X, y, k, b)
    ks = [k]
    bs = [b]
    gradients = [grad]
    for i in range(iters):  
        k -= grad[0]
        b -= grad[1]
        grad = alpha*gradient_function(X, y, k, b)
        gradients.append(grad)
        ks.append(k)
        bs.append(b)
    
    plt.scatter(ks[0], bs[0], s=50, color='g', label="Начальные значения")
    plt.scatter(ks[1:-1], bs[1:-1], s=50, color='gray', label="Промежуточные значения")
    plt.scatter(ks[-1], bs[-1], s=50, color='r', label="Конечныеные значения")
    
    # Make data.
    
    k_min = -1
    k_max = 1

    b_min = -0.3
    b_max = 0.7
    
    ks = np.linspace(k_min, k_max, 60)
    bs = np.linspace(b_min, b_max, 60)
    ks, bs = np.meshgrid(ks, bs)

    Z = np.zeros_like(ks)
    for i in range(len(ks)):
        for j in range(len(bs)):
            Z[i, j] = linearn_loss_function(X, y, ks[i, j], bs[i, j])

    plt.title('Функция ошибки')
    plt.xlabel('Значение параметра $k$')
    plt.ylabel('Значение параметра $b$')

    lines = np.unique(np.round(Z.ravel(), 3))
    lines.sort()
    ind = np.array([2**i for i in range(10)])
    plt.contour(ks, bs, Z, lines[ind], cmap=cm.coolwarm)  # нарисовать указанные линии уровня
    
    plt.legend()
    plt.xlim([-1, 1])
    plt.ylim([-0.3, 0.7])
    plt.show()


    
def plot_gradient_descent_in_3d_interactive(X, y, iters=5, alpha = 0.15):    

    i_slider = IntSlider(min=-1, max=iters, step=1, value=-1, description='iter')

    @interact(it=i_slider)
    def lin_loss(it):    

        fig, axis = plt.subplots(1, 2, figsize=(18, 6), dpi=300)

        k = 0
        b = 0.5

        grad = alpha*gradient_function(X, y, k, b)
        ks = [k]
        bs = [b]
        gradients = [grad]
        for i in range(iters):  
            k -= grad[0]
            b -= grad[1]
            grad = alpha*gradient_function(X, y, k, b)
            gradients.append(grad)
            ks.append(k)
            bs.append(b)
            
        axis[0].scatter(ks[0], bs[0], s=50, color='g', label="Начальные значения")
        
        
        if it >= 0:
            axis[0].scatter(ks[it], bs[it], s=50, color='blue', label="Текущие значения")
            if abs(gradients[it][0]) > 0.025 and abs(gradients[it][1]) > 0.025:
                axis[0].arrow(ks[it], bs[it], -gradients[it][0], -gradients[it][1], length_includes_head=True, head_width=0.01)
            else:
                axis[0].arrow(ks[it], bs[it], -gradients[it][0], -gradients[it][1])
            if it > 0:
                axis[0].scatter(ks[1:it], bs[1:it], s=50, color='gray', label="Промежуточные значения")
                

        k_i = it if it >= 0 else 0
        length = len(X)
        axis[1].plot(np.linspace(0, 1, length), ks[k_i]*np.linspace(0, 1, length) + bs[k_i], label="k={0:.3f}, b={1:.3f}".format(ks[k_i], bs[k_i]), color='black')   
        axis[1].scatter(X, y,  color='black', marker="o", s=50)         
        axis[1].legend(loc="upper left")   


        k_min = -1
        k_max = 1

        b_min = -0.3
        b_max = 0.7

        k_mesh = np.linspace(k_min, k_max, 60)
        b_mesh = np.linspace(b_min, b_max, 60)
        k_mesh, b_mesh = np.meshgrid(k_mesh, b_mesh)

        Z = np.zeros_like(k_mesh)
        for i in range(len(k_mesh)):
            for j in range(len(b_mesh)):
                Z[i, j] = linearn_loss_function(X, y, k_mesh[i, j], b_mesh[i, j])

        axis[0].set_xlabel('Значение параметра $k$')
        axis[0].set_ylabel('Значение параметра $b$')
        axis[1].set_title('Функция ошибки')

        lines = np.unique(np.round(Z.ravel(), 3))
        lines.sort()
        ind = np.array([2**i for i in range(10)])
        axis[0].contour(k_mesh, b_mesh, Z, lines[ind], cmap=cm.coolwarm)  # нарисовать указанные линии уровня

        axis[0].legend()
        axis[0].set_xlim([-1, 1])
        axis[0].set_ylim([-0.3, 0.7])
        axis[1].set_xlim([0, 0.6])
        axis[1].set_ylim([0, 0.6])
        axis[1].set_title("J({0:.3f}, {1:.3f}) = {2:.4f}".format(ks[k_i], bs[k_i], linearn_loss_function(X, y, ks[k_i], bs[k_i])))
        plt.show()