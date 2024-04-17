import numpy as np
import math
import matplotlib.pyplot as plt

x_0 = np.array([-4,-0.5])
w_c_0 = 0.5 * np.ones((3,1))
Tau_0 = 100 * np.identity(3)
w_a_0 = 0.7 * w_c_0
k_c1 = 0.001
k_c2 = 0.005
k_a1 = 0.12
k_a2 = 0.001
beta = 0.003
gamma_1 = 0.05
gamma_2 = 1
d_t = 1e-5
T = 1000000
u_ = []
w_a = np.zeros((3,T+1))
w_c = np.zeros((3,T+1))
x = np.zeros((2,T+1))
Tau = np.zeros((3,3,T+1))

Q, R = np.identity(2),1


p = 0.01
Tau[:,:,0] = Tau_0 
w_a[:,0] = w_a_0.reshape(-1)
w_c[:,0] = w_c_0.reshape(-1)
x[:,0] = x_0

pi = 0.5
mu = 0.5
du = 3


def v_0(x):
    return ((np.linalg.norm(x))**2+0.01)/(1+gamma_2*(np.linalg.norm(x))**2)
def sample(x):
    v0 = v_0(x)
    x1 = v0*np.random.rand()
    x2 = v0*np.random.rand()
    return np.array([x[0] + x1, x[1] + x2])


def f(x):#[−0.6x1 − x2, x3 1]T 
    fx = [-0.6*x[0] - x[1],x[0]**3]
    # fx1 = -x[0] + x[1]
    # fx2 = -0.5 * x[0] - 0.5 * x[1]*(math.cos(2*x[0])+2)**2
    return np.array(fx)

def g(x):
    return np.array([0,x[1]])


def d(x):
    v0 = v_0(x)
    d1 = 0.7 * v0 * np.array([0,1])
    d2 = 0.7 * v0 * np.array([0.87,-0.5])
    d3 = 0.7 * v0 * np.array([-0.87,-0.5])
    return d1,d2,d3

def c(x):
    d1,d2,d3 = d(x)
    c1 = x + d1
    c2 = x + d2 
    c3 = x + d3 
    return c1,c2,c3
def h1(x):
    return -x[1]**2 - x[0] + 1 
def h2(x):
    return x[1]**2 - x[0] + 1 

def delta_h1(x):
    return np.array([-1,-2*x[1]])

def delta_h2(x):
    return np.array([-1,2*x[1]])


def hh1(x):
    return -x**2 + 1 
def hh2(x):
    return x**2 + 1 
yy = np.linspace(-2, 2, 100)
xx = hh1(yy)

def kernel(x):
    c1,c2,c3 = c(x)
    sigma_1 = math.exp(np.dot(x,c1)) -1 
    sigma_2 = math.exp(np.dot(x,c2)) -1 
    sigma_3 = math.exp(np.dot(x,c3)) -1 
    return np.array([sigma_1,sigma_2,sigma_3])
#kernel(np.array([1,2]))

def delta_d(x):
    a = (2-0.02*gamma_2)/(1+gamma_2*np.linalg.norm(x))**2
    delta_d_1 = np.array([[0, 0],
                          [0.7*a*x[0], 0.7*a*x[1]]])
    delta_d_2 = np.array([[0.87*a*x[0], 0.87*a*x[1]],
                          [(-0.5)*a*x[0], (-0.5)*a*x[1]]])
    delta_d_3 = np.array([[(-0.87)*a*x[0], (-0.87)*a*x[1]],
                          [(-0.5)*a*x[0], (-0.5)*a*x[1]]])
    
    return delta_d_1, delta_d_2, delta_d_3


def delta_sigma_v(x):## 行向量乘法
    d1,d2,d3 = d(x)
    x = x/np.linalg.norm(x)
    d1,d2,d3 = d1/np.linalg.norm(d1),d2/np.linalg.norm(d2),d3/np.linalg.norm(d3)

    delta_d_1, delta_d_2, delta_d_3 = delta_d(x)
    e_1x = math.exp(np.dot(x,x+d1)) -1 
    #print(e_1x)
    e_2x = math.exp(np.dot(x,x+d2)) -1 
    e_3x = math.exp(np.dot(x,x+d3)) -1 
    delta_sigma_1 = e_1x*(2*x+d1+np.dot(x,delta_d_1))
    delta_sigma_2 = e_2x*(2*x+d1+np.dot(x,delta_d_2))
    delta_sigma_3 = e_3x*(2*x+d1+np.dot(x,delta_d_3))

    return np.array([delta_sigma_1,delta_sigma_2,delta_sigma_3])


def hjb_error(x, u_ET,w_a_t, w_c_t):
    delta_sigma = delta_sigma_v(x)
   
    omega_ = f(x) + u_ET * g(x)
    omega = np.dot(delta_sigma,omega_)

    error = x.T @ Q @ x + u_ET * R * u_ET + np.dot(np.dot(delta_sigma.T,w_c_t).T,
                                                      omega_)
    
    #compare = delta_h1(x) @ g(x) * u_ET + 0.1*h2(x) +  delta_h1(x) @ f(x)

    rho = math.sqrt(1 + gamma_1 * (np.linalg.norm(omega))**2)
    # u_au = (-compare/np.linalg.norm(delta_h1(x) @ g(x))**2) * (delta_h1(x) @ g(x)) #
    
    return error , omega ,rho
    # if compare >=0:
    #     return error , omega ,rho, u
    # else:
    #     print("uuuuuuu")
    #     return error , omega ,rho, u+u_au


def u_x(x,w_a_t):
    delta_sigma = delta_sigma_v(x)
    tmp = np.dot(g(x), np.dot(delta_sigma.T,w_a_t))
    u = - 0.5 *  tmp
    # omega_ = f(x) + u * g(x)
    # omega = np.dot(delta_sigma,omega_)
    compare = delta_h1(x) @ g(x) * u + h1(x) +  delta_h1(x) @ f(x)
    u_au = (-compare/np.linalg.norm(delta_h1(x) @ g(x))**2) * (delta_h1(x) @ g(x)) #
    
    #return u
    if compare >=0:
        return u
    else:
        # print("uuuuuuu")
        # print(u,u_au)
        return u + u_au
  


def g_sigma(x):
    delta_sigma = delta_sigma_v(x)
    return (np.linalg.norm(delta_sigma @ g(x)))**2
    
def plot_phase(x1_values,x2_values):
    plt.figure(figsize=(8, 6))
    plt.plot(x1_values, x2_values, label='Phase Portrait')
    plt.plot(xx,yy,label = r'$\partial C$')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Phase Portrait of x1 and x2')
    plt.grid(True)
    plt.legend()
    plt.show()

x_ET = x[:,0]
for t in range(T):
    e_j = x[:,t] - x_ET
    lambada_Q = min(np.linalg.eig(Q)[0])
    if np.linalg.norm(e_j) <= np.sqrt((1-pi)*(1-mu)*lambada_Q * np.linalg.norm(x_ET)**2
                                      /(2*du*np.linalg.norm(R) + (1-1/pi)* lambada_Q)):
        w_a[:,t+1] = w_a[:,t]
    else:
        x_ET = x[:,t]
        w_a[:,t+1] = w_a[:,t] + (-k_a1/(np.sqrt(1 + np.linalg.norm(omega_t) **2)
                                     * g_sigma(x[:,t])) * (w_c[:,t] - w_a[:,t]) *error_hjb_t) * d_t
    
    u_ET = u_x( x_ET,w_a[:,t])
    error_hjb_t,omega_t,rho_t = hjb_error(x[:,t],u_ET,w_a[:,t],w_c[:,t])
    #print("u",u)
    #print("rho",rho_t)
    Lambda_t = 1/2 * (delta_sigma_v(x_ET) @ g(x_ET)) * (1/R * g(x[:,t]) @ x[:,t])
   
    if x[:,t] @ (f(x[:,t]) + g(x[:,t]) * u_ET) >0:
        w_c[:,t+1] = w_c[:,t] + ((-k_c1 * np.dot(Tau[:,:,t],omega_t))/rho_t * error_hjb_t - gamma_1 * Lambda_t) * d_t
    else:
        w_c[:,t+1] = w_c[:,t] + ((-k_c1 * np.dot(Tau[:,:,t],omega_t))/rho_t * error_hjb_t) * d_t


    Tau[:,:,t+1] = Tau[:,:,t] + ( beta * Tau[:,:,t] - 
                                 (k_c1 * Tau[:,:,t] @ omega_t.reshape(3,1) @ omega_t.reshape(1,3) @ Tau[:,:,t])/rho_t**2 ) * d_t 
    x[:,t+1] = x[:,t] + (f(x[:,t]) + g(x[:,t]) * u_ET )*d_t
  
    u_.append(u_ET)
    # print(t)
    # print("u",u_ET)
    # print(x[:,t])
    # print("w_a",w_a[:,t])
    # print("w_c",w_c[:,t])
   #print("x",x[:,t])
# 创建图形
plt.figure(figsize=(10, 6))  # 设置图形尺寸

# 绘制四条线
plt.plot(x[1,:],label="x-2-real", color='tab:blue')
plt.plot(x[0,:],label="x-1-real", color='tab:orange')

# 添加标题和轴标签
plt.title('Convex-set', fontsize=16)
plt.xlabel('t', fontsize=14)
#plt.ylabel('Y轴', fontsize=14)

# 设置横坐标轴范围为0到15
plt.xlim(0, 120000)

# 设置横坐标轴刻度标签
plt.xticks(range(0, 120001, 10000), [i/10000 for i in range(0, 120001, 10000)])
#plt.xticks(range(0, 320001, 40000), [i/20000 for i in range(0, 320001, 20000)])

# 添加每根线的标注
plt.annotate("x-1-real", xy=(750, 750**2), xytext = None, arrowprops=dict(arrowstyle='->'), fontsize=10, color='tab:blue')
plt.annotate( "x-0-real",xy=(1200, 1200**2),  xytext = None, arrowprops=dict(arrowstyle='->'), fontsize=10, color='tab:orange')
plt.annotate( "x-1-estimate",xy=(1300, 1300**2),  xytext = None, arrowprops=dict(arrowstyle='->'), fontsize=10, color='tab:green')
plt.annotate( "x-0-estimate",xy=(1450, 1450**2),  xytext = None,  arrowprops=dict(arrowstyle='->'), fontsize=10, color='tab:red')

# 添加图例
plt.legend(loc='best')

# 显示图形
#plt.grid(True)  # 添加网格线
plt.tight_layout()  # 调整布局，避免标签被裁剪

plt.show()

plt.figure() 
#plt.xticks(range(0, 320001, 40000), [i/20000 for i in range(0, 320001, 20000)])
plt.xlim(0, 120000)

# 设置横坐标轴刻度标签
plt.xticks(range(0, 120001, 10000), [i/10000 for i in range(0, 120001, 10000)])
plt.plot(u_,label = "Feed-back Control Policy ")
plt.legend(loc='best')
plt.show()


plt.figure()
plt.xlim(0, 120000)

# 设置横坐标轴刻度标签
plt.xticks(range(0, 120001, 10000), [i/10000 for i in range(0, 120001, 10000)])

plt.legend(loc = "best")
plt.show()
  
plot_phase(x[0,:],x[1,:])

    

