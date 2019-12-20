import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import hypsecant
from scipy.stats import norm
from scipy.special import erf

epsabs = 1e-2
epsrel = 1e-2
dz = 0.05
zmin = -10.0
zmax = 10.0

def hard_tanh(x):
    return np.minimum(np.maximum(x+1, 0) - 1, 1)   
def linear(x):
    return x    
def leaky_relu(x):
    return np.where(x > 0, x, x * 0.01)   
def prelu(x):
    return np.where(x > 0, x, x * 0.5)  
def relu(x):
    return np.maximum(x,0)   
def tanh(x):
    return np.tanh(x)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def erf2(x):
    return erf(np.sqrt(np.pi)/2*x)

def d_tanh(x):
    return (hypsecant.pdf(x) * np.pi)**2   
def d_linear(x):
    return 1
def d_relu(x):
    return np.where(x > 0, 1, 0)   
def d_hard_tanh(x):
    x[(x>=-1)&(x<=1)] =1
    x[x<-1]=0             # can't change the order
    x[x>1] = 0
    return x   

def d_hard_tanh1(x):
    if x > 1:
        return 0
    elif x<-1:
        return 0
    else:
        return 1

def d_erf(x):
    return  np.exp(-np.pi/4.0*x**2)    
      
def fast_integral(integrand, zmin, zmax, dz, ndim=1):
    zs = np.r_[zmin:zmax:dz]
    if ndim > 1:
        zgrid = np.meshgrid(*((zs,) * ndim))
    else:
        zgrid = (zs,)
    out = integrand(*zgrid)
    return out.sum(tuple(np.arange(ndim))) * dz**ndim

def qmap(qin, weight_sigma=1.0 , bias_sigma=0.0, rho = 1.0, active = 'linear',
         epsabs=epsabs, epsrel=epsrel, zmin=-10, zmax=10, dz=dz, fast=True):
    qin = np.atleast_1d(qin)
    # Perform Gaussian integral
    def integrand(z):
        return norm.pdf(z[:, None]) * eval(active)(np.sqrt(qin[None, :]) * z[:, None])**2
    integral = fast_integral(integrand, zmin, zmax, dz=dz)
    return weight_sigma**2 * integral/rho + bias_sigma**2

def cmap(q1, q2, q12, weight_sigma, bias_sigma, active='linear', zmin=-10, zmax=10, dz=dz, fast=True):
    q1 = np.atleast_1d(q1)
    q2 = np.atleast_1d(q2)
    q12 = np.atleast_1d(q12)

    u1 = np.sqrt(q1)
    u2 = q12 / np.sqrt(q1)
    # XXX: tolerance fudge factor
    u3 = np.sqrt(q2 - q12**2 / q1 + 1e-8)
    def integrand(z1, z2):
        return norm.pdf(z1[..., None]) * norm.pdf(z2[..., None]) * (
            eval(active)(u1[None, None, :] * z1[..., None]) *
            eval(active)(u2[None, None, :] * z1[..., None] + u3[None, None, :] * z2[..., None]))
    integral = fast_integral(integrand, zmin, zmax, dz, ndim=2)
    return weight_sigma**2 * integral + bias_sigma**2


#==============================================================================
def compute_chi1(qstar, weight_sigma=1.0, bias_sigma=0.01, rho=1.0, dactive='d_linear'):
    def integrand(z):
        return norm.pdf(z) * eval(dactive)(np.sqrt(qstar) * z)**2
    integral = quad(integrand, zmin, zmax, epsabs=epsabs, epsrel=epsrel)[0]
    return weight_sigma**2/rho * integral

def compute_chi2(qstar, cstar, weight_sigma=1.0, bias_sigma=0.01, rho=1.0, dactive='d_linear'):
    u1 = np.sqrt(qstar)
    u2 = np.sqrt(qstar)*cstar
    u3 = np.sqrt(qstar)*np.sqrt(1 - cstar**2)
    def integrand(z1, z2):
        return norm.pdf(z1[..., None]) * norm.pdf(z2[..., None]) * (
            eval(dactive)(u1 * z1[..., None]) *
            eval(dactive)(u2 * z1[..., None] + u3 * z2[..., None]))
    integral = fast_integral(integrand, zmin, zmax, dz, ndim=2) 
    return weight_sigma**2 * integral   

def compute_chi3(qstar, weight_sigma=1.0, bias_sigma=0.01, rho=1.0, active = 'linear', dactive='d_linear'):
    print (active, qstar)
    def integrand(z):
        return norm.pdf(z) * eval(dactive)(np.sqrt(qstar) * z)**2 * eval(active)(np.sqrt(qstar) * z)**2
    integral = quad(integrand, zmin, zmax, epsabs=epsabs, epsrel=epsrel)[0]
    return weight_sigma**2/rho * integral   

def compute_chi4(qstar, weight_sigma=1.0, bias_sigma=0.01, rho=1.0, active = 'linear', dactive='d_linear'):
    def integrand(z):
        return norm.pdf(z)*eval(dactive)(np.sqrt(qstar)*z)**2*eval(active)(np.sqrt(qstar)*z)**2*z**2
    integral = quad(integrand, zmin, zmax, epsabs=epsabs, epsrel=epsrel)[0]
    return weight_sigma**2/rho * integral 


def compute_qlen(qstar, weight_sigma2=1.0, bias_sigma2=0.01, phi=np.tanh, d2phi=np.tanh):
    def integrand(z):
        return norm.pdf(z) * d2phi(np.sqrt(qstar) * z) * phi(np.sqrt(qstar) * z)
    integral = quad(integrand, zmin, zmax, epsabs=epsabs, epsrel=epsrel)[0]
    return weight_sigma2 * integral 

def compute_clen(qstar, cstar, weight_sigma2=1.0, bias_sigma2=0.01, dphi=np.tanh):
    
    u1 = np.sqrt(qstar)
    u2 = np.sqrt(qstar)*cstar
    # XXX: tolerance fudge factor
    #u3 = np.sqrt(qstar)*np.sqrt(1 - cstar**2 + 1e-8)
    u3 = np.sqrt(qstar)*np.sqrt(1 - cstar**2)
    #print (u1,u2,u3)
    def integrand(z1, z2):
        return norm.pdf(z1[..., None]) * norm.pdf(z2[..., None]) * (
            dphi(u1 * z1[..., None]) *
            dphi(u2 * z1[..., None] + u3 * z2[..., None]))
    integral = fast_integral(integrand, zmin, zmax, dz, ndim=2)    
    return weight_sigma2 * integral     

def q_fixed_point(weight_sigma, bias_sigma, nonlinearity, rho=1.0, max_iter=1000, tol=1e-12, qinit=1.0, fast=True, tol_frac=0.01):
    """Compute fixed point of q map"""
    q = qinit
    qs = []
    for i in range(max_iter):
        qnew = qmap(q, weight_sigma, bias_sigma, rho, nonlinearity, fast=fast)
        err = np.abs(qnew - q)
        qs.append(q)
        if err < tol:
            break
        q = qnew
    # Find first time it gets within tol_frac fracitonal error of q*
    frac_err = (np.array(qs) - q)**2 / (1e-9 + q**2)
    t = np.flatnonzero(frac_err < tol_frac)[0]
    return t, q

def c_fixed_point(qstar, weight_sigma, bias_sigma, nonlinearity='linear',  q12=1.0, max_iter=1000, tol=1e-12, 
                        fast=True, tol_frac=0.01):
    """Compute fixed point of q map"""
    c = q12
    cs = []
    for i in range(max_iter):
        cnew = cmap(qstar, qstar, c, weight_sigma, bias_sigma, nonlinearity, fast=fast)
    #    print (cnew)
        err = np.abs(cnew - c)
        cs.append(c)
        if err < tol:
            break
        c = cnew
    # Find first time it gets within tol_frac fracitonal error of q*
    frac_err = (np.array(cs) - c)**2 / (1e-9 + c**2)
    #t = np.flatnonzero(frac_err < tol_frac)[0]
    #print (c)
    #print (c/qstar)
    return  c/qstar    

