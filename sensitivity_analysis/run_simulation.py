import os
os.environ["OMP_NUM_THREADS"] = str(1)
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from multiprocessing import Pool
import tqdm
import random
import torch
import robust_loss_pytorch
from pathlib import Path


# kinetic network functions
def caf(tt,k1,k2,k3,k4,k5,k6,k7,k8,par0,bro0,phy0):
    k9 = k2+k3+k4+k5
    y = k1/(k9-k1)*(np.exp(-k1*tt)-np.exp(-k9*tt))
    return y

def par(tt,k1,k2,k3,k4,k5,k6,k7,k8,par0,bro0,phy0):
    k9 = k2+k3+k4+k5
    y = k1*k2/(k9-k1)*(np.exp(-k9*tt)/(k9-k6)-np.exp(-k1*tt)/(k1-k6)+np.exp(-k6*tt)*(k9-k1)/((k9-k6)*(k1-k6)))+par0*np.exp(-k6*tt)
    return y

def bro(tt,k1,k2,k3,k4,k5,k6,k7,k8,par0,bro0,phy0):
    k9 = k2+k3+k4+k5
    y = k1*k3/(k9-k1)*(np.exp(-k9*tt)/(k9-k7)-np.exp(-k1*tt)/(k1-k7)+np.exp(-k7*tt)*(k9-k1)/((k9-k7)*(k1-k7)))+bro0*np.exp(-k7*tt)
    return y

def phy(tt,k1,k2,k3,k4,k5,k6,k7,k8,par0,bro0,phy0):
    k9 = k2+k3+k4+k5
    y = k1*k4/(k9-k1)*(np.exp(-k9*tt)/(k9-k8)-np.exp(-k1*tt)/(k1-k8)+np.exp(-k8*tt)*(k9-k1)/((k9-k8)*(k1-k8)))+phy0*np.exp(-k8*tt)
    return y

# fit functions
def optimization_problem_debug(input_):
    '''Optimization function.'''

    if loss_func == max_robust_loss or loss_func == abs_robust_loss:
        global alpha, scale, loss
        alpha, scale = 1, 1

    p0,lb,ub = input_
    alpha, scale = 1, 1
    y_values = with_err.values.flatten()
    x_values = with_err.index.values
    bounds   = (lb.astype('float64') ,ub.astype('float64') )
    para, var = curve_fit(f         = fit_with_sv,
                          xdata     = x_values,
                          ydata     = y_values,
                          p0        = p0,
                          bounds    = bounds,
                          method    = 'trf',
                          max_nfev  = 1000,
                          loss      = loss_func,
                          tr_solver = 'exact')

    if loss_func != max_robust_loss or loss_func != abs_robust_loss:
            observed = y_values
            expected = fit_with_sv(x_values,*para)
            loss = np.sum(loss_func(abs(observed-expected))[0])

    return list(para)+[float(loss)]

def optimization_problem_production(input_):
    '''Optimization function with exceptions of ValueError and RuntimeError.'''

    if loss_func == max_robust_loss or loss_func == abs_robust_loss:
        global alpha, scale, loss
        alpha, scale = 1, 1

    p0,lb,ub = input_
    y_values = with_err.values.flatten()
    x_values = with_err.index.values
    bounds   = (lb.astype('float64') ,ub.astype('float64') )

    # In rare cases the input parameters can be infeasible or the max_nfev are
    # eceeded. This usually does not hurt the simulation and, therefore, these
    # errors are excepted.
    try:
        para, var = curve_fit(f         = fit_with_sv,
                              xdata     = x_values,
                              ydata     = y_values,
                              p0        = p0,
                              bounds    = bounds,
                              method    = 'trf',
                              max_nfev  = 1000,
                              loss      = loss_func,
                              tr_solver = 'exact')

        if loss_func != max_robust_loss or loss_func != abs_robust_loss:
            observed = y_values
            expected = fit_with_sv(x_values,*para)
            loss = np.sum(loss_func(abs(observed-expected))[0])

        return list(para)+[float(loss)]
    except (ValueError,RuntimeError) as e:
        return f' #Error: {e} '

def fit_with_sv(tt,kcincaf,kcafpar,kcafbro,kcafphy,kcafdeg,kpardeg,kbrodeg,kphydeg,par0,bro0,phy0,*sfs):
    '''This is the function takes timepoints array and all fitting parameters
    and calculates c(t)*SV.'''

    global global_args, global_timepoints
    args = [kcincaf,kcafpar,kcafbro,kcafphy,kcafdeg,kpardeg,kbrodeg,kphydeg,par0,bro0,phy0]
    global_args = args
    global_timepoints = tt
    sfs2 = np.array([sfs,sfs]).flatten('F')
    #print(sfs2.shape,tt.shape)
    caf_true = np.array(caf(tt,*args))*np.array(sfs2)
    par_true = np.array(par(tt,*args))*np.array(sfs2)
    bro_true = np.array(bro(tt,*args))*np.array(sfs2)
    phy_true = np.array(phy(tt,*args))*np.array(sfs2)
    y = np.array([caf_true,par_true,bro_true,phy_true])
    y = y.flatten('F')
    return y

def fit_without_sv(tt,kcincaf,kcafpar,kcafbro,kcafphy,kcafdeg,kpardeg,kbrodeg,kphydeg,par0,bro0,phy0):
    '''This is the function takes timepoints and kinetic and concentration
    fitting parameters and calculates c(t).'''

    args = [kcincaf,kcafpar,kcafbro,kcafphy,kcafdeg,kpardeg,kbrodeg,kphydeg,par0,bro0,phy0]
    caf_true = np.array(caf(tt,*args))
    par_true = np.array(par(tt,*args))
    bro_true = np.array(bro(tt,*args))
    phy_true = np.array(phy(tt,*args))
    y = np.array([caf_true,par_true,bro_true,phy_true])
    y = y.flatten('F')
    return y

# loss functions
def abs_robust_loss(z):
    '''Takes array and calculates loss with generalized adaptive loss function.

    [1] Implemented according to Barron, J. T. (2019). A general and adaptive robust
    loss function. In Proceedings of the IEEE/CVF Conference on Computer Vision
    and Pattern Recognition (pp. 4331-4339).
    '''

    global alpha, scale, loss

    # second optimization problem for alpha and scale, bounds are as decribed
    # in [1].(https://github.com/jonbarron/robust_loss_pytorch/blob/
    # master/robust_loss_pytorch/adaptive.py)
    result = minimize(fun=min_problem,
                      args=z,
                      x0=(alpha,scale),
                      bounds=[(.001,1.999),(10e-5,np.inf)],
                      method='TNC',
                      options={'maxiter':100,'accuracy':10e-10})
    alpha = result['x'][0]
    scale = result['x'][1]
    loss = result['fun']

    # calculate row-wise loss and gradient
    distribution = robust_loss_pytorch.distribution.Distribution()
    nll          = distribution.nllfun(torch.tensor(z,dtype=torch.float64),
                                       torch.clamp(torch.ones(1,dtype=torch.float64)*alpha,0,1.999),
                                       torch.clamp(torch.ones(1,dtype=torch.float64)*scale,0,np.inf))
    gradient = exact_grad(z,alpha,scale)
    rho = np.zeros((3,len(z)))
    rho[0] = nll.detach().numpy()               # loss
    rho[1] = gradient                           # first  derivative of loss
    # second derivative, rho[2], is not used and thus not calculated
    return rho

def exact_grad(z,alpha,scale):
    '''Calculates robust loss gradient as described in [1], Equation 9.'''

    if alpha == 2:
        grad = z/(scale**2)
    elif alpha == 0:
        grad = 2*z/(z**2+2*scale**2)
    elif alpha == -np.inf:
        grad = z/scale**2 * np.exp(-1/2*(z/scale)**2)
    else:
        exponent = (alpha/2-1)
        grad = z/(scale**2)*((z/scale)**2/(abs(alpha-2))+1)**exponent
    return grad

def min_problem(parameters,z):
    '''Optimizing alpha and scale for minimizing the loss.'''

    alpha, scale = parameters
    distribution = robust_loss_pytorch.distribution.Distribution()
    nll          = distribution.nllfun(torch.tensor(z,dtype=torch.float64),
                                       torch.clamp(torch.ones(1,dtype=torch.float64)*alpha,0,1.999),
                                       torch.clamp(torch.ones(1,dtype=torch.float64)*scale,0,np.inf))
    return torch.mean(nll).detach().numpy()

def max_robust_loss(z):
    '''Takes array, calculates maximum of relative and absolute error and
    calculates loss with generalized adaptive loss function [1].'''

    global alpha, scale, loss
    abs_error = z
    # calculate relative error, except absolute error is 0
    y = fit_without_sv(global_timepoints,*global_args)
    rel_error = np.divide(z, y, out=z.copy(), where=y!=0)
    # maximum error
    z = np.maximum(abs_error,rel_error)

    # second optimization problem for alpha and scale, bounds are as decribed
    # in [1].(https://github.com/jonbarron/robust_loss_pytorch/blob/
    # master/robust_loss_pytorch/adaptive.py)
    result = minimize(fun=min_problem,
                      args=z,
                      x0=(alpha,scale),
                      bounds=[(.001,1.999),(10e-5,np.inf)],
                      method='TNC',
                      options={'maxiter':100,'accuracy':10e-10})
    alpha = result['x'][0]
    scale = result['x'][1]
    loss = result['fun']

    # calculate row-wise loss and gradient
    distribution = robust_loss_pytorch.distribution.Distribution()
    nll    = distribution.nllfun(torch.tensor(z,dtype=torch.float64),
                                 torch.clamp(torch.ones(1,dtype=torch.float64)*alpha,0,1.999),
                                 torch.clamp(torch.ones(1,dtype=torch.float64)*scale,0,np.inf))
    gradient = exact_grad(z,alpha,scale)
    rho = np.zeros((3,len(z)))
    rho[0] = nll.detach().numpy()               # loss
    rho[1] = gradient                           # first  derivative of loss
    # second derivative, rho[2], is not used and thus not calculated
    return rho

def abs_huber_loss(z):
    '''Takes array, calculates Huber loss as implemented in SciPy.

    [2] Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T.,
    Cournapeau, D., ... & van Mulbregt, P. (2020). SciPy 1.0: fundamental
    algorithms for scientific computing in Python. Nature methods, 17(3),
    261-272.
    '''

    rho = np.empty((3,len(z)))
    mask = z <= 1
    rho[0, mask] = z[mask]
    rho[0, ~mask] = 2 * z[~mask]**0.5 - 1
    rho[1, mask] = 1
    rho[1, ~mask] = z[~mask]**-0.5
    rho[2, mask] = 0
    rho[2, ~mask] = -0.5 * z[~mask]**-1.5
    return rho

def max_huber_loss(z):
    '''Takes array, calculates maximum of relative and absolute error and
    calculates Huber loss [2].
    '''

    global alpha, scale, loss
    abs_error = z
    # calculate relative error, except absolute error is 0
    y = fit_without_sv(global_timepoints,*global_args)
    rel_error = np.divide(z, y, out=z.copy(), where=y!=0)
    # maximum error
    z = np.maximum(abs_error,rel_error)

    rho = np.empty((3,len(z)))
    mask = z <= 1
    rho[0, mask] = z[mask]
    rho[0, ~mask] = 2 * z[~mask]**0.5 - 1
    rho[1, mask] = 1
    rho[1, ~mask] = z[~mask]**-0.5
    rho[2, mask] = 0
    rho[2, ~mask] = -0.5 * z[~mask]**-1.5
    return rho

def abs_cauchy_loss(z):
    '''Takes array, calculates Cauchy loss as implemented in SciPy [2].'''
    rho = np.empty((3,len(z)))
    rho[0] = np.log1p(z)
    t = 1 + z
    rho[1] = 1 / t
    rho[2] = -1 / t**2
    return rho

def max_cauchy_loss(z):
    '''Takes array, calculates maximum of relative and absolute error and
    calculates Cauchy loss [2].
    '''

    global alpha, scale, loss
    abs_error = z
    # calculate relative error, except absolute error is 0
    y = fit_without_sv(global_timepoints,*global_args)
    rel_error = np.divide(z, y, out=z.copy(), where=y!=0)
    # maximum error
    z = np.maximum(abs_error,rel_error)

    rho = np.empty((3,len(z)))
    rho[0] = np.log1p(z)
    t = 1 + z
    rho[1] = 1 / t
    rho[2] = -1 / t**2
    return rho

def abs_soft_l1_loss(z):
    '''Takes array, calculates Soft-l1 loss as implemented in SciPy [2].'''

    rho = np.empty((3,len(z)))
    t = 1 + z
    rho[0] = 2 * (t**0.5 - 1)
    rho[1] = t**-0.5
    rho[2] = -0.5 * t**-1.5
    return rho

def max_soft_l1_loss(z):
    '''Takes array, calculates maximum of relative and absolute error and
    calculates Soft-l1 loss [2].
    '''

    global alpha, scale, loss
    abs_error = z
    # calculate relative error, except absolute error is 0
    y = fit_without_sv(global_timepoints,*global_args)
    rel_error = np.divide(z, y, out=z.copy(), where=y!=0)
    # maximum error
    z = np.maximum(abs_error,rel_error)

    rho = np.empty((3,len(z)))
    t = 1 + z
    rho[0] = 2 * (t**0.5 - 1)
    rho[1] = t**-0.5
    rho[2] = -0.5 * t**-1.5
    return rho

# data generation functions
def generate_sv_data(original,path):
    '''Takes original concentration DataFrame and multiplies randomly sampled
    SV values.'''

    # generate random SV values from truncated Gaussian distribution
    sv    = [0]
    mu    = np.mean([.05*4,.62*4])
    sigma = np.std([.05*4,.62*4])
    while np.min(sv) <= 0:
        sv = np.random.normal(mu,sigma/2,len(timepoints))
    # put SV on original data
    with_sv = original.copy()
    for metabolite in ['caf','par','bro','phy']:
        with_sv[metabolite] = with_sv[metabolite]*sv
    # write SV values to output file
    with open(path+'.txt','a') as file:
        file.write('\n#SV_VALUES ')
        file.write(str(list(sv))[1:-1].replace(',',''))
    return with_sv

def generate_err_data(with_sv,path):
    '''Takes original*SV concentration DataFrame and multiplies randomly sampled
    errors.'''

    # generate random relative error values
    e     = [0]
    mu    = 1
    sigma = error_sigma
    while np.min(e) <= 0:
        # in contrast to SV the error is not equal for every time point
        e = np.random.normal(mu,sigma,(len(timepoints)*2,4))
    # Put error on SV*oiginal data. Measurements are done in replicates, this is
    # also represented here.
    with_err = pd.concat([with_sv,with_sv]).sort_index()*e
    # write error values to output file
    with open(path+'.txt','a') as file:
        file.write('\n#ERR_VALUES ')
        file.write(str(list(e))[1:-1]) #.replace(',','').replace('array(','').replace(')',''))
    return with_err

def generate_original_data(path,timepoints):
    '''Takes time points and generates original concentration DataFrame.'''

    # Kinetic model parameters
    kcincaf    = 1.60
    kcafpar    = 0.02
    kcafbro    = 0.01
    kcafphy    = 0.01
    kcafdeg    = 0.04
    kpardeg    = 0.13
    kbrodeg    = 0.10
    kphydeg    = 0.10
    par0       = 0.02
    bro0       = 0.01
    phy0       = 0.01
    parameters = np.array((kcincaf,kcafpar,kcafbro,kcafphy,kcafdeg,kpardeg,kbrodeg,kphydeg,par0,bro0,phy0))
    caf_values = caf(timepoints,*parameters)
    par_values = par(timepoints,*parameters)
    bro_values = bro(timepoints,*parameters)
    phy_values = phy(timepoints,*parameters)
    original   = pd.DataFrame([caf_values,par_values,bro_values,phy_values],columns=timepoints,index=['caf','par','bro','phy']).transpose()
    # Write original kinetic model parameters and time points into output file.
    with open(path+'.txt','w') as file:
        file.write('#ORIGINAL_PARAMETERS ')
        file.write(str(list(parameters))[1:-1].replace(',',''))
        file.write('\n#TIMEPOINTS ')
        file.write(str(list(timepoints))[1:-1].replace(',',''))
    return original

def save_raw_data(path,output_list,n_rep):
    '''Takes the output of the Monte Carlo replicates and saves it.'''

    with open(path+'_raw/'+str(n_rep)+'.raw','w') as file:
        for i in output_list:
            file.write(str(i)[1:-1].replace(',',''))
            file.write('\n')
    return

def get_best_fit(path,max_tries,n_rep,n_cpu):
    '''Runs the fitting protocol with the given settings.'''
    # Set bounds of model parameters
    n_sf       = len(timepoints)
    lb         = np.concatenate((np.zeros(11),np.ones(n_sf)*.05))
    ub         = np.concatenate(([10],np.ones(10)*.2,np.ones(n_sf)*4))
    # Generate input data for Monte Carlo replicates
    input_list = []
    n_try      = 0
    while n_try < max_tries:
        n_try += 1
        p0     = []
        for l,u in zip(lb,ub):
            p0.append(random.uniform(l,u))
        p0 = np.array(p0)
        input_list.append([p0,lb,ub])

    # Runs optimization problem as multiprocess
    output_list = []
    p = Pool(processes = n_cpu)
    if debug:
        for _ in tqdm.tqdm(p.imap_unordered(optimization_problem_debug,input_list),total=len(input_list)):
            output_list.append(_)
        p.close()
    else:
        for _ in tqdm.tqdm(p.imap_unordered(optimization_problem_production,input_list),total=len(input_list)):
            output_list.append(_)
        p.close()

    # Remove failed optimizations from output_list and make an array out of it.
    tmp = []
    for i in output_list:
        if '#Error' not in i:
            tmp.append(i)
    out_array = np.array(tmp)

    # Write out "raw" Monte Carlo data
    Path(path+'_raw').mkdir(exist_ok=True)
    save_raw_data(path,output_list,n_rep)

    # Check which MC replicate has the lowest loss and write out result.
    best_fit  = out_array[np.argmin(out_array[:,-1]),:-1]
    with open(path+'.txt','a') as file:
        file.write('\n')
        file.write(str(list(best_fit))[1:-1].replace(',',''))
    return

if __name__ == '__main__':

    # SIMULATION INPUT PARAMETERS
    max_reps    = 300                                 # Number of bootstrap replicates, i
    max_tries   = 100                                 # Number of Monte Carlso replicates, n
    path        = __file__.strip('.py').strip('/')    # output file path + name, without file extension
    n_cpu       = 32                                  # Number of CPUs used for multiprocessing
    timepoints  = np.array([0,.25,.5,.75,1,1.5,2,3,4,5,6,7,8,9,10,11,12,13,14,24]) # timepoints in h
    debug       = False                               # if False, a try function will except errors 
                                                      # that can rarly happen during the simulation. 
                                                      # For production False is recommended.
    error_sigma = 10/100                              # sigma of measurement error
    loss_func   = max_robust_loss                     # loss function
    # following loss functions are implemented
        # abs_cauchy_loss
        # abs_huber_loss
        # abs_soft_l1_loss
        # abs_robust_loss
        # max_cauchy_loss
        # max_huber_loss
        # max_soft_l1_loss
        # max_robust_loss

    # START SIMULATION
    n_rep      = 0
    original   = generate_original_data(path,timepoints)
    while n_rep < max_reps:
        n_rep += 1
        print('Replicate ',n_rep)
        with_sv  = generate_sv_data(original,path)
        with_err = generate_err_data(with_sv,path)
        get_best_fit(path,max_tries,n_rep,n_cpu)
    print('done')
