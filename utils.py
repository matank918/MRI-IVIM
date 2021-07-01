
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
import fitting_algorithms as fit
from joblib import Parallel, delayed
import copy
import warnings
from config import *



def isnan(x):
    # this program indicates what are NaNs
    return x != x


def plot_progress(X_batch, X_pred, bvalues, loss_train, loss_val, arg):
    # this program plots the progress of the training. It will plot the loss and validatin loss, as well as 4 IVIM curve
    # fits to 4 data points from the input
    inds1 = np.argsort(bvalues)
    X_batch = X_batch[:, inds1]
    X_pred = X_pred[:, inds1]
    bvalues = bvalues[inds1]
    if arg.fig:
        matplotlib.use('TkAgg')
        plt.close('all')
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(bvalues, X_batch.data[0], 'o')
        axs[0, 0].plot(bvalues, X_pred.data[0])
        axs[0, 0].set_ylim(min(X_batch.data[0]) - 0.3, 1.2 * max(X_batch.data[0]))
        axs[1, 0].plot(bvalues, X_batch.data[1], 'o')
        axs[1, 0].plot(bvalues, X_pred.data[1])
        axs[1, 0].set_ylim(min(X_batch.data[1]) - 0.3, 1.2 * max(X_batch.data[1]))
        axs[0, 1].plot(bvalues, X_batch.data[2], 'o')
        axs[0, 1].plot(bvalues, X_pred.data[2])
        axs[0, 1].set_ylim(min(X_batch.data[2]) - 0.3, 1.2 * max(X_batch.data[2]))
        axs[1, 1].plot(bvalues, X_batch.data[3], 'o')
        axs[1, 1].plot(bvalues, X_pred.data[3])
        axs[1, 1].set_ylim(min(X_batch.data[3]) - 0.3, 1.2 * max(X_batch.data[3]))
        plt.legend(('data', 'estimate from network'))
        for ax in axs.flat:
            ax.set(xlabel='b-value (s/mm2)', ylabel='normalised signal')
        for ax in axs.flat:
            ax.label_outer()
        plt.ion()
        plt.show()
        plt.pause(0.001)
        plt.figure(2)
        plt.clf()
        plt.plot(loss_train)
        plt.plot(loss_val)
        plt.yscale("log")
        plt.xlabel('epoch #')
        plt.ylabel('loss')
        plt.legend(('training loss', 'validation loss (after training epoch)'))
        plt.ion()
        plt.show()
        plt.pause(0.001)


def make_data_complete(dw_data,bvalues,fraction_threshold=0.2):
    """
    This function is specific to missing data. For example, due to motion, after image registration our dataset
    contained gaps of information in some patients. As the Neural Network might get confused by empty slots,
    this program was desigend to fill up these slots with more realistic data estimates.
    :param bvalues: Array with the b-values
    :param dw_data: 1D Array with diffusion-weighted signal at different b-values
    :param fraction_threshold: an optional parameter determining the maximum fraction of missing data allowed.
    if more data is missing, the algorithm will not correct to prrvent too unrealistic (noiseless) data.
    :return dw_data: corrected dataset
    """
    if len(np.shape(dw_data)) is 4:
        sx, sy, sz, n_b_values = dw_data.shape
        dw_data = np.reshape(dw_data, (sx * sy * sz, n_b_values))
        reshape = True
    dw_data[isnan(dw_data)] = 0
    zeros = (dw_data == 0)
    locs = np.mean(zeros,axis=1)
    sels = (locs > 0) & (locs < fraction_threshold)
    data_to_correct = dw_data[sels,:]
    print('correcting {} datapoints'.format(len(data_to_correct)))
    def parfun(i):
        datatemp = data_to_correct[i,:]
        nonzeros = datatemp > 0
        bvaltemp = bvalues[nonzeros]
        datatempf=datatemp[nonzeros]
        norm=np.nanmean(datatempf)
        datatemp = datatemp / norm
        datatempf = datatempf / norm
        [Dt,Fp,Dp,S0]=fit.fit_least_squares(bvaltemp, datatempf, S0_output=True, fitS0=True, bounds=([0, 0, 0, 0.8], [0.005, 0.7, 0.3, 3]))
        datatemp[~nonzeros] = fit.ivim(bvalues,Dt,Fp,Dp,S0)[~nonzeros]
        return datatemp * norm
    data_to_correct = Parallel(n_jobs=4,batch_size=64)(delayed(parfun)(i) for i in tqdm(range(len(data_to_correct)), position=0,
                                                                    leave=True))
    dw_data[sels, :] = data_to_correct
    if reshape:
        dw_data = np.reshape(dw_data, (sx, sy, sz, n_b_values))
    return dw_data


# def checkarg_train_pars(arg):
#     if not hasattr(arg,'optim'):
#         warnings.warn('arg.train.optim not defined. Using default ''adam''')
#         arg.optim = 'adam'  # these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
#     if not hasattr(arg,'lr'):
#         warnings.warn('arg.train.lr not defined. Using default value 0.0001')
#         arg.lr = 0.0001  # this is the learning rate. adam needs order of 0.001; others order of 0.05? sgdr can do 0.5
#     if not hasattr(arg, 'patience'):
#         warnings.warn('arg.train.patience not defined. Using default value 10')
#         arg.patience = 10  # this is the number of epochs without improvement that the network waits untill determining it found its optimum
#     if not hasattr(arg,'batch_size'):
#         warnings.warn('arg.train.batch_size not defined. Using default value 128')
#         arg.batch_size = 128  # number of datasets taken along per iteration
#     if not hasattr(arg,'maxit'):
#         warnings.warn('arg.train.maxit not defined. Using default value 500')
#         arg.maxit = 500  # max iterations per epoch
#     if not hasattr(arg,'split'):
#         warnings.warn('arg.train.split not defined. Using default value 0.9')
#         arg.split = 0.9  # split of test and validation data
#     if not hasattr(arg,'load_nn'):
#         warnings.warn('arg.train.load_nn not defined. Using default of False')
#         arg.load_nn = False
#     if not hasattr(arg,'loss_fun'):
#         warnings.warn('arg.train.loss_fun not defined. Using default of ''rms''')
#         arg.loss_fun = 'rms'  # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
#     if not hasattr(arg,'skip_net'):
#         warnings.warn('arg.train.skip_net not defined. Using default of False')
#         arg.skip_net = False
#     if not hasattr(arg,'use_cuda'):
#         arg.use_cuda = torch.cuda.is_available()
#     if not hasattr(arg, 'device'):
#         arg.device = torch.device("cuda:0" if arg.use_cuda else "cpu")
#     return arg
#
#
# def checkarg_net_pars(arg):
#     if not hasattr(arg,'dropout'):
#         warnings.warn('arg.net_pars.dropout not defined. Using default value of 0.1')
#         arg.dropout = 0.1  # 0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
#     if not hasattr(arg,'batch_norm'):
#         warnings.warn('arg.net_pars.batch_norm not defined. Using default of True')
#         arg.batch_norm = True  # False/True turns on batch normalistion
#     if not hasattr(arg,'parallel'):
#         warnings.warn('arg.net_pars.parallel not defined. Using default of True')
#         arg.parallel = True  # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
#     if not hasattr(arg,'con'):
#         warnings.warn('arg.net_pars.con not defined. Using default of ''sigmoid''')
#         arg.con = 'sigmoid'  # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
#     if not hasattr(arg,'cons_min'):
#         warnings.warn('arg.net_pars.cons_min not defined. Using default values of  [-0.0001, -0.05, -0.05, 0.7]')
#         arg.cons_min = [-0.0001, -0.05, -0.05, 0.7, -0.05, 0.06]  # Dt, Fp, Ds, S0 F2p, D2*
#     if not hasattr(arg,'cons_max'):
#         warnings.warn('arg.net_pars.cons_max not defined. Using default values of  [-0.0001, -0.05, -0.05, 0.7]')
#         arg.cons_max = [0.005, 0.7, 0.3, 1.3, 0.3, 0.3]  # Dt, Fp, Ds, S0
#     if not hasattr(arg,'fitS0'):
#         warnings.warn('arg.net_pars.parallel not defined. Using default of False')
#         arg.fitS0 = False  # indicates whether to fix S0 to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
#     if not hasattr(arg,'depth'):
#         warnings.warn('arg.net_pars.depth not defined. Using default value of 4')
#         arg.depth = 4  # number of layers
#     if not hasattr(arg, 'width'):
#         warnings.warn('arg.net_pars.width not defined. Using default of number of b-values')
#         arg.width = 0
#     return arg
#
#
# def checkarg_sim(arg):
#     if not hasattr(arg, 'bvalues'):
#         warnings.warn('arg.sim.bvalues not defined. Using default value of [0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700]')
#         arg.bvalues = [0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700]
#     if not hasattr(arg, 'repeats'):
#         warnings.warn('arg.sim.repeats not defined. Using default value of 1')
#         arg.repeats = 1  # this is the number of repeats for simulations
#     if not hasattr(arg, 'rician'):
#         warnings.warn('arg.sim.rician not defined. Using default of False')
#         arg.rician = False
#     if not hasattr(arg, 'SNR'):
#         warnings.warn('arg.sim.SNR not defined. Using default of [20]')
#         arg.SNR = [20]
#     if not hasattr(arg, 'sims'):
#         warnings.warn('arg.sim.sims not defined. Using default of 100000')
#         arg.sims = 100000
#     if not hasattr(arg, 'num_samples_eval'):
#         warnings.warn('arg.sim.num_samples_eval not defined. Using default of 100000')
#         arg.num_samples_eval = 100000
#     if not hasattr(arg, 'range'):
#         warnings.warn('arg.sim.range not defined. Using default of ([0.0005, 0.05, 0.01],[0.003, 0.4, 0.1])')
#         arg.range = ([0.0005, 0.05, 0.01],
#                   [0.003, 0.4, 0.1])
#     return arg
#
# def checkarg(arg):
#     if not hasattr(arg, 'fig'):
#         arg.fig = False
#         warnings.warn('arg.fig not defined. Using default of False')
#     if not hasattr(arg, 'save_name'):
#         warnings.warn('arg.save_name not defined. Using default of ''default''')
#         arg.save_name = 'default'
#     if not hasattr(arg,'net_pars'):
#         warnings.warn('arg no net_pars. Using default initialisation')
#         arg.net_pars = net_pars()
#     if not hasattr(arg, 'train_pars'):
#         warnings.warn('arg no train_pars. Using default initialisation')
#         arg.train_pars = train_pars()
#     if not hasattr(arg, 'sim'):
#         warnings.warn('arg no sim. Using default initialisation')
#         arg.sim = sim()
#     if not hasattr(arg, 'fit'):
#         warnings.warn('arg no lsq. Using default initialisation')
#         arg.fit = lsqfit()
#     arg.net_pars=checkarg_net_pars(arg.net_pars)
#     arg.train_pars = checkarg_train_pars(arg.train_pars)
#     arg.sim = checkarg_sim(arg.sim)
#     arg.fit = fit.checkarg_lsq(arg.fit)
#     return arg


