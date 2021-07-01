"""
September 2020 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion
Code is uploaded as part of our publication in MRM
 (Kaandorp et al. Improved physics-informed deep learning of the intravoxel-incoherent motion model: accurate, unique and consistent. MRM 2021)
requirements:
numpy
torch
tqdm
matplotlib
scipy
joblib
"""

# import libraries
import numpy as np
import fitting_algorithms as fit
import time
import torch
import matplotlib
from train import train_IVIM
matplotlib.use('TkAgg')
from predict import predict_IVIM
from eval import plot_results, plots_correlation
from scipy.stats import spearmanr


class sim:
    def __init__(self, SNR, arg):
        """ This function defines how well the different fit approaches perform on simulated data. Data is simulated by
        randomly selecting a value of D, f and D* from within the predefined range. The script calculates the random,
        systematic, root-mean-squared error (RMSE) and Spearman Rank correlation coefficient for each of the IVIM parameters.
        Furthermore, it calculates the stability of the neural network (when trained multiple times).
        Relevant attributes are:
        arg.sim.sims = number of simulations to be performed (need a large amount for training)
        arg.sim.num_samples_eval = number of samples to evaluate (save time for lsq fitting)
        arg.sim.repeats = number of times to repeat the training and evaluation of the network (to assess stability)
        arg.sim.bvalues: 1D Array of b-values used
        arg.fit contains the parameters regarding lsq fitting
        arg.train_pars and arg.net_pars contain the parameters regarding the neural network
        arg.sim.range gives the simulated range of D, f and D* in a 2D array
        :return matlsq: 2D array containing the performance of the lsq fit (if enabled). The rows indicate D, f (Fp), D*
        (Dp), whereas the colums give the mean input value, the random error and the systematic error
        :return matNN: 2D array containing the performance of the NN. The rows indicate D, f (Fp), D*
        (Dp), whereas the colums give the mean input value, the random error and the systematic error
        :return stability: a 1D array with the stability of D, f and D* as a fraction of their mean value.
        Stability is only relevant for neural networks and is calculated from the repeated network training.
        """

        self.SNR = SNR
        self.arg = arg

        # this simulated the signal
        self.IVIM_signal_noisy, self.D, self.f, self.Dp = self.sim_signal(self.arg.sim.bvalues, sims=self.arg.sim.sims,
                                                      Dmin=self.arg.sim.range[0][0],
                                                      Dmax=self.arg.sim.range[1][0], fmin=self.arg.sim.range[0][1],
                                                      fmax=self.arg.sim.range[1][1], Dsmin=self.arg.sim.range[0][2],
                                                      Dsmax=self.arg.sim.range[1][2], rician=self.arg.sim.rician)

        # only remember the D, Dp and f needed for evaluation
        self.D = np.squeeze(self.D[:self.arg.sim.num_samples_eval])
        self.Dp = np.squeeze(self.Dp[:self.arg.sim.num_samples_eval])
        self.f = np.squeeze(self.f[:self.arg.sim.num_samples_eval])
        plots_correlation(self.D, self.f, self.Dp, 'real')

        self.cv_mat = np.zeros([self.arg.sim.repeats, 3])

    def run_NN(self):
        # prepare a larger array in case we repeat training
        if self.arg.sim.repeats > 1:
            paramsNN = np.zeros([self.arg.sim.repeats, 4, self.arg.sim.num_samples_eval])
        else:
            paramsNN = np.zeros([4, self.arg.sim.num_samples_eval])

        # loop over repeats
        for aa in range(self.arg.sim.repeats):
            start_time = time.time()
            # train network
            print('\nRepeat: {repeat}\n'.format(repeat=aa))
            train = train_IVIM(self.arg)
            net = train.learn_IVIM(self.IVIM_signal_noisy, self.arg.sim.bvalues)
            # net = learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg)
            elapsed_time = time.time() - start_time
            print('\ntime elapsed for training: {}\n'.format(elapsed_time))
            start_time = time.time()
            # predict parameters
            if self.arg.sim.repeats > 1:
                paramsNN[aa] = predict_IVIM(self.IVIM_signal_noisy[:self.arg.sim.num_samples_eval, :],
                                            self.arg.sim.bvalues, net, self.arg)
            else:
                paramsNN = predict_IVIM(self.IVIM_signal_noisy[:self.arg.sim.num_samples_eval, :], self.arg.sim.bvalues,
                                        net, self.arg)

            elapsed_time = time.time() - start_time
            print('\ntime elapsed for inference: {}\n'.format(elapsed_time))
            # remove network to save memory
            del net
            if self.arg.train_pars.use_cuda:
                torch.cuda.empty_cache()

        print('results for NN')
        # if we repeat training, then evaluate stability
        if self.arg.sim.repeats > 1:
            matNN = np.zeros([self.arg.sim.repeats, 3, 4])
            for aa in range(self.arg.sim.repeats):
                # determine errors and Spearman Rank
                matNN[aa] = self.print_errors(paramsNN[aa], aa)
                plot_results(self.D, self.f, self.Dp, paramsNN[aa], 'NN' + str(aa))
            matNN = np.mean(matNN, axis=0)
            # calculate Stability Factor
            stability = np.sqrt(np.mean(np.square(np.std(paramsNN, axis=0)), axis=1))
            stability = stability[[0, 1, 2]] / [np.mean(self.D), np.mean(self.f), np.mean(self.Dp)]
        else:
            matNN = self.print_errors(paramsNN)
            stability = np.zeros(3)
            # show figures if requested
            plot_results(self.D, self.f, self.Dp, paramsNN, 'NN')

        return matNN, stability

    def run_fit(self):
        start_time = time.time()
        # all fitting is done in the fit.fit_dats for the other fitting algorithms (lsq, segmented and Baysesian)
        paramsf = fit.fit_dats(self.arg.sim.bvalues, self.IVIM_signal_noisy[:self.arg.sim.num_samples_eval, :], self.arg.fit)
        elapsed_time = time.time() - start_time
        print('\ntime elapsed for fit: {}\n'.format(elapsed_time))
        print('results for fit')

        # determine errors and Spearman Rank
        matlsq = self.print_errors(paramsf)
        # show figures if requested
        plot_results(self.D, self.f, self.Dp, paramsf, 'LS')

        return matlsq

    def print_errors(self, params, aa=0):
        # this function calculates and prints the random, systematic, root-mean-squared (RMSE) errors and Spearman Rank correlation coefficient
        rmse_D = np.sqrt(np.square(np.subtract(self.D, params[0])).mean())
        rmse_f = np.sqrt(np.square(np.subtract(self.f, params[1])).mean())
        rmse_Dp = np.sqrt(np.square(np.subtract(self.Dp, params[2])).mean())

        # initialise Spearman Rank matrix
        Spearman = np.zeros([3, 2])
        # calculate Spearman Rank correlation coefficient and p-value
        Spearman[0, 0], Spearman[0, 1] = spearmanr(params[0], params[2])  # DvDp
        Spearman[1, 0], Spearman[1, 1] = spearmanr(params[0], params[1])  # Dvf
        Spearman[2, 0], Spearman[2, 1] = spearmanr(params[1], params[2])  # fvDp
        # If spearman is nan, set as 1 (because of constant estimated IVIM parameters)
        Spearman[np.isnan(Spearman)] = 1
        # take absolute Spearman
        Spearman = np.absolute(Spearman)
        del params

        normD_lsq = np.mean(self.D)
        normf_lsq = np.mean(self.f)
        normDp_lsq = np.mean(self.Dp)

        self.cv_mat[aa, 0] = np.square(rmse_D)
        self.cv_mat[aa, 1] = np.square(rmse_f)
        self.cv_mat[aa, 2] = np.square(rmse_Dp)

        cv_result = np.sqrt(np.mean(self.cv_mat, axis=0))

        mats = np.array([
            [normD_lsq, rmse_D / normD_lsq, Spearman[0, 0], cv_result[0] / normD_lsq],
            [normf_lsq, rmse_f / normf_lsq, Spearman[1, 0], cv_result[1] / normf_lsq],
            [normDp_lsq, rmse_Dp / normDp_lsq, Spearman[2, 0], cv_result[2] / normDp_lsq]
        ])

        print(
            '\nresults from NN: columns show themean, the SD/mean, the systematic error/mean, the RMSE/mean and the Spearman coef [DvDp,Dvf,fvDp] \n'
            'the rows show D, f and D*\n')
        print([normD_lsq, '  ', rmse_D / normD_lsq, ' ', Spearman[0, 0], cv_result[0]])
        print([normf_lsq, '  ', rmse_f / normf_lsq, ' ', Spearman[1, 0], cv_result[1]])
        print([normDp_lsq, '  ', rmse_Dp / normDp_lsq, ' ', Spearman[2, 0], cv_result[2]])

        return mats

    def sim_signal(self, bvalues, sims=100000, Dmin=0.5 / 1000, Dmax=2.0 / 1000, fmin=0.1, fmax=0.5, Dsmin=0.05,
                   Dsmax=0.2,
                   rician=False, state=123):
        """
        This simulates IVIM curves. Data is simulated by randomly selecting a value of D, f and D* from within the
        predefined range.
        input:
        :param SNR: SNR of the simulated data. If SNR is set to 0, no noise is added
        :param bvalues: 1D Array of b-values used
        :param sims: number of simulations to be performed (need a large amount for training)
        optional:
        :param Dmin: minimal simulated D. Default = 0.0005
        :param Dmax: maximal simulated D. Default = 0.002
        :param fmin: minimal simulated f. Default = 0.1
        :param Dmax: minimal simulated f. Default = 0.5
        :param Dpmin: minimal simulated D*. Default = 0.05
        :param Dpmax: minimal simulated D*. Default = 0.2
        :param rician: boolean giving whether Rician noise is used; default = False
        :return data_sim: 2D array with noisy IVIM signal (x-axis is sims long, y-axis is len(b-values) long)
        :return D: 1D array with the used D for simulations, sims long
        :return f: 1D array with the used f for simulations, sims long
        :return Dp: 1D array with the used D* for simulations, sims long
        """

        # randomly select parameters from predefined range
        rg = np.random.RandomState(state)
        test = rg.uniform(0, 1, (sims, 1))
        D = Dmin + (test * (Dmax - Dmin))
        test = rg.uniform(0, 1, (sims, 1))
        f = fmin + (test * (fmax - fmin))
        test = rg.uniform(0, 1, (sims, 1))
        Dp = Dsmin + (test * (Dsmax - Dsmin))

        # initialise data array
        data_sim = np.zeros([len(D), len(bvalues)])
        bvalues = np.array(bvalues)

        # loop over array to fill with simulated IVIM data
        for aa in range(len(D)):
            data_sim[aa, :] = fit.ivim(bvalues, D[aa][0], f[aa][0], Dp[aa][0], 1)

        # if SNR is set to zero, don't add noise
        if self.SNR > 0:
            # initialise noise arrays
            noise_imag = np.zeros([sims, len(bvalues)])
            noise_real = np.zeros([sims, len(bvalues)])
            # fill arrays
            for i in range(0, sims - 1):
                noise_real[i,] = rg.normal(0, 1 / self.SNR,
                                           (1, len(
                                               bvalues)))  # wrong! need a SD per input. Might need to loop to maD noise
                noise_imag[i,] = rg.normal(0, 1 / self.SNR, (1, len(bvalues)))
            if rician:
                # add Rician noise as the square root of squared gaussian distributed real signal + noise and imaginary noise
                data_sim = np.sqrt(np.power(data_sim + noise_real, 2) + np.power(noise_imag, 2))
            else:
                # or add Gaussian noise
                data_sim = data_sim + noise_imag
        else:
            data_sim = data_sim

        # normalise signal
        S0_noisy = np.mean(data_sim[:, bvalues == 0], axis=1)
        data_sim = data_sim / S0_noisy[:, None]
        return data_sim, D, f, Dp
