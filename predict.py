
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from utils import isnan
import torch.utils.data as utils
import torch
from tqdm import tqdm
import copy




def predict_IVIM(data, bvalues, net, arg):
    """
    This program takes a trained network and predicts the IVIM parameters from it.
    :param data: 2D array of IVIM data we want to predict the IVIM parameters from.
    First axis are the voxels and second axis are the b-values
    :param bvalues: a 1D array with the b-values
    :param net: the trained IVIM-NET network
    :param arg: an object with network design options, as explained in the publication check hyperparameters.py for
    options
    :return param: returns the predicted parameters
    """
    ## normalise the signal to b=0 and remove data with nans
    S0 = np.mean(data[:, bvalues == 0], axis=1).astype('<f')
    data = data / S0[:, None]
    np.delete(data, isnan(np.mean(data, axis=1)), axis=0)
    # skip nans.
    mylist = isnan(np.mean(data, axis=1))
    sels = [not i for i in mylist]

    # remove data with non-IVIM-like behaviour. Estimating IVIM parameters in these data is meaningless anyways.
    sels = sels & (np.percentile(data[:, bvalues < 50], 0.95, axis=1) < 1.3) & (
                np.percentile(data[:, bvalues > 50], 0.95, axis=1) < 1.2) & (
                       np.percentile(data[:, bvalues > 150], 0.95, axis=1) < 1.0)
    # we need this for later
    lend = len(data)
    data = data[sels]

    # tell net it is used for evaluation
    net.eval()
    # initialise parameters and data
    Dp = np.array([])
    Dt = np.array([])
    Fp = np.array([])
    S0 = np.array([])
    # initialise dataloader. Batch size can be way larger as we are still training.
    inferloader = utils.DataLoader(torch.from_numpy(data.astype(np.float32)),
                                   batch_size=2056,
                                   shuffle=False,
                                   drop_last=False)
    # start predicting
    with torch.no_grad():
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            X_batch = X_batch.to(arg.train_pars.device)
            # here the signal is predicted. Note that we now are interested in the parameters and no longer in the predicted signal decay.
            _, Dtt, Fpt, Dpt, S0t = net(X_batch)
            # Quick and dirty solution to deal with networks not predicting S0
            try:
                S0 = np.append(S0, (S0t.cpu()).numpy())
            except:
                S0 = np.append(S0, S0t)
            Dp = np.append(Dp, (Dpt.cpu()).numpy())
            Dt = np.append(Dt, (Dtt.cpu()).numpy())
            Fp = np.append(Fp, (Fpt.cpu()).numpy())

    # The 'abs' and 'none' constraint networks have no way of figuring out what is D and D* a-priori. However, they do
    # tend to pick one output parameter for D or D* consistently within the network. If the network has swapped D and
    # D*, we swap them back here.
    if np.mean(Dp) < np.mean(Dt):
        Dp22 = copy.deepcopy(Dt)
        Dt = Dp
        Dp = Dp22
        Fp = 1 - Fp
    # here we correct for the data that initially was removed as it did not have IVIM behaviour, by returning zero
    # estimates
    Dptrue = np.zeros(lend)
    Dttrue = np.zeros(lend)
    Fptrue = np.zeros(lend)
    S0true = np.zeros(lend)
    Dptrue[sels] = Dp
    Dttrue[sels] = Dt
    Fptrue[sels] = Fp
    S0true[sels] = S0
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    return [Dttrue, Fptrue, Dptrue, S0true]

