import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import matplotlib, os, copy
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from utils import isnan
from net import Net


class train_IVIM:
    def __init__(self, arg):
        # Initialising parameters
        self.best = 1e16
        self.num_bad_epochs = 0
        self.loss_train = []
        self.loss_val = []
        self.prev_lr = 0

        self.arg = arg

        torch.backends.cudnn.benchmark = True

    def learn_IVIM(self, X_train, bvalues, net=None):
        """
        This program builds a IVIM-NET network and trains it.
        :param X_train: 2D array of IVIM data we use for training. First axis are the voxels and second axis are the b-values
        :param bvalues: a 1D array with the b-values
        :param arg: an object with network design options, as explained in the publication check hyperparameters.py for
        options
        :param net: an optional input pre-trained network with initialized weights for e.g. transfer learning or warm start
        :return net: returns a trained network
        """
        X_train, S0 = self._preprocess_train_data(X_train, bvalues)
        net, bvalues = self._get_net(net, bvalues)
        trainloader, inferloader, totalit, batch_norm2 = self._get_data(X_train)
        criterion = self._get_loss()
        # defining optimiser
        if self.arg.train_pars.scheduler:
            optimizer, scheduler = self.load_optimizer(net)
        else:
            optimizer = self.load_optimizer(net)

        final_model = copy.deepcopy(net.state_dict())

        ## Train
        for epoch in range(self.arg.train_pars.max_epochs):
            print("-----------------------------------------------------------------")
            print("Epoch: {}; Bad epochs: {}".format(epoch, self.num_bad_epochs))
            # initialising and resetting parameters
            net.train()
            running_loss_train = 0.
            running_loss_val = 0.
            for i, X_batch in enumerate(tqdm(trainloader, position=0, leave=True, total=totalit), 0):
                if i > totalit:
                    # have a maximum number of batches per epoch to ensure regular updates of whether we are improving
                    break
                # zero the parameter gradients
                optimizer.zero_grad()
                # put batch on GPU if pressent
                X_batch = X_batch.to(self.arg.train_pars.device)
                ## forward + backward + optimize
                X_pred, Dt_pred, Fp_pred, Dp_pred, S0pred = net(X_batch)
                # removing nans and too high/low predictions to prevent overshooting
                X_pred = self._postprocess_data(X_pred)
                # determine loss for batch;
                # note that the loss is determined by the difference between the predicted signal and the actual signal.
                # The loss does not look at Dt, Dp or Fp.
                loss = criterion(X_pred, X_batch)
                # updating network
                loss.backward()
                optimizer.step()
                # total loss and determine max loss over all batches
                running_loss_train += loss.item()

            # show some figures if desired, to show whether there is a correlation between Dp and f
            # self._display_result(Dp_pred, Fp_pred)

            # after training, do validation in unseen data without updating gradients
            print('\n validation \n')
            net.eval()
            # validation is always done over all validation data
            for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
                optimizer.zero_grad()
                X_batch = X_batch.to(self.arg.train_pars.device)
                # do prediction, only look at predicted IVIM signal
                X_pred, _, _, _, _ = net(X_batch)
                X_pred = self._postprocess_data(X_pred)
                # validation loss
                loss = criterion(X_pred, X_batch)
                # running_loss_val.update(loss.item(), (32*self.arg.train_pars.batch_size))
                running_loss_val += loss.item()

            # scale losses
            running_loss_train = running_loss_train / totalit
            running_loss_val = running_loss_val / batch_norm2

            # save loss history for plot
            self.loss_train.append(running_loss_train)
            self.loss_val.append(running_loss_val)
            # as discussed in the article, LR is important. This approach allows to reduce the LR if we think it is too
            # high, and return to the network state before it went poorly
            if self.arg.train_pars.scheduler:
                scheduler.step(running_loss_val)
                if optimizer.param_groups[0]['lr'] < self.prev_lr:
                    net.load_state_dict(final_model)
                self.prev_lr = optimizer.param_groups[0]['lr']

            # print stuff
            print("\nLoss: {loss}, validation_loss: {val_loss}, lr: {lr}".format(loss=running_loss_train,
                                                                                 val_loss=running_loss_val,
                                                                                 lr=optimizer.param_groups[0]['lr']))
            # early stopping criteria
            if running_loss_val < self.best:
                print("\n############### Saving good model ###############################")
                final_model = copy.deepcopy(net.state_dict())
                self.best = running_loss_val
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs = self.num_bad_epochs + 1
                if self.num_bad_epochs == self.arg.train_pars.patience:
                    print("\nDone, best val loss: {}".format(self.best))
                    break

            # plot loss and plot 4 fitted curves
            if epoch % 5 == 0:
                # plot progress and intermediate results (if enabled)
                self.plot_progress(X_batch.cpu(), X_pred.cpu(), bvalues.cpu())
        print("Done")

        # save final fits
        self._save_results()
        # Restore best model
        if self.arg.train_pars.select_best:
            net.load_state_dict(final_model)
        del trainloader
        del inferloader
        if self.arg.train_pars.use_cuda:
            torch.cuda.empty_cache()
        return net

    @staticmethod
    def _preprocess_train_data(X_train, bvalues):
        ## normalise the signal to b=0 and remove data with nans
        S0 = np.mean(X_train[:, bvalues == 0], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        np.delete(X_train, isnan(np.mean(X_train, axis=1)), axis=0)
        # removing non-IVIM-like data; this often gets through when background data is not correctly masked
        # Estimating IVIM parameters in these data is meaningless anyways.
        X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
        X_train[X_train > 1.5] = 1.5

        return X_train, S0

    @staticmethod
    def _postprocess_data(X_pred):
        X_pred[isnan(X_pred)] = 0
        X_pred[X_pred < 0] = 0
        X_pred[X_pred > 3] = 3

        return X_pred

    def _get_net(self, net, bvalues):
        # initialising the network of choice using the input argument arg
        if net is None:
            bvalues = torch.FloatTensor(bvalues[:]).to(self.arg.train_pars.device)
            net = Net(bvalues, self.arg.net_pars).to(self.arg.train_pars.device)
        else:
            # if a network was used as input parameter, work with that network instead (transfer learning/warm start).
            net.to(self.arg.train_pars.device)

        return net, bvalues

    def _get_loss(self):
        # defining the loss function; not explored in the publication
        if self.arg.train_pars.loss_fun == 'rms':
            criterion = nn.MSELoss(reduction='mean').to(self.arg.train_pars.device)
        elif self.arg.train_pars.loss_fun == 'L1':
            criterion = nn.L1Loss(reduction='mean').to(self.arg.train_pars.device)

        return criterion

    def _get_data(self, X_train):
        # splitting data into learning and validation set; subsequently initialising the Dataloaders
        split = int(np.floor(len(X_train) * self.arg.train_pars.split))
        train_set, val_set = torch.utils.data.random_split(torch.from_numpy(X_train.astype(np.float32)),
                                                           [split, len(X_train) - split])
        # train loader loads the trianing data.
        # We want to shuffle to make sure data order is modified each epoch and different data is selected each epoch.
        trainloader = utils.DataLoader(train_set,
                                       batch_size=self.arg.train_pars.batch_size,
                                       shuffle=True,
                                       drop_last=True)
        # validation data is loaded here.
        # By not shuffling, we make sure the same data is loaded for validation every time.
        # We can use substantially more data per batch as we are not training.
        inferloader = utils.DataLoader(val_set,
                                       batch_size=32 * self.arg.train_pars.batch_size,
                                       shuffle=False,
                                       drop_last=True)

        # defining the number of training and validation batches for normalisation later
        totalit = np.min([self.arg.train_pars.maxit, np.floor(split // self.arg.train_pars.batch_size)])
        batch_norm2 = np.floor(len(val_set) // (32 * self.arg.train_pars.batch_size))

        return trainloader, inferloader, totalit, batch_norm2

    def load_optimizer(self, net):
        if self.arg.net_pars.parallel:
            if self.arg.net_pars.fitS0:
                par_list = [{'params': net.encoder0.parameters(), 'lr': self.arg.train_pars.lr},
                            {'params': net.encoder1.parameters()}, {'params': net.encoder2.parameters()},
                            {'params': net.encoder3.parameters()}]
            else:
                par_list = [{'params': net.encoder0.parameters(), 'lr': self.arg.train_pars.lr},
                            {'params': net.encoder1.parameters()}, {'params': net.encoder2.parameters()}]
        else:
            par_list = [{'params': net.encoder0.parameters()}]
        if self.arg.train_pars.optim == 'adam':
            optimizer = optim.Adam(par_list, lr=self.arg.train_pars.lr, weight_decay=1e-4)
        elif self.arg.train_pars.optim == 'sgd':
            optimizer = optim.SGD(par_list, lr=self.arg.train_pars.lr, momentum=0.9, weight_decay=1e-4)
        elif self.arg.train_pars.optim == 'adagrad':
            optimizer = torch.optim.Adagrad(par_list, lr=self.arg.train_pars.lr, weight_decay=1e-4)
        if self.arg.train_pars.scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, threshold=1e-3,
                                                             patience=round(self.arg.train_pars.patience / 2), verbose=True)
            return optimizer, scheduler
        else:
            return optimizer

    def plot_progress(self, X_batch, X_pred, bvalues):
        # this program plots the progress of the training. It will plot the loss and validatin loss, as well as 4 IVIM curve
        # fits to 4 data points from the input
        inds1 = np.argsort(bvalues)
        X_batch = X_batch[:, inds1]
        X_pred = X_pred[:, inds1]
        bvalues = bvalues[inds1]
        if self.arg.fig:
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
            plt.plot(self.loss_train)
            plt.plot(self.loss_val)
            plt.yscale("log")
            plt.xlabel('epoch #')
            plt.ylabel('loss')
            plt.legend(('training loss', 'validation loss (after training epoch)'))
            plt.ion()
            plt.show()
            plt.pause(0.001)

    def _display_result(self, Dp_pred, Fp_pred):
        if self.arg.fig:
            plt.figure(3)
            plt.clf()
            plt.plot(Dp_pred.tolist(), Fp_pred.tolist(), 'rx', markersize=5)
            plt.ion()
            plt.show()

    def _save_results(self):
        if self.arg.fig:
            if not os.path.isdir('plots'):
                os.makedirs('plots')
            plt.figure(1)
            plt.gcf()
            plt.savefig('plots/fig_fit.png')
            plt.figure(2)
            plt.gcf()
            plt.savefig('plots/fig_train.png')
            plt.close('all')
