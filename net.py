import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')


class Net(nn.Module):
    def __init__(self, bvalues, net_pars):
        """
        this defines the Net class which is the network we want to train.
        :param bvalues: a 1D array with the b-values
        :param net_pars: an object with network design options, as explained in the publication, with attributes:
        fitS0 --> Boolean determining whether S0 is fixed to 1 (False) or fitted (True)
        times len(bvalues), with data sorted per voxel. This option was not explored in the publication
        dropout --> Number between 0 and 1 indicating the amount of dropout regularisation
        norm --> string determining batch method to use
        parallel --> Boolean determining whether to use separate networks for estimating the different IVIM parameters
        (True), or have them all estimated by a single network (False)
        con --> string which determines what type of constraint is used for the parameters. Options are:
        'sigmoid' allowing a sigmoid constraint
        'abs' having the absolute of the estimated values to constrain parameters to be positive
        'none' giving no constraints
        cons_min --> 1D array, if sigmoid is the constraint, these values give [Dmin, fmin, D*min, S0min]
        cons_max --> 1D array, if sigmoid is the constraint, these values give [Dmax, fmax, D*max, S0max]
        depth --> integer giving the network depth (number of layers)
        """
        super(Net, self).__init__()
        self.bvalues = bvalues
        self.net_pars = net_pars
        if self.net_pars.width is 0:
            self.net_pars.width = len(bvalues)
        # define number of parameters being estimated
        self.est_pars = 3
        if self.net_pars.fitS0:
            self.est_pars += 1
        # define module lists. If network is not parallel, we can do with 1 list, otherwise we need a list per parameter
        self.fc_layers0 = nn.ModuleList()
        if self.net_pars.parallel:
            self.fc_layers1 = nn.ModuleList()
            self.fc_layers2 = nn.ModuleList()
            self.fc_layers3 = nn.ModuleList()
        # loop over the layers
        width = len(bvalues)
        for i in range(self.net_pars.depth):
            # extend with a fully-connected linear layer
            self.fc_layers0.extend([nn.Linear(width, self.net_pars.width)])
            if self.net_pars.parallel:
                self.fc_layers1.extend([nn.Linear(width, self.net_pars.width)])
                self.fc_layers2.extend([nn.Linear(width, self.net_pars.width)])
                self.fc_layers3.extend([nn.Linear(width, self.net_pars.width)])
            width = self.net_pars.width
            # if desired, add batch normalisation
            if self.net_pars.norm == 'batch':
                self.fc_layers0.extend([nn.BatchNorm1d(self.net_pars.width)])
                if self.net_pars.parallel:
                    self.fc_layers1.extend([nn.BatchNorm1d(self.net_pars.width)])
                    self.fc_layers2.extend([nn.BatchNorm1d(self.net_pars.width)])
                    self.fc_layers3.extend([nn.BatchNorm1d(self.net_pars.width)])
            elif self.net_pars.norm == 'group':
                self.fc_layers0.extend([nn.BatchNorm1d(self.net_pars.width)])
                if self.net_pars.parallel:
                    self.fc_layers1.extend([nn.GroupNorm(num_channels=self.net_pars.width, num_groups=net_pars.num_group)])
                    self.fc_layers2.extend([nn.GroupNorm(num_channels=self.net_pars.width, num_groups=net_pars.num_group)])
                    self.fc_layers3.extend([nn.GroupNorm(num_channels=self.net_pars.width, num_groups=net_pars.num_group)])
            elif self.net_pars.norm == 'instance':
                self.fc_layers0.extend([nn.BatchNorm1d(self.net_pars.width)])
                if self.net_pars.parallel:
                    self.fc_layers1.extend([nn.InstanceNorm1d(self.net_pars.width)])
                    self.fc_layers2.extend([nn.InstanceNorm1d(self.net_pars.width)])
                    self.fc_layers3.extend([nn.InstanceNorm1d(self.net_pars.width)])
            elif self.net_pars.norm == 'layer':
                self.fc_layers0.extend([nn.BatchNorm1d(self.net_pars.width)])
                if self.net_pars.parallel:
                    self.fc_layers1.extend([nn.LayerNorm(self.net_pars.width)])
                    self.fc_layers2.extend([nn.LayerNorm(self.net_pars.width)])
                    self.fc_layers3.extend([nn.LayerNorm(self.net_pars.width)])
            # add ELU units for non-linearity
            self.fc_layers0.extend([nn.ELU()])
            if self.net_pars.parallel:
                self.fc_layers1.extend([nn.ELU()])
                self.fc_layers2.extend([nn.ELU()])
                self.fc_layers3.extend([nn.ELU()])
            # if dropout is desired, add dropout regularisation
            if self.net_pars.dropout is not 0 and i is not (self.net_pars.depth - 1):
                self.fc_layers0.extend([nn.Dropout(self.net_pars.dropout)])
                if self.net_pars.parallel:
                    self.fc_layers1.extend([nn.Dropout(self.net_pars.dropout)])
                    self.fc_layers2.extend([nn.Dropout(self.net_pars.dropout)])
                    self.fc_layers3.extend([nn.Dropout(self.net_pars.dropout)])
        # Final layer yielding output, with either 3 (fix S0) or 4 outputs of a single network, or 1 output
        # per network in case of parallel networks.
        if self.net_pars.parallel:
            self.encoder0 = nn.Sequential(*self.fc_layers0, nn.Linear(self.net_pars.width, 1))
            self.encoder1 = nn.Sequential(*self.fc_layers1, nn.Linear(self.net_pars.width, 1))
            self.encoder2 = nn.Sequential(*self.fc_layers2, nn.Linear(self.net_pars.width, 1))
            if self.net_pars.fitS0:
                self.encoder3 = nn.Sequential(*self.fc_layers3, nn.Linear(self.net_pars.width, 1))
        else:
            self.encoder0 = nn.Sequential(*self.fc_layers0, nn.Linear(self.net_pars.width, self.est_pars))

    def forward(self, X):
        # select constraint method
        if self.net_pars.con == 'sigmoid':
            # define constraints
            Dmin = self.net_pars.cons_min[0]
            Dmax = self.net_pars.cons_max[0]
            fmin = self.net_pars.cons_min[1]
            fmax = self.net_pars.cons_max[1]
            Dpmin = self.net_pars.cons_min[2]
            Dpmax = self.net_pars.cons_max[2]
            S0min = self.net_pars.cons_min[3]
            S0max = self.net_pars.cons_max[3]
            # this network constrains the estimated parameters between two values by taking the sigmoid.
            # Advantage is that the parameters are constrained and that the mapping is unique.
            # Disadvantage is that the gradients go to zero close to the prameter bounds.
            params0 = self.encoder0(X)
            # if parallel again use each param comes from a different output
            if self.net_pars.parallel:
                params1 = self.encoder1(X)
                params2 = self.encoder2(X)
                if self.net_pars.fitS0:
                    params3 = self.encoder3(X)
        elif self.net_pars.con == 'none' or self.net_pars.con == 'abs':
            if self.net_pars.con == 'abs':
                # this network constrains the estimated parameters to be positive by taking the absolute.
                # Advantage is that the parameters are constrained and that the derrivative of the function remains
                # constant. Disadvantage is that -x=x, so could become unstable.
                params0 = torch.abs(self.encoder0(X))
                if self.net_pars.parallel:
                    params1 = torch.abs(self.encoder1(X))
                    params2 = torch.abs(self.encoder2(X))
                    if self.net_pars.fitS0:
                        params3 = torch.abs(self.encoder3(X))
            else:
                # this network is not constraint
                params0 = self.encoder0(X)
                if self.net_pars.parallel:
                    params1 = self.encoder1(X)
                    params2 = self.encoder2(X)
                    if self.net_pars.fitS0:
                        params3 = self.encoder3(X)
        else:
            raise Exception('the chose parameter constraint is not implemented. Try ''sigmoid'', ''none'' or ''abs''')

        X_temp = []

        if self.net_pars.con == 'sigmoid':
            # applying constraints
            if self.net_pars.parallel:
                Dp = Dpmin + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (Dpmax - Dpmin)
                Dt = Dmin + torch.sigmoid(params1[:, 0].unsqueeze(1)) * (Dmax - Dmin)
                Fp = fmin + torch.sigmoid(params2[:, 0].unsqueeze(1)) * (fmax - fmin)
                if self.net_pars.fitS0:
                    S0 = S0min + torch.sigmoid(params3[:, 0].unsqueeze(1)) * (S0max - S0min)
            else:
                Dp = Dpmin + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (Dpmax - Dpmin)
                Dt = Dmin + torch.sigmoid(params0[:, 1].unsqueeze(1)) * (Dmax - Dmin)
                Fp = fmin + torch.sigmoid(params0[:, 2].unsqueeze(1)) * (fmax - fmin)
                if self.net_pars.fitS0:
                    S0 = S0min + torch.sigmoid(params0[:, 3].unsqueeze(1)) * (S0max - S0min)
        elif self.net_pars.con == 'none' or self.net_pars.con == 'abs':
            if self.net_pars.parallel:
                Dp = params0[:, 0].unsqueeze(1)
                Dt = params1[:, 0].unsqueeze(1)
                Fp = params2[:, 0].unsqueeze(1)
                if self.net_pars.fitS0:
                    S0 = params3[:, 0].unsqueeze(1)
            else:
                Dp = params0[:, 0].unsqueeze(1)
                Dt = params0[:, 1].unsqueeze(1)
                Fp = params0[:, 2].unsqueeze(1)
                if self.net_pars.fitS0:
                    S0 = params0[:, 3].unsqueeze(1)
        # here we estimate X, the signal as function of b-values given the predicted IVIM parameters. Although
        # this parameter is not interesting for prediction, it is used in the loss function
        # in this a>0 case, we fill up the predicted signal of the neighbouring voxels too, as these are used in
        # the loss function.

        if self.net_pars.fitS0:
            X_temp.append(S0 * (Fp * torch.exp(-self.bvalues * Dp) + (1 - Fp) * torch.exp(-self.bvalues * Dt)))
        else:
            X_temp.append((Fp * torch.exp(-self.bvalues * Dp) + (1 - Fp) * torch.exp(-self.bvalues * Dt)))
        X = torch.cat(X_temp,dim=1)
        if self.net_pars.fitS0:
            return X, Dt, Fp, Dp, S0
        else:
            return X, Dt, Fp, Dp, torch.ones(len(Dt))


