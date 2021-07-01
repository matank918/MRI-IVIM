# MRI-IVIM

ivim deep lerning net based on the work of Oliver Gurney-Champion & Misha Kaandorp in https://github.com/oliverchampion/IVIMNET

runnig example1 file start run of the net with the baseline parameters  
runnig example2 file reproduce our Dropout optimization serch 

the model ran on simulated data. SNR level is hyperparameter. 

all parmaeters of the net, training scheme and simulation can be cahnge in the config file. 
figure result are saved to plots folder. 

Net perfeomnce messure are displayed in 3-4 matix for mean, NRMSE, covariance, and the normalized coefficient of variation over the repeated trainings for the IVIM parmeters D, D* and f.
