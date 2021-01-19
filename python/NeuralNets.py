import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
import keras
from pickle import load, dump
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from variables import nominal_vars, gen_vars, gev_vars
from Utils import Utils
from DataHandling import ROOTHelpers
from PlottingUtils import Plotter

class LSTM_DNN(object):
    """ 
    Class for training a DNN that uses LSTM and fully connected layers

    :param data_obj: instance of ROOTHelpers class. containing Dataframes for simulated signal, simulated background, and possibly data
    :type data_obj: ROOTHelpers
    :param low_level_vars: 2d list of low-level objects used as inputs to LSTM network layers
    :type low_level_vars: list
    :param high_level_vars: 1d list of high-level objects used as inputs to fully connected network layers
    :type high_level_vars: list
    :param train_frac: fraction of events to train the network. Test on 1-train_frac
    :type train_frac: float
    :param eq_weights: whether to train with the sum of signal weights scaled equal to the sum of background weights
    :type eq_weights: bool
    :param batch_boost: option to increase batch size based on ROC improvement. Needed for submitting to IC computing batch in hyper-parameter optimisation
    :type batch_boost: bool

    """ 

    def __init__(self, data_obj, low_level_vars, high_level_vars, train_frac, eq_weights=True, batch_boost=False):
        self.data_obj            = data_obj
        self.low_level_vars      = low_level_vars
        self.low_level_vars_flat = [var for sublist in low_level_vars for var in sublist]
        self.high_level_vars     = high_level_vars
        self.train_frac          = train_frac
        self.batch_boost         = batch_boost #needed for HP opt
        self.eq_train            = eq_weights
        self.max_epochs          = 100

        self.X_tot               = None
        self.y_tot               = None

        self.X_train_low_level   = None
        self.X_train_high_level  = None
        self.y_train             = None
        self.train_weights       = None
        self.train_eqw           = None
        self.proc_arr_train      = None
        self.y_pred_train        = None

        self.X_test_low_level    = None
        self.X_test_high_level   = None
        self.y_test              = None
        self.test_weights        = None
        self.proc_arr_test       = None
        self.y_pred_test         = None

        self.X_train_low_level   = None
        self.X_valid_low_level   = None
        self.y_valid             = None
        self.valid_weights       = None

        self.X_data_train_low_level  = None
        self.X_data_train_high_level = None

        self.X_data_test_low_level   = None
        self.X_data_test_high_level  = None
        
        #self.set_model(n_lstm_layers=1, n_lstm_nodes=150, n_dense_1=2, n_nodes_dense_1=300, 
        #               n_dense_2=3, n_nodes_dense_2=200, dropout_rate=0.2,
        #               learning_rate=0.001, batch_norm=True, batch_momentum=0.99)

        self.set_model(n_lstm_layers=1, n_lstm_nodes=100, n_dense_1=1, n_nodes_dense_1=100, 
                       n_dense_2=2, n_nodes_dense_2=50, dropout_rate=0.2,
                       learning_rate=0.001, batch_norm=True, batch_momentum=0.99)

        self.hp_grid_rnge           = {'n_lstm_layers': [1,2,3], 'n_lstm_nodes':[100,150,200], 
                                       'n_dense_1':[1,2,3], 'n_nodes_dense_1':[100,200,300],
                                       'n_dense_2':[1,2,3,4], 'n_nodes_dense_2':[100,200,300], 
                                       'dropout_rate':[0.1,0.2,0.3]
                                      }

        #assign plotter attribute before data_obj is deleted for mem
        self.plotter = Plotter(data_obj, self.low_level_vars_flat+self.high_level_vars)
        del data_obj

    def var_transform(self, do_data=False):
        """
        Apply natural log to GeV variables and change empty variable default values. Do this for signal, background, and potentially data
        
        Arguments
        ---------
        do_data : bool
            whether to apply the transforms to X_train in data. Used if plotting the DNN output score distribution in data
        
        """

        if 'subsubleadJetPt' in (self.low_level_vars_flat+self.high_level_vars):
            self.data_obj.mc_df_sig['subsubleadJetPt'] = self.data_obj.mc_df_sig['subsubleadJetPt'].replace(-9999., 1) #FIXME: zero after logging... so looks like a normal Z-scaled jet! fix this
            self.data_obj.mc_df_bkg['subsubleadJetPt'] = self.data_obj.mc_df_bkg['subsubleadJetPt'].replace(-9999., 1) #zero after logging
            if do_data: self.data_obj.data_df['subsubleadJetPt'] = self.data_obj.data_df['subsubleadJetPt'].replace(-9999., 1) #zero after logging

        #df['subsubleadJetEta'] = df['subsubleadJetEta'].replace(-9999., -10) #angles can't be zero because its still meaningfull. ?
        #df['subsubleadJetPhi'] = df['subsubleadJetPhi'].replace(-9999., -10)
        #df['subsubleadJetQGL'] = df['subsubleadJetQGL'].replace(-9999., -10) 

        for var in gev_vars:
            if var in (self.low_level_vars_flat+self.high_level_vars):
                self.data_obj.mc_df_sig[var] = np.log(self.data_obj.mc_df_sig[var].values)
                self.data_obj.mc_df_bkg[var] = np.log(self.data_obj.mc_df_bkg[var].values)
                if do_data: self.data_obj.data_df[var]   = np.log(self.data_obj.data_df[var].values)

    def create_X_y(self, mass_res_reweight=True):
        """
        Create X and y matrices to be used later for training and testing. 

        Arguments
        ---------
        mass_res_reweight: bool 
            re-weight signal events by 1/sigma(m_ee), in training only. Currently only implemented if also equalising weights,

        Returns
        --------
        X_tot: pandas dataframe of both low-level and high-level featues. Low-level features are returned as 1D columns.
        y_tot: numpy ndarray of the target column (1 for signal, 0 for background)
        """
        
        if self.eq_train:
            if mass_res_reweight:
                self.data_obj.mc_df_sig['MoM_weight'] = (self.data_obj.mc_df_sig['weight']) * (1./self.data_obj.mc_df_sig['dielectronSigmaMoM'])
                b_to_s_ratio = np.sum(self.data_obj.mc_df_bkg['weight'].values)/np.sum(self.data_obj.mc_df_sig['MoM_weight'].values)
                self.data_obj.mc_df_sig['eq_weight'] = (self.data_obj.mc_df_sig['MoM_weight']) * (b_to_s_ratio)
            else:
                b_to_s_ratio = np.sum(self.data_obj.mc_df_bkg['weight'].values)/np.sum(self.data_obj.mc_df_sig['weight'].values)
                self.data_obj.mc_df_sig['eq_weight'] = self.data_obj.mc_df_sig['weight'] * b_to_s_ratio 
            self.data_obj.mc_df_bkg['eq_weight'] = self.data_obj.mc_df_bkg['weight'] 

        self.data_obj.mc_df_sig.reset_index(drop=True, inplace=True)
        self.data_obj.mc_df_bkg.reset_index(drop=True, inplace=True)
        X_tot = pd.concat([self.data_obj.mc_df_sig, self.data_obj.mc_df_bkg], ignore_index=True)

        #add y_target label (1 for signal, 0 for background). Keep separate from X-train until after Z-scaling
        y_sig = np.ones(self.data_obj.mc_df_sig.shape[0])
        y_bkg = np.zeros(self.data_obj.mc_df_bkg.shape[0])
        y_tot = np.concatenate((y_sig,y_bkg))

        return X_tot, y_tot

    def split_X_y(self, X_tot, y_tot, do_data=False):
        """
        Split X and y matrices into a set for training the LSTM, and testing set to evaluate model performance

        Arguments
        ---------
        X_tot: pandas Dataframe
            pandas dataframe of both low-level and high-level featues. Low-level features are returned as 1D columns.
        y_tot: numpy ndarray 
            numpy ndarray of the target column (1 for signal, 0 for background)
        do_data : bool
            whether to form a test (and train) dataset in data, to use for plotting
        """

        if not self.eq_train:
            self.all_vars_X_train, self.all_vars_X_test, self.train_weights, self.test_weights, self.y_train, self.y_test, self.proc_arr_train, self.proc_arr_test =  train_test_split(X_tot[self.low_level_vars_flat+self.high_level_vars], 
                                                                                                                                                           X_tot['weight'], 
                                                                                                                                                           y_tot,
                                                                                                                                                           X_tot['proc'],
                                                                                                                                                           train_size=self.train_frac, test_size=1-self.train_frac, shuffle=True, random_state=1357
                                                                                                                                                          )
        else:
            self.all_vars_X_train, self.all_vars_X_test, self.train_weights, self.test_weights, self.train_eqw, self.test_eqw, self.y_train, self.y_test, self.proc_arr_train, self.proc_arr_test = train_test_split(X_tot[self.low_level_vars_flat+self.high_level_vars], 
                                                                                                                                                                                        X_tot['weight'],
                                                                                                                                                                                        X_tot['eq_weight'], 
                                                                                                                                                                                        y_tot, 
                                                                                                                                                                                        X_tot['proc'],
                                                                                                                                                                                        train_size=self.train_frac, test_size=1-self.train_frac, shuffle=True, random_state=1357
                                                                                                                                                                                        )
            self.train_weights_eq = self.train_eqw.values


        if do_data: #for plotting purposes
            self.all_X_data_train, self.all_X_data_test  = train_test_split(self.data_obj.data_df[self.low_level_vars_flat+self.high_level_vars],
                                                                  train_size=self.train_frac, 
                                                                  test_size=1-self.train_frac, shuffle=True, random_state=1357)

    def get_X_scaler(self, X_train, out_tag='lstm_scaler'):
        """
        Derive transform on X features to give to zero mean and unit std. Derive on train set. Save for use later

        Arguments
        ---------
        X_train : Dataframe/ndarray
            training matrix on which to derive the transform
        out_tag : string
           output tag from the configuration file for the wrapper script e.g. LSTM_DNN
        """

        X_scaler = StandardScaler()
        X_scaler.fit(X_train.values)
        self.X_scaler = X_scaler
        print('saving X scaler: models/{}_X_scaler.pkl'.format(out_tag))
        dump(X_scaler, open('models/{}_X_scaler.pkl'.format(out_tag),'wb'))

    def load_X_scaler(self, out_tag='lstm_scaler'): 
        """
        Load X feature scaler, where the transform has been derived from training sample

        Arguments
        ---------
        out_tag : string
           output tag from the configuration file for the wrapper script e.g. LSTM_DNN
        """ 

        self.X_scaler = load(open('models/{}_X_scaler.pkl'.format(out_tag),'rb'))
    
    def X_scale_train_test(self, do_data=False):
        """ 
        Scale train and test X matrices to give zero mean and unit std. Annoying conversions between numpy <-> pandas but necessary for keeping feature names

        Arguments
        ---------
        do_data : bool
            whether to scale test (and train) dataset in data, to use for plotting
        """

        X_scaled_all_vars_train     = self.X_scaler.transform(self.all_vars_X_train) #returns np array so need to re-cast into pandas to get colums/variables
        X_scaled_all_vars_train     = pd.DataFrame(X_scaled_all_vars_train, columns=self.low_level_vars_flat+self.high_level_vars)
        self.X_train_low_level      = X_scaled_all_vars_train[self.low_level_vars_flat].values #will get changed to 2D arrays later
        self.X_train_high_level     = X_scaled_all_vars_train[self.high_level_vars].values

        X_scaled_all_vars_test      = self.X_scaler.transform(self.all_vars_X_test) #important to use scaler tuned on X train
        X_scaled_all_vars_test      = pd.DataFrame(X_scaled_all_vars_test, columns=self.low_level_vars_flat+self.high_level_vars)
        self.X_test_low_level       = X_scaled_all_vars_test[self.low_level_vars_flat].values #will get changed to 2D arrays later
        self.X_test_high_level      = X_scaled_all_vars_test[self.high_level_vars].values

        if do_data: #for plotting purposes
            X_scaled_data_all_vars_train      = self.X_scaler.transform(self.all_X_data_train)
            X_scaled_data_all_vars_train      = pd.DataFrame(X_scaled_data_all_vars_train, columns=self.low_level_vars_flat+self.high_level_vars)
            self.X_data_train_high_level      = X_scaled_data_all_vars_train[self.high_level_vars].values 
            self.X_data_train_low_level       = X_scaled_data_all_vars_train[self.low_level_vars_flat].values

            X_scaled_data_all_vars_test       = self.X_scaler.transform(self.all_X_data_test)
            X_scaled_data_all_vars_test       = pd.DataFrame(X_scaled_data_all_vars_test, columns=self.low_level_vars_flat+self.high_level_vars)
            self.X_data_test_high_level       = X_scaled_data_all_vars_test[self.high_level_vars].values
            self.X_data_test_low_level        = X_scaled_data_all_vars_test[self.low_level_vars_flat].values
       
    def set_low_level_2D_test_train(self, do_data=False, ignore_train=False):
        """
        Transform the 1D low-level variables into 2D variables, and overwrite corresponding class atributes

        Arguments
        ---------
        do_data : bool
            whether to scale test (and train) dataset in data, to use for plotting
        ignore_train: bool
            do not join 2D train objects. Useful if we want to keep low level as a 1D array when splitting train --> train+validate,
            since we want to do a 2D transform on 1D sequence on the rseulting train and validation sets.
        """

        if not ignore_train: self.X_train_low_level = self.join_objects(self.X_train_low_level)
        self.X_test_low_level   = self.join_objects(self.X_test_low_level)
        if do_data:
            self.X_data_train_low_level  = self.join_objects(self.X_data_train_low_level)
            self.X_data_test_low_level   = self.join_objects(self.X_data_test_low_level)

    def create_train_valid_set(self):
        """
        Partition the X and y training matrix into a train + validation set (i.e. X_train -> X_train + X_validate, and same for y and w)
        This also means turning ordinary arrays into 2D arrays, which we should be careful to keep as 1D arrays earlier

        Note that validation weights should always be the nominal MC weights
        """

        if not self.eq_train:
            X_train_high_level, X_valid_high_level, X_train_low_level, X_valid_low_level, train_w, valid_w, y_train, y_valid  = train_test_split(self.X_train_high_level, self.X_train_low_level, self.train_weights, self.y_train,
                                                                                                                                                 train_size=0.7, test_size=0.3
                                                                                                                                                 )
        else:
            X_train_high_level, X_valid_high_level, X_train_low_level, X_valid_low_level, train_w, valid_w, w_train_eq, w_valid_eq, y_train, y_valid  = train_test_split(self.X_train_high_level, self.X_train_low_level,
                                                                                                                                                                         self.train_weights, self.train_weights_eq, self.y_train,
                                                                                                                                                                         train_size=0.7, test_size=0.3
                                                                                                                                                                        )
            self.train_weights_eq = w_train_eq

        #NOTE: might need to re-equalise weights in each folds as sumW_sig != sumW_bkg anymroe!
        self.train_weights = train_w
        self.valid_weights = valid_w #validation weights should never be equalised weights!

        print 'creating validation dataset'
        self.X_train_high_level = X_train_high_level
        self.X_train_low_level  = self.join_objects(X_train_low_level)

        self.X_valid_high_level = X_valid_high_level
        self.X_valid_low_level  = self.join_objects(X_valid_low_level)
        print 'finished creating validation dataset'

        self.y_train            = y_train
        self.y_valid            = y_valid


    def join_objects(self, X_low_level):
        """
        Function take take all low level objects for each event, and transform into a matrix:
           [ [jet1-pt, jet1-eta, ...,
              jet2-pt, jet2-eta, ...,
              jet3-pt, jet3-eta, ... ]_evt1 ,

             [jet1-pt, jet1-eta, ...,
              jet2-pt, jet2-eta, ...,
              jet3-pt, jet3-eta, ...]_evt2 ,

             ...
           ]
        Note that the order of the low level inputs is important, and should be jet objects in descending pT

        Arguments
        ---------
        X_low_level: numpy ndarray
            array of X_features, with columns labelled in order: low-level vars to high-level vars

        Returns
        --------
        numpy ndarray: 2D representation of all jets in each event, for all events in X_low_level
        """

        print 'Creating 2D object vars...'
        l_to_convert = []
        for index, row in pd.DataFrame(X_low_level, columns=self.low_level_vars_flat).iterrows(): #very slow
            l_event = []
            for i_object_list in self.low_level_vars:
                l_object = []
                for i_var in i_object_list:
                    l_object.append(row[i_var])
                l_event.append(l_object)
            l_to_convert.append(l_event)
        print 'Finished creating train object vars'
        return np.array(l_to_convert, np.float32)

        
    def set_model(self, n_lstm_layers=3, n_lstm_nodes=150, n_dense_1=1, n_nodes_dense_1=300, n_dense_2=4, n_nodes_dense_2=200, dropout_rate=0.1, learning_rate=0.001, batch_norm=True, batch_momentum=0.99):
        """
        Set hyper parameters of the network, including the general structure, learning rate, and regularisation coefficients.
        Resulting model is set as a class attribute, overwriting existing model.

        Arguments
        ---------
        n_lstm_layers : int
            number of lstm layers/units 
        n_lstm_nodes : int
            number of nodes in each lstm layer/unit
        n_dense_1 : int
            number of dense fully connected layers
        n_dense_nodes_1 : int
            number of nodes in each dense fully connected layer
        n_dense_2 : int
            number of regular fully connected layers
        n_dense_nodes_2 : int
            number of nodes in each regular fully connected layer
        dropout_rate : float
            fraction of weights to be dropped during training, to regularise the network
        learning_rate: float
            learning rate for gradient-descent based loss minimisation
        batch_norm: bool
             option to normalise each batch before training
        batch_momentum : float
             momentum for the gradient descent, evaluated on a given batch
        """

        input_objects = keras.layers.Input(shape=(len(self.low_level_vars), len(self.low_level_vars[0])), name='input_objects') 
        input_global  = keras.layers.Input(shape=(len(self.high_level_vars),), name='input_global')
        lstm = input_objects
        for i_layer in range(n_lstm_layers):
            lstm = keras.layers.LSTM(n_lstm_nodes, activation='tanh', return_sequences=(i_layer!=(n_lstm_layers-1)), name='lstm_{}'.format(i_layer))(lstm)

        #inputs to dense layers are output of lstm and global-event variables. Also batch norm the FC layers
        dense = keras.layers.concatenate([input_global, lstm])
        for i in range(n_dense_1):
            dense = keras.layers.Dense(n_nodes_dense_1, activation='relu', kernel_initializer='lecun_uniform', name = 'dense1_%d' % i)(dense)
            if batch_norm:
                dense = keras.layers.BatchNormalization(name = 'dense_batch_norm1_%d' % i)(dense)
        dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout1_%d' % i)(dense)

        for i in range(n_dense_2):
            dense = keras.layers.Dense(n_nodes_dense_2, activation='relu', kernel_initializer='lecun_uniform', name = 'dense2_%d' % i)(dense)
            #add droput and norm if not on last layer
            if batch_norm and i < (n_dense_2 - 1):
                dense = keras.layers.BatchNormalization(name = 'dense_batch_norm2_%d' % i)(dense) 
            if i < (n_dense_2 - 1):
                dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout2_%d' % i)(dense)

        output = keras.layers.Dense(1, activation = 'sigmoid', name = 'output')(dense)
        optimiser = keras.optimizers.Nadam(lr = learning_rate)

        model = keras.models.Model(inputs = [input_global, input_objects], outputs = [output])
        model.compile(optimizer = optimiser, loss = 'binary_crossentropy')
        self.model = model

    def train_w_batch_boost(self, out_tag='my_lstm', save=True, auc_threshold=0.01):
        """
        Alternative method of tranining, where the batch size is increased during training, 
        if the improvement in (1-AUC) is above some threshold.
        Terminate the training early if no improvement is seen after max batch size update

        Arguments
        --------
        out_tag: string
            output tag used as part of the model name, when saving
        save: bool
            option to save the best model
        auc_threshold: float
            minimum improvement in (1-AUC) to warrant not updating the batch size. 
        """

        self.create_train_valid_set()

        #paramaters that control batch size
        best_auc           = 0.5
        current_batch_size = 1024
        max_batch_size     = 50000

        #keep track of epochs for plotting loss vs epoch, and for getting best model
        epoch_counter      = 0 
        best_epoch         = 1 

        keep_training = True

        while keep_training:
            epoch_counter += 1
            print('beginning training iteration for epoch {}'.format(epoch_counter))
            self.train_network(epochs=1, batch_size=current_batch_size)

            self.save_model(epoch_counter, out_tag)
            val_roc = self.compute_roc(batch_size=current_batch_size, valid_set=True)  #FIXME: what is the best BS here? final BS from batch boost... initial BS? current BS??

            #get average of validation rocs and clear list entries 
            improvement  = ((1-best_auc) - (1-val_roc)) / (1-best_auc)

            #FIXME: if the validation roc does not improve after n bad "epochs", then update the batch size accordingly. Rest bad epochs to zero each time the batch size increases, if it does

            #do checks to see if batch size needs to change etc
            if improvement > auc_threshold:
                print('Improvement in (1-AUC) of {:.4f} percent. Keeping batch size at {}'.format(improvement*100, current_batch_size))
                best_auc = val_roc
                best_epoch = epoch_counter
            elif current_batch_size*4 < max_batch_size:
                print('Improvement in (1-AUC) of only {:.4f} percent. Increasing batch size to {}'.format(improvement*100, current_batch_size*4))
                current_batch_size *= 4
                if val_roc > best_auc: 
                    best_auc = val_roc
                    best_epoch = epoch_counter
            elif current_batch_size < max_batch_size: 
                print('Improvement in (1-AUC) of only {:.4f} percent. Increasing to max batch size of {}'.format(improvement*100, max_batch_size))
                current_batch_size = max_batch_size
                if val_roc > best_auc: 
                    best_auc = val_roc
                    best_epoch = epoch_counter
            elif improvement > 0:
                print('Improvement in (1-AUC) of only {:.4f} percent. Cannot increase batch further'.format(improvement*100))
                best_auc = val_roc
                best_epoch = epoch_counter
            else: 
                print('AUC did not improve and batch size cannot be increased further. Stopping training...')
                keep_training = False

            if epoch_counter > self.max_epochs:
                print('At the maximum number of training epochs ({}). Stopping training...'.format(self.max_epochs))
                keep_training = False
                best_epoch = self.max_epochs
            
        print 'best epoch was: {}'.format(best_epoch)
        print 'best validation auc was: {}'.format(best_auc)
        self.val_roc = best_auc
      

        #delete all models that aren't from the best training. Re-load best model for predicting on test set 
        for epoch in range(1,epoch_counter+1):
            if epoch is not best_epoch:
                os.system('rm {}/models/{}_model_epoch_{}.hdf5'.format(os.getcwd(), out_tag, epoch))
                os.system('rm {}/models/{}_model_architecture_epoch_{}.json'.format(os.getcwd(), out_tag, epoch))
        os.system('mv {0}/models/{1}_model_epoch_{2}.hdf5 {0}/models/{1}_model.hdf5'.format(os.getcwd(), out_tag, best_epoch))
        os.system('mv {0}/models/{1}_model_architecture_epoch_{2}.json {0}/models/{1}_model_architecture.json'.format(os.getcwd(), out_tag, best_epoch))

        #reset model state and load in best weights
        with open('{}/models/{}_model_architecture.json'.format(os.getcwd(), out_tag), 'r') as model_json:
            best_model_architecture = model_json.read()
        self.model = keras.models.model_from_json(best_model_architecture)
        self.model.load_weights('{}/models/{}_model.hdf5'.format(os.getcwd(), out_tag))

        if not save:
            os.system('rm {}/models/{}_model_architecture.json'.format(os.getcwd(), out_tag))
            os.system('rm {}/models/{}_model.hdf5'.format(os.getcwd(), out_tag))
        
    def train_network(self, batch_size, epochs):
        """
        Train the network over a given number of epochs
        Arguments
        ---------
        batch_size: int
            number of training samples to compute the gradient on during training
        epochs: int
            number of full passes oevr the training sample
        """

        if self.eq_train: self.model.fit([self.X_train_high_level, self.X_train_low_level], self.y_train, epochs=epochs, batch_size=batch_size, sample_weight=self.train_weights_eq)       
        else: self.model.fit([self.X_train_high_level, self.X_train_low_level], self.y_train, epochs=epochs, batch_size=batch_size, sample_weight=self.train_weights)       
    
    def save_model(self, epoch=None, out_tag='my_lstm'):
        """
        Save the deep learning model, training up to a given epoch
        
        Arguments:
        ---------
        epoch: int
            the epoch to which to model is trained up to    
        out_tag: string
            output tag used as part of the model name, when saving
        """

        Utils.check_dir('./models/')
        if epoch is not None:
            self.model.save_weights('{}/models/{}_model_epoch_{}.hdf5'.format(os.getcwd(), out_tag, epoch))
            with open("{}/models/{}_model_architecture_epoch_{}.json".format(os.getcwd(), out_tag, epoch), "w") as f_out:
                f_out.write(self.model.to_json())
        else: 
            self.model.save_weights('{}/models/{}_model.hdf5'.format(os.getcwd(), out_tag))
            with open("{}/models/{}_model_architecture.json".format(os.getcwd(), out_tag), "w") as f_out:
                f_out.write(self.model.to_json())



    def compare_rocs(self, roc_file, hp_string):
        """
        Compare the AUC for the current model, to the current best AUC saved in a .txt file 
        Arguments
        ---------
        roc_file: string
            path for the file holding the current best AUC (as the final line)
        hp_string: string
            string contraining each hyper_paramter for the network, with the following syntax: 'hp_1_name:hp_1_value, hp_2_name:hp_2_value, ...'
        """

        hp_roc = roc_file.readlines()
        val_auc = self.val_roc
        print 'validation roc is: {}'.format(val_auc)
        if len(hp_roc)==0: 
            roc_file.write('{};{:.4f}'.format(hp_string, val_auc))
        elif float(hp_roc[-1].split(';')[-1]) < val_auc:
            roc_file.write('\n')
            roc_file.write('{};{:.4f}'.format(hp_string, val_auc))


    def batch_gs_cv(self):
        """
        Submit sets of hyperparameters permutations (based on attribute hp_grid_rnge) to the IC batch.
        Take care to separate training weights, which may be modified w.r.t nominal weights, 
        and the weights used when evaluating on the validation set which should be the nominal weights
        """
        #get all possible HP sets from permutations of the above dict
        hp_perms = self.get_hp_perms()
        #submit job to the batch for the given HP range:
        for hp_string in hp_perms:
            Utils.sub_lstm_hp_script(self.eq_train, self.batch_boost, hp_string)

    def get_hp_perms(self):
        """
        Get all possible combinations of the hyper-parameters specified in self.hp_grid_range
        
        Returns
        -------        
        final_hps: list of all possible hyper parameter combinations in format 'hp_1_name:hp_1_value, hp_2_name:hp_2_value, ...'
        """

        from itertools import product
        hp_perms  = [perm for perm in apply(product, self.hp_grid_rnge.values())]
        final_hps = []
        counter   = 0
        for hp_perm in hp_perms:
            l_entry = ''
            for hp_name, hp_value in zip(self.hp_grid_rnge.keys(), hp_perm):
                l_entry+='{}:{},'.format(hp_name,hp_value)
                counter+=1
                if (counter % len(self.hp_grid_rnge.keys())) == 0: final_hps.append(l_entry[:-1])
        return final_hps

    def set_hyper_parameters(self, hp_string):
        """
        Set the hyperparameters for the network, given some inut string of parameters
        Arguments:
        ---------
        hp_string: string
            string contraining each hyper_paramter for the network, with the following syntax: 'hp_1_name:hp_1_value, hp_2_name:hp_2_value, ...'
        """

        hp_dict = {}
        for params in hp_string.split(','):
            hp_name = params.split(':')[0]
            hp_value =params.split(':')[1]
            try: hp_value = int(hp_value)
            except ValueError: hp_value = float(hp_value)
            hp_dict[hp_name] = hp_value
            self.set_model(**hp_dict)

    def compute_roc(self, batch_size=64, valid_set=False):
        """
        Compute the area under the associated ROC curve, with usual mc weights
        Arguments
        ---------
        batch_size: int
            necessary to evaluate the network. Has no impact on the output score.
        valid_set: bool
            compute the roc score on validation set instead of than the test set
        Returns
        -------
        roc_test : float
            return the score on the test set (or validation set if performing any model selection)
        """

        self.y_pred_train = self.model.predict([self.X_train_high_level, self.X_train_low_level], batch_size=batch_size).flatten()
        roc_train = roc_auc_score(self.y_train, self.y_pred_train, sample_weight=self.train_weights)
        print 'ROC train score: {}'.format(roc_train)

        if valid_set:
            self.y_pred_valid = self.model.predict([self.X_valid_high_level, self.X_valid_low_level], batch_size=batch_size).flatten()
            roc_test  = roc_auc_score(self.y_valid, self.y_pred_valid, sample_weight=self.valid_weights)
            print 'ROC valid score: {}'.format(roc_test)
        else:
            self.y_pred_test = self.model.predict([self.X_test_high_level, self.X_test_low_level], batch_size=batch_size).flatten()
            roc_test  = roc_auc_score(self.y_test, self.y_pred_test, sample_weight=self.test_weights)
            print 'ROC test score: {}'.format(roc_test)

        return roc_test

    def plot_roc(self, out_tag):
        """
        Plot the roc curve for the classifier, using method from Plotter() class
        Arguments
        ---------
        out_tag: string
            output tag used as part of the image name, when saving
        """
        roc_fig = self.plotter.plot_roc(self.y_train, self.y_pred_train, self.train_weights, 
                                        self.y_test, self.y_pred_test, self.test_weights, out_tag=out_tag
                                       )

        Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(), out_tag))
        roc_fig.savefig('{0}/plotting/plots/{1}/{1}_ROC_curve.pdf'.format(os.getcwd(),out_tag))
        print('saving: {0}/plotting/plots/{1}/{1}_ROC_curve.pdf'.format(os.getcwd(),out_tag))
        plt.close()

        #for MVA ROC comparisons later on
        np.savez("{}/models/{}_ROC_comp_arrays".format(os.getcwd(), out_tag),  self.y_pred_test, self.y_pred_test, self.test_weights)

    def plot_output_score(self, out_tag, batch_size=64, ratio_plot=False, norm_to_data=False):
        """
        Plot the output score for the classifier, for signal, background, and data
        Arguments
        ---------
        out_tag: string
            output tag used as part of the image name, when saving
        batch_size: int
            necessary to evaluate the network. Has no impact on the output score.
        ratio_plot: bool
            whether to plot the ratio between simulated background and data
        norm_to_data: bool
            whether to normalise the integral of the simulated background, to the integral in data
        """

        output_score_fig = self.plotter.plot_output_score(self.y_test, self.y_pred_test, self.test_weights, self.proc_arr_test,
                                                          self.model.predict([self.X_data_test_high_level, self.X_data_test_low_level], batch_size=batch_size).flatten(),
                                                          MVA='DNN', ratio_plot=ratio_plot, norm_to_data=norm_to_data)

        Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(),out_tag))
        output_score_fig.savefig('{0}/plotting/plots/{1}/{1}_output_score.pdf'.format(os.getcwd(), out_tag))
        print('saving: {0}/plotting/plots/{1}/{1}_output_score.pdf'.format(os.getcwd(), out_tag))
        plt.close()
    
