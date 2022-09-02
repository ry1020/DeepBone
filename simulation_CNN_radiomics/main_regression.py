from __future__ import print_function
import csv
from datetime import datetime
import os
from pathlib import Path
import json
import pickle
import random
import time
import numpy as np
import pandas as pd

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import SVR, LinearSVR

from combine_features import compute_and_combine_features, load_and_combine_features
from utils import worker_init_fn
from model_wDeepFeature import generate_model, make_data_parallel
from data import get_inference_data
from opts_regression import parse_opts


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()
    
    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.n_input_channels = 1

    print(opt)
    with (opt.result_path / 'opts.json').open('w') as opt_file:
        json.dump(vars(opt), opt_file, default=json_serial)

    return opt

def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model

def get_inference_utils(opt, subset):
    inference_data = get_inference_data(opt.data_path, opt.no_sim, opt.noise_scales, opt.resolution_scales,
                                    opt.voxel_size_simulated, opt.manual_seed, subset)

    inference_loader = DataLoader(inference_data,
                                batch_size=opt.inference_batch_size,
                                shuffle=False,
                                num_workers=opt.n_threads,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)
    return inference_loader

def evaluate(model, features_array, targets):
    y_pred = model.predict(features_array)
    rms = mean_squared_error(targets, y_pred, squared=False)
    r_square = np.corrcoef(y_pred, targets)[0,1]**2
    return rms, r_square


def main_worker(opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    best_estimator_path = os.path.join(opt.result_path, 'best_estimator.sav')

    if not opt.no_compute_features:

        opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
        if not opt.no_cuda:
            cudnn.benchmark = True

        opt.is_master_node = not opt.distributed or opt.dist_rank == 0

        model = generate_model(opt)
        if opt.resume_path is not None:
            model = resume_model(opt.resume_path, opt.arch, model)

        model = make_data_parallel(model, opt.distributed, opt.device)
        parameters = model.parameters()

        # if opt.is_master_node:
        #     print(model)

        train_loader = get_inference_utils(opt, 'trainVal')
        features_array_train = compute_and_combine_features(train_loader, model, opt.data_path, 'trainVal')
        features_file = os.path.join(opt.features_path, ''.join(['features_sim_', 'trainVal' ,'.csv']))
        pd.DataFrame(features_array_train).to_csv(features_file)

        test_loader = get_inference_utils(opt, 'test')
        features_array_test = compute_and_combine_features(test_loader, model, opt.data_path, 'test')
        features_file = os.path.join(opt.features_path, ''.join(['features_sim_','test','.csv']))
        pd.DataFrame(features_array_test).to_csv(features_file)

        # L1_loader = get_inference_utils(opt, 'L1')
        # features_array_L1 = compute_and_combine_features(L1_loader, model, opt.data_path, 'L1')
        # features_file = os.path.join(get_features_path(opt.data_path), ''.join(['features_sim_','L1','.csv']))
        # pd.DataFrame(features_array_L1).to_csv(features_file)

    if not opt.no_train:
        print('training-'+opt.train_subset)
        print('Start Time = {}'.format(datetime.now().time()))

        reg_pipe = Pipeline([('scl', StandardScaler()),
                    ('reg',BayesianRidge())]) # ('reg',RandomForestRegressor(random_state=0, n_jobs=-1))])

        scoring = {"RMSE": "neg_root_mean_squared_error", "R2": "r2"}


        features_array_train, targets_train, group= load_and_combine_features(opt.data_path, opt.train_subset, opt.features_path)
        gss = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        # for train, test in gss.split(features_array_train, targets_train, groups = group):        
        #     print("%s %s" % (train, test))

        if opt.is_random_search:
            print('Random search')
            # Create the random grid
            random_grid ={'reg__n_estimators': [200, 600, 1000, 1400, 1600],
                        'reg__max_features': [14, 16],
                        'reg__max_depth': [5, 10, 15],
                        'reg__min_samples_split': [8, 12],
                        'reg__min_samples_leaf': [8, 10, 15],
                        'reg__bootstrap': [True]}

            model_search = RandomizedSearchCV(estimator=reg_pipe, param_distributions=random_grid, n_iter = 5000, scoring=scoring, cv=gss, refit= 'RMSE', verbose = 3, random_state = 20, n_jobs=-1)

        else:
            print('Grid search')
            # Create the parameter grid based on the results of random search 
            param_grid = {'reg__alpha_1': [1e-5, 1e-4],
                        'reg__alpha_2': [1e-7, 1e-6],
                        'reg__tol': [1e-4, 1e-5],
                        'reg__n_iter': [int(x) for x in np.linspace(start = 1000, stop = 2000, num = 5)],
                        'reg__lambda_1': [1e-7, 1e-6],
                        'reg__lambda_2': [1e-7, 1e-6]}
            model_search = GridSearchCV(estimator=reg_pipe, param_grid=param_grid, scoring=scoring, cv=gss, refit= 'RMSE', verbose = 1,  n_jobs=-1)


        model_search.fit(features_array_train, targets_train, groups= group)
        print("Best parameters set found on development set:")
        print(model_search.best_params_)
        
        best_estimator = model_search.best_estimator_
        rms, r_square = evaluate(best_estimator, features_array_train, targets_train)
        print('Training RMSE: {}\t''R_square:{}'.format(rms, r_square))

        pickle.dump(best_estimator, open(best_estimator_path, 'wb'))

        importances = best_estimator.named_steps.reg.feature_importances_
        importances_result_path = os.path.join(opt.result_path, 'importance.csv')
        pd.DataFrame({'importances':importances}).to_csv(importances_result_path)

        print('End Time = {}'.format(datetime.now().time()))


    if opt.inference:
        print('inference-'+opt.inference_subset)
        print('Start Time = {}'.format(datetime.now().time()))


        features_array_inference, targets_inference = load_and_combine_features(opt.data_path, opt.inference_subset, opt.features_path)
        
        best_estimator = pickle.load(open(best_estimator_path, 'rb'))
                
        rms, r_square = evaluate(best_estimator, features_array_inference, targets_inference)
        print('Inference-'+opt.inference_subset+' RMSE: {}\t''R_square:{}'.format(rms, r_square))

        y_pred_inference = best_estimator.predict(features_array_inference)
        inference_result_path = os.path.join(opt.result_path, '_'.join((opt.inference_subset, 'inference.csv')))
        pd.DataFrame({'target':targets_inference,'output':y_pred_inference}).to_csv(inference_result_path)
        pd.DataFrame({'rms':[rms], 'r_square':[r_square]}).to_csv(inference_result_path, mode = 'a')

        print('End Time = {}'.format(datetime.now().time()))

if __name__ == '__main__':

    opt = get_opt()

    main_worker(opt)
