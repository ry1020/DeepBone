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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


from combine_features import compute_and_combine_features, load_and_combine_features
from utils import worker_init_fn
from model_wDeepFeature import generate_model, make_data_parallel
from data import get_inference_data
from opts import parse_opts


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

def main_worker(opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    model = generate_model(opt)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)

    model = make_data_parallel(model, opt.distributed, opt.device)
    parameters = model.parameters()

    # if opt.is_master_node:
    #     print(model)

    rf_pipe_path = opt.result_path / 'rf_pipe.sav'

    if not opt.no_train:
        print('training')
        print('Start Time = {}'.format(datetime.now().time()))
                            
        train_loader = get_inference_utils(opt, 'train')

        # features_array_train, targets_train = compute_and_combine_features(train_loader, model, opt.data_path, 'train')
        features_array_train, targets_train = load_and_combine_features(opt.data_path, 'train')
        
        rf_pipe = Pipeline([('scl', StandardScaler()),
                            ('reg',RandomForestRegressor(n_estimators=100, min_samples_split=10, random_state=0, n_jobs=-1))])
                
        rf_pipe.fit(features_array_train, targets_train)
        
        importances = rf_pipe['reg'].feature_importances_
        importances_result_path = os.path.join(opt.result_path, 'importance.csv')
        pd.DataFrame({'importances':importances}).to_csv(importances_result_path)

        pickle.dump(rf_pipe, open(rf_pipe_path, 'wb'))

        y_pred_train = rf_pipe.predict(features_array_train)
        rms = mean_squared_error(targets_train, y_pred_train, squared=False)
        r_square = np.corrcoef(y_pred_train, targets_train)[0,1]**2


        print('Training RMSE: {}\t''R_square:{}'.format(rms, r_square))
        print('ENd Time = {}'.format(datetime.now().time()))

    if not opt.no_val:
        print('validation')
        print('Start Time = {}'.format(datetime.now().time()))
        val_loader = get_inference_utils(opt, 'val')
                        
        # features_array_val, targets_val = compute_and_combine_features(val_loader, model, opt.data_path, 'val')
        features_array_val, targets_val = load_and_combine_features(opt.data_path, 'val')

        rf_pipe = pickle.load(open(rf_pipe_path, 'rb'))

        y_pred_val = rf_pipe.predict(features_array_val)
        rms = mean_squared_error(targets_val, y_pred_val, squared=False)
        r_square = np.corrcoef(y_pred_val, targets_val)[0,1]**2

        print('Validation RMSE: {}\t''R_square:{}'.format(rms, r_square))
        print('ENd Time = {}'.format(datetime.now().time()))

    if opt.inference:
        opt.inference_subset = 'train'
        print('inference-'+opt.inference_subset)
        print('Start Time = {}'.format(datetime.now().time()))

        inference_loader = get_inference_utils(opt, opt.inference_subset)

        # features_array_inference, targets_inference = compute_and_combine_features(inference_loader, model, opt.data_path, opt.inference_subset)
        features_array_inference, targets_inference = load_and_combine_features(opt.data_path, opt.inference_subset)
        rf_pipe = pickle.load(open(rf_pipe_path, 'rb'))
                
        y_pred_inference = rf_pipe.predict(features_array_inference)
        rms = mean_squared_error(targets_inference, y_pred_inference, squared=False)
        r_square = np.corrcoef(y_pred_inference, targets_inference)[0,1]**2

        print('inference-'+opt.inference_subset+' RMSE: {}\t''R_square:{}'.format(rms, r_square))

        inference_result_path = os.path.join(opt.result_path, '_'.join((opt.inference_subset, 'inference.csv')))
        pd.DataFrame({'target':targets_inference,'output':y_pred_inference}).to_csv(inference_result_path)
        pd.DataFrame({'rms':[rms], 'r_square':[r_square]}).to_csv(inference_result_path, mode = 'a')

        print('ENd Time = {}'.format(datetime.now().time()))

        opt.inference_subset = 'val'
        print('inference-'+opt.inference_subset)
        print('Start Time = {}'.format(datetime.now().time()))

        inference_loader = get_inference_utils(opt, opt.inference_subset)

        # features_array_inference, targets_inference = compute_and_combine_features(inference_loader, model, opt.data_path, opt.inference_subset)
        features_array_inference, targets_inference = load_and_combine_features(opt.data_path, opt.inference_subset)
        rf_pipe = pickle.load(open(rf_pipe_path, 'rb'))
                
        y_pred_inference = rf_pipe.predict(features_array_inference)
        rms = mean_squared_error(targets_inference, y_pred_inference, squared=False)
        r_square = np.corrcoef(y_pred_inference, targets_inference)[0,1]**2

        print('inference-'+opt.inference_subset+' RMSE: {}\t''R_square:{}'.format(rms, r_square))

        inference_result_path = os.path.join(opt.result_path, '_'.join((opt.inference_subset, 'inference.csv')))
        pd.DataFrame({'target':targets_inference,'output':y_pred_inference}).to_csv(inference_result_path)
        pd.DataFrame({'rms':[rms], 'r_square':[r_square]}).to_csv(inference_result_path, mode = 'a')

        print('ENd Time = {}'.format(datetime.now().time()))

        opt.inference_subset = 'test'
        print('inference-'+opt.inference_subset)
        print('Start Time = {}'.format(datetime.now().time()))

        # inference_loader = get_inference_utils(opt, opt.inference_subset)

        # features_array_inference, targets_inference = compute_and_combine_features(inference_loader, model, opt.data_path, opt.inference_subset)
        features_array_inference, targets_inference = load_and_combine_features(opt.data_path, opt.inference_subset)
        rf_pipe = pickle.load(open(rf_pipe_path, 'rb'))
                
        y_pred_inference = rf_pipe.predict(features_array_inference)
        rms = mean_squared_error(targets_inference, y_pred_inference, squared=False)
        r_square = np.corrcoef(y_pred_inference, targets_inference)[0,1]**2

        print('inference-'+opt.inference_subset+' RMSE: {}\t''R_square:{}'.format(rms, r_square))

        inference_result_path = os.path.join(opt.result_path, '_'.join((opt.inference_subset, 'inference.csv')))
        pd.DataFrame({'target':targets_inference,'output':y_pred_inference}).to_csv(inference_result_path)
        pd.DataFrame({'rms':[rms], 'r_square':[r_square]}).to_csv(inference_result_path, mode = 'a')

        print('ENd Time = {}'.format(datetime.now().time()))

if __name__ == '__main__':

    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True

    main_worker(opt)
