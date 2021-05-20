import pandas as pd
import torch
import numpy as np
import argparse
import os
import json
from collections import defaultdict as ddict
from pathlib import Path
from GBDT import *
from Base import *
import networkx as nx
import dgl
import fire
import pdb
from modules import LaplacianKernel

class RunModel:


    def run_one_model(self, data_name, task, seed):
        
        print("dataset/seed:", data_name, seed)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load data 
        input_folder = Path(__file__).parent.parent / 'datasets' / data_name
        self.X = pd.read_csv(f'{input_folder}/X.csv')
        self.y = pd.read_csv(f'{input_folder}/y.csv')


        categorical_columns = []
        if os.path.exists(f'{input_folder}/cat_features.txt'):
            with open(f'{input_folder}/cat_features.txt') as f:
                for line in f:
                    if line.strip():
                        categorical_columns.append(line.strip())

        self.cat_features = None
        if categorical_columns:
            columns = self.X.columns
            self.cat_features = np.where(columns.isin(categorical_columns))[0]

            for col in list(columns[self.cat_features]):
                self.X[col] = self.X[col].astype(str)



        # load mask
        if os.path.exists(f'{input_folder}/masks.json'):
            with open(f'{input_folder}/masks.json') as f:
                self.masks = json.load(f)


        ## split train_mask, test_mask and val_mask
        train_mask, val_mask, test_mask = self.masks[seed]['train'], self.masks[seed]['val'], self.masks[seed]['test']



        # data preprocessing
        encoded_X = self.X.copy()
        if self.cat_features is None:
            self.cat_features = []
        if len(self.cat_features):
            encoded_X = encode_cat_features(encoded_X, self.y, self.cat_features, train_mask, val_mask, test_mask)
        encoded_X = normalize_features(encoded_X, train_mask, val_mask, test_mask)
        encoded_X = replace_na(encoded_X, train_mask)
        encoded_X = pandas_to_torch(encoded_X)
        target = torch.from_numpy(self.y.to_numpy(copy=True)).float()




        ## load graph structure
        networkx_graph = nx.read_graphml(f'{input_folder}/graph.graphml')
        networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})
        graph = dgl.from_networkx(networkx_graph)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph = graph.to(self.device)
        use_kernel=True
        kernel_type="linear"
        if use_kernel:
            lapKernel=LaplacianKernel(graph)
            if kernel_type=="diffusion":
                kernel=lapKernel.diffusion_kernel(sigma=.2)
            elif kernel_type=="linear":
                kernel=lapKernel.linear_kernel(sigma=10**-1)
            elif kernel_type=="bandlimited":
                kernel=lapKernel.bandlimited_kernel(bandwidth=10)

        else:
            kernel=None
        params = {}
        print('Start training...')
        gbt = GBDT(task, graph, train_mask, test_mask, val_mask,kernel=kernel)
        metrics = gbt.train(params,
                encoded_X,
                target,
                cat_features=None,
                num_boost_round=1000,
                early_stopping_rounds=15)

        return metrics


    def run(self, max_seeds: int=5):

        parser = argparse.ArgumentParser(description='Train a GBDT with graph information',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--datasets', '-s', type=str)
        parser.add_argument('--task', '-t', type=str, choices=["regression", "classification"])
        args = parser.parse_args()

        aggregated = dict()
        seed_results = []
        for seed in range(max_seeds):
            seed_results.append(self.run_one_model(args.datasets, args.task, str(seed)))
        aggregated[args.datasets] = (np.mean(seed_results), np.std(seed_results))

     

        save_path = f'results/{args.datasets}' 
        os.makedirs(save_path, exist_ok=True)
        with open(f'{save_path}/seed_results.json', 'w+') as f:
            json.dump(str(seed_results), f) 
        with open(f'{save_path}/aggregated.json', 'w+') as f:
            json.dump(str(aggregated), f)
        exit()

      

if __name__ == '__main__':
    fire.Fire(RunModel().run)