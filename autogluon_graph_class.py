import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
import os
import json
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
import pdb
import networkx as nx
import dgl
from autogluon.tabular import TabularDataset, TabularPredictor
from modules import UnfoldindAndAttention
from sklearn import preprocessing


propagation = UnfoldindAndAttention(lam=20, prop_step = 5)
propagation_error = UnfoldindAndAttention(lam=5, prop_step = 5)
propagation_label = UnfoldindAndAttention(lam=5, prop_step = 5)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def normalize_features(X, train_mask, val_mask, test_mask):
    min_max_scaler = preprocessing.MinMaxScaler()
    A = X.to_numpy(copy=True)
    A[train_mask] = min_max_scaler.fit_transform(A[train_mask])
    A[val_mask + test_mask] = min_max_scaler.transform(A[val_mask + test_mask])
    return pd.DataFrame(A, columns=X.columns).astype(float)

def replace_na(X, train_mask):
    if X.isna().any().any():
        return X.fillna(X.iloc[train_mask].min() - 1)
    return X

def encode_cat_features(X, y, cat_features, train_mask, val_mask, test_mask):
    from category_encoders import CatBoostEncoder
    enc = CatBoostEncoder()
    A = X.to_numpy(copy=True)
    b = y.to_numpy(copy=True)
    A[np.ix_(train_mask, cat_features)] = enc.fit_transform(A[np.ix_(train_mask, cat_features)], b[train_mask])
    A[np.ix_(val_mask + test_mask, cat_features)] = enc.transform(A[np.ix_(val_mask + test_mask, cat_features)])
    A = A.astype(float)
    return pd.DataFrame(A, columns=X.columns)

def pandas_to_torch(args):
    return torch.from_numpy(args.to_numpy(copy=True)).float().squeeze().to(device) 




data_name = 'slap'
seed = '0'
save_path = f'results_cat_X/{data_name}/{seed}'  # specifies folder to store trained models


print('Loading data...')
dataset_dir = Path(__file__).parent.parent / 'datasets'
input_folder = dataset_dir / data_name

X = pd.read_csv(f'{input_folder}/X.csv')
X_orig = X

y = pd.read_csv(f'{input_folder}/y.csv')

categorical_columns = []
if os.path.exists(f'{input_folder}/cat_features.txt'):
    with open(f'{input_folder}/cat_features.txt') as f:
        for line in f:
            if line.strip():
                categorical_columns.append(line.strip())

cat_features = None
if categorical_columns:
    columns = X.columns
    cat_features = np.where(columns.isin(categorical_columns))[0]

    for col in list(columns[cat_features]):
        X[col] = X[col].astype(str)




# load mask
if os.path.exists(f'{input_folder}/masks.json'):
    with open(f'{input_folder}/masks.json') as f:
        masks = json.load(f)



## load graph structure
networkx_graph = nx.read_graphml(f'{input_folder}/graph.graphml')
networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})
graph = dgl.from_networkx(networkx_graph)
graph = dgl.remove_self_loop(graph)
graph = dgl.add_self_loop(graph)
graph = graph.to(device)



## split train_mask, test_mask and val_mask
train_mask, val_mask, test_mask = masks[seed]['train'], masks[seed]['val'], masks[seed]['test']



encoded_X = X.copy()
if cat_features  is None:
    cat_features = []
if len(cat_features):
    encoded_X = encode_cat_features(encoded_X, y, cat_features, train_mask, val_mask, test_mask)
encoded_X = normalize_features(encoded_X, train_mask, val_mask, test_mask)
encoded_X = replace_na(encoded_X, train_mask)
encoded_X = pandas_to_torch(encoded_X)


## propagate the feature
assert len(encoded_X.size()) == 2
corrected_X = propagation.forward(graph, encoded_X)  

## cat the propagated features and orignal features
X = torch.cat((encoded_X, corrected_X), 1).cpu().numpy()
Column_index = ['Column_{}'.format(i) for i in range(X.shape[1])]
X_cat = pd.DataFrame(X, columns=Column_index)
dataset = pd.concat([X_cat, y], axis=1)
# dataset = pd.concat([X_orig, y], axis=1)



# pdb.set_trace()

# loading existing model
# predictor = TabularPredictor.load(save_path)

predictor = TabularPredictor(label='class', path=save_path).fit(dataset.iloc[train_mask], presets='best_quality', time_limit=200)
performance = predictor.evaluate(dataset.iloc[test_mask])




# error smoothing
y_pred = predictor.predict_proba(X_cat)
y_pred = torch.from_numpy(y_pred.to_numpy()).float().to(device)
y_true = torch.from_numpy(y.to_numpy()).long().to(device)
# y_true[y_true == -1] = 0  ## slap and dblp has label -1, avoid using label -1 for train, val and test
y_one_hot = torch.zeros_like(y_pred).scatter_(1, y_true, 1)


error_smooth = torch.zeros_like(y_pred)
error_smooth[train_mask] = y_one_hot[train_mask] - y_pred[train_mask]
assert len(error_smooth.size()) == 2
error_smooth = propagation_error.forward(graph, error_smooth, train_mask, error=True)  
y_pred = y_pred + error_smooth



# label smoothing
label_smooth = y_pred.clone()
label_smooth[train_mask] = y_one_hot[train_mask]
assert len(label_smooth.size()) == 2
label_smooth = propagation_label.forward(graph, label_smooth, train_mask, label=True)  
    
  
# accuracy
pred = label_smooth[test_mask]
target_test = y_true[test_mask].squeeze()
print(torch.Tensor([(target_test == pred.max(1)[1]).sum().item()/target_test.shape[0]]))














