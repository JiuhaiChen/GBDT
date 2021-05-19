import sys
import time
LARGE_NUMBER = sys.maxsize
import numpy as np
import torch
import pdb
import torch.nn.functional as F
from modules import UnfoldindAndAttention
import pandas as pd
from Base import *
from collections import defaultdict as ddict
from catboost import Pool, CatBoostClassifier, CatBoostRegressor, sum_models



class GBDT(object):
    def __init__(self, task, graph, train_mask, test_mask, val_mask):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.task = task
        self.graph = graph
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.val_mask = val_mask

        self.params = {'learning_rate': 1.0}
        self.best_iteration = None
        self.iter_per_epoch = 10
        self.depth = 6
        self.gbdt_lr = 0.1
        self.propagation_X = UnfoldindAndAttention(lam=20.0, prop_step=5)
        self.propagation_y = UnfoldindAndAttention(lam=2.0, prop_step=5)
        
        

        

    def _calc_data_scores(self, X, epoch):


        if epoch == 0:
            scores = torch.zeros(self.num_samples, self.out_dim)
        else:       
            scores = self.gbdt_model.predict(X)
            scores = torch.from_numpy(scores).float().view(self.num_samples, self.out_dim)

        return scores.to(self.device)


    def _calc_gradient(self, scores, labels):
      
        
        scores.requires_grad_()

        with torch.enable_grad():
            assert len(scores.size()) == 2
            scores_correct = self.propagation_y.forward(self.graph, scores)

            if self.task == 'regression':
                loss = F.mse_loss(scores_correct[self.train_mask], labels[self.train_mask], reduction='sum')
            elif self.task == 'classification':
                loss = F.cross_entropy(scores_correct[self.train_mask], labels[self.train_mask].long(), reduction='sum')
                

        grad = torch.autograd.grad(loss, scores, only_inputs=True)[0]
        grad = grad.detach()


        return  - grad.cpu().numpy() 

      
                    
    def _calc_loss(self, X, y, metrics):

        pred = self.gbdt_model.predict(X)
        pred = torch.from_numpy(pred).float().view(self.num_samples, self.out_dim).to(self.device)

        assert len(pred.size()) == 2
        scores_correct = self.propagation_y.forward(self.graph, pred)


        train_results = self.evaluate_model(scores_correct, y, self.train_mask)
        test_results = self.evaluate_model(scores_correct, y, self.test_mask)
        val_results = self.evaluate_model(scores_correct, y, self.val_mask)

        # pdb.set_trace()

        for metric_name in train_results:
            metrics[metric_name].append((train_results[metric_name].detach().item(),
                               val_results[metric_name].detach().item(),
                               test_results[metric_name].detach().item()
                               ))
        return train_results, test_results, val_results


             
        # return self.evaluate_model(scores_correct, y, self.train_mask), \
        #     self.evaluate_model(scores_correct, y, self.test_mask), \
        #     self.evaluate_model(scores_correct, y, self.val_mask),



    def evaluate_model(self, logits, target_labels, mask):
        metrics = {}
        y = target_labels[mask]
        with torch.no_grad():
            pred = logits[mask]
            if self.task == 'regression':
                metrics['loss'] = torch.sqrt(F.mse_loss(pred, y))
                metrics['accuracy'] = F.l1_loss(pred, y)
            elif self.task == 'classification':
                metrics['loss'] = F.cross_entropy(pred, y.long())
                metrics['accuracy'] = torch.Tensor([(y == pred.max(1)[1]).sum().item()/y.shape[0]])

            return metrics


    def init_gbdt_model(self, num_epochs):

        if self.task == 'regression':
            catboost_model_obj = CatBoostRegressor
            catboost_loss_fn = 'RMSE'
        else:
            catboost_model_obj = CatBoostRegressor
            catboost_loss_fn = 'MultiRMSE'


        return catboost_model_obj(iterations=num_epochs,
                                  depth=self.depth,
                                  learning_rate=self.gbdt_lr,
                                  loss_function=catboost_loss_fn,
                                  random_seed=0,
                                  nan_mode='Min')

    def fit_gbdt(self, pool, trees_per_epoch):
        gbdt_model = self.init_gbdt_model(trees_per_epoch)
        gbdt_model.fit(pool, verbose=False)
        return gbdt_model

    def append_gbdt_model(self, new_gbdt_model, weights):
        if self.gbdt_model is None:
            return new_gbdt_model
        return sum_models([self.gbdt_model, new_gbdt_model], weights=weights)


    def train_gbdt(self, gbdt_X_train, gbdt_y_train, cat_features, epoch,
                   gbdt_trees_per_epoch, gbdt_alpha):

        pool = Pool(gbdt_X_train, gbdt_y_train, cat_features=cat_features)
        epoch_gbdt_model = self.fit_gbdt(pool, gbdt_trees_per_epoch)
        
        self.gbdt_model = self.append_gbdt_model(epoch_gbdt_model, weights=[1, gbdt_alpha])
    



    def train(self, params, encoded_X, target, cat_features=None, num_boost_round=20, early_stopping_rounds=5):
        self.params.update(params)
        self.gbdt_model = None
        self.epoch_gbdt_model = None
        metrics = ddict(list)
        shrinkage_rate = 1.0
        best_iteration = None
        best_val_loss = LARGE_NUMBER
        train_start_time = time.time()

        self.num_samples = target.size(0)
        if self.task == 'regression':
            self.out_dim = 1
        elif self.task == 'classification':
            self.out_dim = int(target.max() + 1)
            target = target.squeeze()    


        print("Training until validation scores don't improve for {} rounds.".format(early_stopping_rounds))

        

        ## propagate the feature
        assert len(encoded_X.size()) == 2
        corrected_X = self.propagation_X.forward(self.graph, encoded_X)  
        ## cat the propagated features and orignal features
        feature = torch.cat((encoded_X, corrected_X), 1).cpu().numpy()



        for iter_cnt in range(num_boost_round):
            iter_start_time = time.time()

          
            scores = self._calc_data_scores(feature, iter_cnt)
            grad = self._calc_gradient(scores, target.cuda())

            
            self.train_gbdt(feature, grad, cat_features, iter_cnt, self.iter_per_epoch, gbdt_alpha=shrinkage_rate)

            # if iter_cnt > 0:
            #     shrinkage_rate *= self.params['learning_rate']
            if iter_cnt > 0:
                shrinkage_rate = self.params['learning_rate']


            train_metric, test_metric, val_metric = self._calc_loss(feature, target.cuda(), metrics)
            train_loss = train_metric['loss']
            test_loss = test_metric['loss']
            val_loss = val_metric['loss']
            test_accuracy = test_metric['accuracy']
         

            val_loss_str = '{:.10f}'.format(val_loss) if val_loss else '-'
            print("Iter {:>3}, Train's Loss: {:.10f}, Test's Loss: {}, Valid's Loss: {}, Test's Accuracy: {}, Elapsed: {:.2f} secs"
                  .format(iter_cnt, train_loss, test_loss, val_loss_str, test_accuracy.item(), time.time() - iter_start_time))

            
            
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_iteration = iter_cnt
                best_test_accuracy = test_accuracy


            # if iter_cnt - best_iteration >= early_stopping_rounds:
            #     print("Early stopping, best iteration is:")
            #     print("Iter {:>3}, Test Loss: {:.10f}".format(best_iteration, best_test_accuracy.item()))
            #     break


            
        self.best_iteration = best_iteration
        print("Training finished. Elapsed: {:.2f} secs".format(time.time() - train_start_time))

        plot(metrics, ['train', 'val', 'test'], 'CBS', 'CBS')
        exit()

        if self.task == 'regression':
            return best_test_loss.cpu().numpy()
        elif self.task == 'classification':
            return best_test_accuracy.cpu().numpy()

    