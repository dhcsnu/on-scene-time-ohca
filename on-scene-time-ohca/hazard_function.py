#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score as ibs
import time

class BuildRSF():
    def __init__(self, train, val, vars):
        self.vars = vars
        self.x_train, self.y_train = self.build_dataset(train)
        self.x_val, self.y_val = self.build_dataset(val)
    
    def build_dataset(self, data):
        assert isinstance(data, pd.DataFrame), "The variable is not a pandas DataFrame"
        
        x = data[self.vars]
        # If the patient achieved on-scene rosc, get the No. of 2-min cycles till ROSC, else get the No. of 2-min cycles on scene
        y = data.apply(lambda x: (True, x.crosc_time) if x.s_rosc==1 else (False, x.csti), axis=1)
        y = np.array(y.tolist(), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    
        return x, y
    
    def grid_search(self, parameters, cv=5):
        t0 = time.time()
        rsf = RandomSurvivalForest()
        clf = GridSearchCV(rsf, parameters, cv=cv)
        clf.fit(self.x_train, self.y_train)
        print(clf.best_params_)
        print("Elapsed time: %.1f s" %(time.time() - t0))
        return clf.best_params_
    
    def fit(self, **kwargs):
        self.rsf = RandomSurvivalForest(**kwargs)
        self.rsf.fit(self.x_train, self.y_train)
        
        surv = self.rsf.predict_survival_function(self.x_val)
        cycles = np.arange(0,15)
        preds = np.asarray([[fn(t) for t in cycles] for fn in surv])
        
        print("Evaluation results in validation set")
        print("C-index: %.3f" %(self.rsf.score(self.x_val, self.y_val)))
        print("Integrated Brier Score: %.3f" %(ibs(self.y_train, self.y_val, preds, cycles)))
        
        return self.rsf
    
    
def build_hazard_function(data, rsf):
    surv = rsf.predict_survival_function(data, return_array=True)
    hazard_function = -np.diff(surv) / surv[:,:-1]
    hazard_function = pd.DataFrame(hazard_function)
    
    return hazard_function

