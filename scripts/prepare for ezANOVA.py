# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:26:49 2019

@author: ning
"""

import os
from glob import glob

import pandas as pd
import numpy as np

working_dir = '../results/linear_mixed'
working_data = [item for item in\
                glob(os.path.join(working_dir,'*.csv')) if\
                ('fit' not in item)]
n_back = 4
att,pos = [pd.read_csv(f) for f in working_data]

def process(df):
    res = []
    for ii,row in df.iterrows():
        for n_ in range(n_back):
            temp = {}
            attrs = [item for item in row.index if (f'_{n_back - n_}' in item)]
            for col in np.array(list(row.index))[-3:]:
                temp[col] = [row[col]]
            for col in attrs:
                temp[col.split('_')[0]] = [row[col]]
            temp['time'] = [n_back - n_]
            temp = pd.DataFrame(temp)
            res.append(temp)
    res = pd.concat(res)
    return res

pos = process(pos.copy())
att = process(att.copy())

pos.to_csv(os.path.join(working_dir,
                        'for_ezANOVA_pos.csv'),
           index = False)
att.to_csv(os.path.join(working_dir,
                        'for_ezANOVA_att.csv'),
           index = False)

