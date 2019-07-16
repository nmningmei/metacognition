#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:26:49 2018

@author: nmei
"""

working_dir = '/export/home/nmei/nmei/metacognition/data/dot'
saving_dir = '/export/home/nmei/nmei/metacognition/data/graph'
import os
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
import pydot
from glob import glob
files = glob(working_dir + '/*.dot')
for f in files:
    out_file = f.split('/')[-1].split('.')[0] + '.png'
    (graph,) = pydot.graph_from_dot_file(f)
    graph.write_png(os.path.join(saving_dir,out_file))