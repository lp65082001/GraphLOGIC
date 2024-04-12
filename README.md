## Table of Contents

- [Introduction](#introduction)
- [Environment](#environment)
- [Preduction](#prediction)

## Introduction
GraphLOGIC: Graph-based Lethality predictor for OsteoGenesis Imperfecta for Collagen is implementation osteogenesis imperfecta of model for structural-informed lethality prediction. The code is published with "Developing clinical lethality predictive model of Osteogenesis Imperfecta by using graph neural network".

On the other hand, others checkpoint and structure data were available in https://doi.org/10.6084/m9.figshare.24633969

## Environment

1. **Hardware**
The model is training on the hardware Cpu:r5 3600 only with pytorch and pytorch_geometric framework.

2. **Package**
All of the script are run on python, you can conda to create visual environment and install it:
‘’‘
pip install -r requirements.txt  
’‘’

## Prediction

You can easy to run predicter.py to get result with position and mutation type:
‘’‘
python predicter.py -pos 20 -mut ser
’‘’
and the result:

## Citing

