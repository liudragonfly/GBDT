# -*- coding:utf-8 -*- 
__author__ = 'Dragonfly'
from gbdt.data import DataSet
from gbdt.model import GBDT

if __name__ == '__main__':
    data_file = './data/credit.data.csv'
    dateset = DataSet(data_file)
    gbdt = GBDT(max_iter=20, sample_rate=0.8, learn_rate=0.5, max_depth=7, loss_type='binary-classification')
    gbdt.fit(dateset, dateset.get_instances_idset())