#-*- coding:utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import operator
import pandas as pd
import ch
ch.set_ch()
def readFile(filename):
    '''
    读取数据编程pandas的frame形式
    '''
    filetype=filename.split('.')[-1]
##    headers=['one','two','three','four']
    f=open(filename)#避免中文出现错误
    pd.set_option('max_row',None)
    if filetype=='txt':
        data = pd.read_table(f,header=None,encoding='gb2312',
                             delim_whitespace=True)
    if filetype=='csv':
        data=pd.read_csv(f)
    return data
