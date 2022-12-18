# Add Gaussian White noise to a certain sequence

import numpy as np
from matplotlib import pyplot as plt
import random


# Gaussian White，SNR is the signal-to-noise ratio
def wgn(x, snr):
    x = np.array(x)
    for i in range(len(x)):
        temp = x[i]
        x[i] = temp + np.random.randint(-255,255)
        while (x[i]>255) or (x[i]<0):
            x[i] = temp + np.random.randint(-255, 255)
    return x
    # snr = 10 ** (snr / 10.0)
    # xpower = np.sum(np.log10(x) ** 2) / len(x)
    # npower = xpower / snr
    # #return np.random.randn(len(x)) * np.sqrt(npower)
    # #使每次生成的随机数相同
    # # np.random.seed(len(x))
    # perm = np.random.permutation(len(x))
    # prem = np.random.randn(len(x))

    # # z_perm=((perm-np.mean(perm))/np.var(perm))
    # return prem * np.sqrt(npower)


def incrementSeq(seq,start,end,Len):
    '''
    :param seq: Original time series
    :param start: the start position of the subsequence
    :param end: the end position of the subsequence
    :param Len: the length of seq
    :return: the sequence after adding noise
    '''

    _seq=[]

    seq_withnoise=wgn(seq[start:end+1], 10)

    for i in range(0,start):
        _seq.append(0)

    '''mu = 0
    sigma = 1'''
    for i in range(end-start+1):
        '''_seq.append(seq[i] + random.gauss(mu,sigma))'''
        _seq.append(seq_withnoise[i])
    #print(random.gauss(mu,sigma))
    #print(_seq)
    for i in range(end+1,Len):
        _seq.append(0)
    return _seq






