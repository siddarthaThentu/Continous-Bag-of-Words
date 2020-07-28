# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:38:26 2020

@author: siddarthaThentu
"""

import numpy as np
from scipy import linalg
from collections import defaultdict

def get_dict(data):
    
    words = sorted(list(set(data)))
    word2Ind = {}
    ind2Word = {}
    
    for idx,word in enumerate(words):
        if word not in word2Ind:
            word2Ind[word] = idx
            ind2Word[idx] = word
    
    return word2Ind,ind2Word

def softmax(z):
    
    numerator = np.exp(z)
    denominator = np.sum(numerator,axis=0)
    
    y_hat = np.divide(numerator,denominator)
    
    return y_hat

def RELU(z):
    
    result = z.copy()
    result[result<0] = 0
    
    return result

def get_idx(context_words,word2Ind):
    
    idxs = []
    
    for word in context_words:
        idxs.append(word2Ind[word])
    
    return idxs

def pack_idx_with_freq(context_words,word2Ind):
    
    freq_dict = defaultdict(int)
    for word in context_words:
        freq_dict[word] += 1
    idxs = get_idx(context_words,word2Ind)
    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx,freq))
    
    return packed
        
def get_vectors(data,word2Ind,V,C):
    
    i = C
    while True:
        x = np.zeros(V)
        y = np.zeros(V)
        center_word = data[i]
        y[word2Ind[center_word]] = 1
        context_words = data[(i-C):i] + data[(i+1):(i+C+1)]
        num_context_words = len(context_words)
        for idx,frequency in pack_idx_with_freq(context_words,word2Ind):
            x[idx] = frequency/num_context_words
        yield x,y
        i+=1
        if(i>=len(data)):
            print("i is being set to 0")
            i=0
        
def get_batches(data,word2Ind,V,C,batch_size):
    
    batch_x = []
    batch_y = []
    
    for x,y in get_vectors(data,word2Ind,V,C):
        while(len(batch_x)<batch_size):
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T,np.array(batch_y).T
        
def compute_pca(data,n_components=2):
    
    m,n = data.shape
    data -= data.mean(axis=0)
    R = np.cov(data,rowvar=False)
    evals,evecs = linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evecs = evecs[:,:n_components]
    
    return np.dot(evecs.T,data.T).T
        
        
        
        
        
        









