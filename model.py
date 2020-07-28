# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:55:18 2020

@author: siddarthaThentu
"""

import numpy as np
from utils import RELU,softmax,get_batches

def initialize_model(N,V,random_seed=1):
    
    np.random.seed(random_seed)
    
    W1 = np.random.rand(N,V)
    W2 = np.random.rand(V,N)
    b1 = np.random.rand(N,1)
    b2 = np.random.rand(V,1)
    
    return W1,W2,b1,b2

def forward_prop(x,W1,W2,b1,b2):
    
    h = np.dot(W1,x) + b1
    h = RELU(h)
    z = np.dot(W2,h) + b2
    
    return z,h
    
def compute_cost(y,yhat,batch_size):
    
    logprobs = np.multiply(np.log(yhat),y) + np.multiply(np.log(1-yhat),1-y)
    cost = - np.sum(logprobs)/batch_size
    cost = np.squeeze(cost)
    
    return cost

def back_prop(x,yhat,y,h,W1,W2,b1,b2,batch_size):
    
    l1 = np.dot(W2.T,yhat-y)
    l1 = RELU(l1)
    
    grad_W1 = np.dot(l1,x.T)/batch_size
    grad_W2 = np.dot(yhat-y,h.T)/batch_size
    grad_b1 = np.dot(l1,np.ones([1,batch_size],dtype=int).T)/batch_size
    grad_b2 = np.dot(yhat-y,np.ones([1,batch_size],dtype=int).T)/batch_size    
    
    return grad_W1,grad_W2,grad_b1,grad_b2

def gradient_descent(data,word2Ind,N,V,C,num_iters,alpha=0.03):
    
    W1,W2,b1,b2 = initialize_model(N,V,random_seed=282)
    batch_size = 128
    iters = 0
    
    for x,y in get_batches(data,word2Ind,V,C,batch_size):
        
        z,h = forward_prop(x,W1,W2,b1,b2)
        yhat = softmax(z)
        cost = compute_cost(y,yhat,batch_size)
        
        if((iters+1)%10==0):
            print("Iteration ",iters+1,"  cost: ",cost)
        
        grad_W1,grad_W2,grad_b1,grad_b2 = back_prop(x,yhat,y,h,W1,W2,b1,b2,batch_size)
        
        W1 -= alpha*grad_W1
        W2 -= alpha*grad_W2
        b1 -= alpha*grad_b1
        b2 -= alpha*grad_b2
        
        iters += 1
        
        if(iters==num_iters):
            break
        if(iters%100==0):
            alpha*=0.66
        
    return W1,W2,b1,b2

def predict_word(context_words,word2Ind,ind2Word,V,W1,W2,b1,b2):
    
    def word_to_one_hot_vector(word,word2Ind,V):
        one_hot_vector = np.zeros(V)
        one_hot_vector[word2Ind[word]] = 1
        return one_hot_vector
    
    context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
    context_words_vectors = np.mean(context_words_vectors, axis=0)
    context_words_vectors.shape = (V,1)
    x = context_words_vectors
    print(x)
    h = RELU(np.dot(W1,x)+b1)
    z = np.dot(W2,h)+b2
    y_pred = softmax(z)
    idx = np.argmax(y_pred)
    
    return x,ind2Word[idx]
    
    
    
    
    
    
    
    