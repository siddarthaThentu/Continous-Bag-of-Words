# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:06:46 2020

@author: siddarthaThentu
"""

import re
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from utils import *
from model import *

#Load tokenize and process the data
with open("shakespeare.txt","r") as fp:
    data = fp.read()
data = re.sub("[,!?;-]+",".",data)    
data = word_tokenize(data)
data = [ch.lower() for ch in data if ch.isalpha() or ch=="."]
#print("Number of tokens : ",len(data),"\n",data[:15])  
  
#compute the frequency distribution of the words in the dataset
fdist = nltk.FreqDist(word for word in data)

word2Ind,ind2Word = get_dict(data)

#half of context window size
C = 2
#Number of hidden layers
N = 50
#Length of vocabulary
V = len(word2Ind)
#number of iterations
num_iters = 150

W1,W2,b1,b2 = gradient_descent(data,word2Ind,N,V,C,num_iters,alpha=0.03)

embs = (W1.T+W2)/2.0

from model import *
trail_cntx_words = ["a","kingdom","a","stage"]
trail_cntx_words2 = ["the","brightest","of","invention"]
trail_cntx_words3 = ["think","when","talk","of"]

vec1,word1 = predict_word(trail_cntx_words,word2Ind,ind2Word,V,W1,W2,b1,b2)
vec2,word2= predict_word(trail_cntx_words2,word2Ind,ind2Word,V,W1,W2,b1,b2)
vec3,word3 = predict_word(trail_cntx_words3,word2Ind,ind2Word,V,W1,W2,b1,b2)

print(word1,word2,word3)




