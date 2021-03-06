# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_U4iIxeXe3W-4IkAXykm9VcqA5pPXvGQ
"""

def get_metric(input,pred_text):
    '''Calcuating the performance metric')'''
    import numpy as np

    
    def jaccard(str1, str2):
        
        a = set(str(str1).lower().split()) 
        b = set(str(str2).lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    
    actual_text = input['selected_text'].values
    scores = []
    for i in range(len(pred_text)):
        scores.append(jaccard(pred_text[i],actual_text[i]))

    res = np.array(scores).mean()
    print('Successfully ran!!!!!')
    print('*'*50)
    return res