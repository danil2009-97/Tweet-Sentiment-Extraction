# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_U4iIxeXe3W-4IkAXykm9VcqA5pPXvGQ
"""

def preprocess_data(input,max_len,tokenizer):
  '''Get the input data and preprocess the data so that it can be passed as i/p to the model'''
  import numpy as np
  import pandas as pd 
  from transformers import RobertaTokenizer
  from tqdm import tqdm
  
  print('Preprocessing input data')
  if 'text' in input.columns:
    input['text'] = input['text'].apply(lambda x : str(x).lower())
  if 'selected_text' in input.columns:
    input['selected_text'] = input['selected_text'].apply(lambda x : str(x).lower())
  MAX_LEN=max_len
  count = input.shape[0]
  input_ids = np.zeros((count,MAX_LEN),dtype='int32')
  attention_mask = np.zeros((count,MAX_LEN),dtype='int32')
  ip_data = input[['text','sentiment']]
    
  print('Getting input_ids and attention_mask for the input')
  
  for i,each in tqdm(enumerate(ip_data.values)):
    val = tokenizer.encode_plus(each[0],each[1],add_special_tokens=True,max_length=128,return_attention_mask=True,pad_to_max_length=True                                            ,return_tensors='tf',verbose=False)
    input_ids[i] = val['input_ids']
    attention_mask[i] = val['attention_mask']
  
  print('Shape of input id and attention mask:',input_ids.shape,attention_mask.shape)
    
  return input_ids,attention_mask,input