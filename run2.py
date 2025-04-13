# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 20:10:06 2022

@author: 24412
"""

import math
import os
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn


import numpy as np

import torch.nn.functional as F
from pytorch_i3d import InceptionI3d

import cv2
from keytotext import pipeline

import language
from dotenv import load_dotenv

from itertools import chain

import pickle
import google.generativeai as genai
load_dotenv("posts/nlp/.env", override=True)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()
genai.configure(api_key="AIzaSyDA4V5JD2syQVoj8Jija_Y2VJBDcjN3maw")

def generate_sentence(predicted_words):
    """Takes a list of predicted words and generates a meaningful sentence using Gemini API"""
    prompt = f"Form a short meaningful grammatically correct one sentence from these words which can be useful for communicating through sign language in healthcare domain and the order of the words should be same in the final sentence and dont give response in bold letters: {', '.join(predicted_words)}"
    print("Words sent to gemini:  ",predicted_words)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text if response else "Failed to generate sentence."
def load_rgb_frames_from_video():
    
    vidcap = cv2.VideoCapture(0)

    
    frames = []
    
    offset = 0
    text = " "
    batch = 40
    text_list = []
    word_list = []
    sentence = ""
    text_count = 0
    text_count_gen = 0
    """
    To maintain the continous flow of actions we bring in the the batch size and offest modulo factor.
    the batch size and the offset can be varied.
    
    """
    
    while True:
        ret, frame1 = vidcap.read()
        offset = offset + 1
        font = cv2.FONT_HERSHEY_TRIPLEX
        
        if ret == True:
            
            w, h, c = frame1.shape
            sc = 224 / w
            sx = 224 / h
            frame = cv2.resize(frame1, dsize=(0, 0), fx=sx, fy=sc)
            frame1 = cv2.resize(frame1, dsize = (1280,720))
    
            frame = (frame / 255.) * 2 - 1
            
            if offset > batch:
                frames.pop(0)
                frames.append(frame)
                
                if offset % 20 == 0:
                    text = run_on_tensor(torch.from_numpy((np.asarray(frames, dtype=np.float32)).transpose([3, 0, 1, 2])))
                    if text != " " and text != 'insurance' and text != 'help':
                        text_count += 1
                        text_count_gen += 1
                        
                        if text != "yes":
                            if not text_list or text_list[-1] != text:
                                text_list.append(text)
                                sentence = " ".join(text_list)
                        
                        # Using Gemini for sentence generation
                        if text_count_gen > 2 and text == 'yes':
                            sentence = generate_sentence(text_list)
                            print(sentence)
                            text_count_gen = 0
                            
                            
                
                cv2.putText(frame1, sentence, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('frame', frame1)

            else:
                frames.append(frame)
                if offset == batch:
                    text = run_on_tensor(torch.from_numpy((np.asarray(frames, dtype=np.float32)).transpose([3, 0, 1, 2])))
                    if text != " ":
                        text_count += 1
                        text_count_gen += 1
                        
                        if not text_list or text_list[-1] != text:
                            text_list.append(text)
                            sentence = " ".join(text_list)
                        
                        # Using Gemini for sentence generation
                        if text_count_gen > 2 and text == 'yes':
                            sentence = generate_sentence(text_list)
                            print(sentence)
                            text_count_gen = 0
                                
                        
                        cv2.putText(frame1, sentence, (120, 520), font, 0.9, (0, 255, 255), 2, cv2.LINE_4)
                
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        else:
            break
    
            
    vidcap.release()
    cv2.destroyAllWindows()
    


def load_model(weights, num_classes):
    global i3d 
    i3d = InceptionI3d(39, in_channels=3)
    i3d.replace_logits(num_classes)
    
    # Load checkpoint and extract state_dict
    checkpoint = torch.load(weights)
    state_dict = checkpoint['state_dict']
    
    # Adjust for DataParallel prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]  # Remove DataParallel prefix
        new_state_dict[key] = value
    
    # Load adjusted state dict
    i3d.load_state_dict(new_state_dict)
    
    i3d.cuda()
    i3d = nn.DataParallel(i3d)  # Apply DataParallel AFTER loading
    i3d.eval()
    
    #Loading the KeytoText model
    
    global nlp
    nlp = pipeline("k2t-new") # The pre-trained models available are 'k2t', 'k2t-base', 'mrm8488/t5-base-finetuned-common_gen', 'k2t-new'
    global params
    params = {"do_sample":True, "num_beams": 5, "no_repeat_ngram_size":2, "early_stopping":True}
    
    #Loading the NGram model
    
    with open("NLP/nlp_data_processed", "rb") as fp:   # Unpickling
           train_data_processed = pickle.load(fp)
    
    global n_gram_counts_list
    with open("NLP/nlp_gram_counts", "rb") as fp:   # Unpickling
        n_gram_counts_list = pickle.load(fp)
        
    global vocabulary
    vocabulary = list(set(chain.from_iterable(train_data_processed)))
    
    
    load_rgb_frames_from_video()
    

def run_on_tensor(ip_tensor):

    ip_tensor = ip_tensor[None, :]
    
    t = ip_tensor.shape[2] 
    ip_tensor.cuda()
    per_frame_logits = i3d(ip_tensor)

    predictions = F.upsample(per_frame_logits, t, mode='linear')

    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    arr = predictions.cpu().detach().numpy()[0] 

    print(float(max(F.softmax(torch.from_numpy(arr[0]), dim=0))))
    print(wlasl_dict[out_labels[0][-1]])
    
    """
    
    The 0.5 is threshold value, it varies if the batch sizes are reduced.
    
    """
    if max(F.softmax(torch.from_numpy(arr[0]), dim=0)) > 0.25:
        return wlasl_dict[out_labels[0][-1]]
    else:
        return " " 
        
    
def create_WLASL_dictionary():
    
    global wlasl_dict 
    wlasl_dict = {}
    
    with open('preprocess/wlasl_class_39.txt') as file:
        for line in file:
            split_list = line.split()
            if len(split_list) != 2:
                key = int(split_list[0])
                value = split_list[1] + " " + split_list[2]
            else:
                key = int(split_list[0])
                value = split_list[1]
            wlasl_dict[key] = value
            

if __name__ == '__main__':
   
    # ================== test i3d on a dataset ==============
    # need to add argparse

    mode = 'rgb'
    num_classes = 39
    save_model = './checkpoints/'

    root = '../../data/WLASL2000'

    train_split = 'preprocess/nslt_{}.json'.format(num_classes)
    weights = 'archived/checkpoint_35.pth'
    # weights = 'archived/asl39/nslt_39_000745_0.089286.pt'
    # weights = 'archived/asl39/nslt_39_004172_0.339286.pt'
    # weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    # weights = 'archived/asl1000/FINAL_nslt_1000_iters=5104_top1=47.33_top5=76.44_top10=84.33.pt'
    # weights = 'archived/asl300/FINAL_nslt_300_iters=2997_top1=56.14_top5=79.94_top10=86.98.pt'
    # weights = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
    
    
    create_WLASL_dictionary()
    
    load_model(weights, num_classes)
        
    
    