# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import re
import math

def calculate_class_pr(train_dir):
    class_dirs = os.listdir(train_dir)
    class_files = {}
    total_file_count = 0
    for directory in class_dirs:
        class_files[directory]= os.listdir(train_dir +'/'+ directory)
        total_file_count += len(class_files[directory])
    
    pr = {}
    for class_name, file_names in class_files.iteritems():
        pr[class_name] = math.log(float(len(file_names))/total_file_count,10)
    return pr

def calculate_term_binomial(train_dir):
    class_dirs = os.listdir(train_dir)
    class_files = {}
    exp = r'\w+(\.?\w+)*'
    terms = {}
    total_classes = len(train_dir)
    total_files = {}
    stop_words = np.loadtxt('stoplist.txt',dtype=str)
    for c in class_dirs:
        class_files[c] = os.listdir(train_dir+'/'+c)
        terms[c] = {}
        
        for file_name in class_files[c]:
            path = train_dir + '/' + c + '/' + file_name
            f = open(path)
            data = f.read()
            f.close()
            tokens = re.finditer(exp, data)
            f_dict = {}
            for token in tokens:
                if token:
                    token = token.group()
                    if token not in stop_words:
                        token = token.replace('\n','')
                        if f_dict.has_key(token) == False:
                            f_dict[token] = 1
                            if terms[c].has_key(token) == False:
                                terms[c][token] = 1
                            else:
                                terms[c][token] += 1
    pr = {}
    for c in terms:
        pr[c] = {}
        
    for c in terms:
        for token in terms[c]:
            pr[c][token] = (float(terms[c][token] + 1)/(len(class_files[c]) + len(terms)))
            for other_c in terms:
                if c != other_c and pr[other_c].has_key(token) == False:
                    pr[other_c][token] = (float(1)/(len(class_files[c]) + len(terms)))
    return pr

def calculate_term_multinomial(train_dir):
    class_dirs = os.listdir(train_dir)
    class_files = {}
    exp = r'\w+(\.?\w+)*'
    terms = {}
    vocab_count = 0
    for directory in class_dirs:
        class_files[directory]= os.listdir(train_dir +'/'+ directory)
        stop_words = np.loadtxt('stoplist.txt',dtype=str)
        terms[directory] = {}
        for file_name in class_files[directory]:
            path = train_dir + '/' + directory + '/' + file_name
            f = open(path)
            data = f.read()
            f.close()
            
            exp = r'\w+(\.?\w+)*'
            tokens = re.finditer(exp,data)
            for token in tokens:
                if token:
                    token = token.group()
                    if token not in stop_words:
                        token = token.replace('\n','')
                        if terms[directory].has_key(token):
                            terms[directory][token] += 1
                        else:
                            terms[directory][token] = 1
                        vocab_count += 1
    pr = {}
    
    for directory in terms:
        pr[directory] = {}
        
    for directory in terms:
        for token in terms[directory]:
            pr[directory][token] = math.log(float(terms[directory][token] + 1)/(vocab_count + len(terms[directory])),10)
            for other_dir in terms:
                if other_dir != directory and terms[other_dir].has_key(token) == False:
                    pr[other_dir][token]=math.log(float(0+1)/(vocab_count + len(terms[other_dir])),10)
                    
    return pr
def main(train_dir,output_filename):
    class_pr = calculate_class_pr(train_dir)
    term_pr = calculate_term_multinomial(train_dir)
    f = open(output_filename + '_multinomial', 'w')
    f.write('#model_type,multinomial')
    f.write('\n')            
    for class_name in class_pr:        
        f.write('#class')
        f.write(',')
        f.write(class_name)
        f.write(',')
        f.write(str(class_pr[class_name]))
        f.write('\n')
        for token in term_pr[class_name]:
            f.write(token)
            f.write(',')
            f.write(str(term_pr[class_name][token]))
            f.write('\n')
    f.close()
    
    term_pr = calculate_term_binomial(train_dir)
    
    f = open(output_filename + '_binomial', 'w')
    f.write('#model_type,binomial')
    f.write('\n')
    for class_name in class_pr:        
        f.write('#class')
        f.write(',')
        f.write(class_name)
        f.write(',')
        f.write(str(class_pr[class_name]))
        f.write('\n')
        for token in term_pr[class_name]:
            f.write(token)
            f.write(',')
            f.write(str(term_pr[class_name][token]))
            f.write('\n')
    f.close()    
    
    return
    
if __name__=='__main__':
    #main('train', 'model1')
    if len(sys.argv) != 3:
        print "usage: python train.py <directory_name> <output_filename>"
    else:
        main(train_dir=sys.argv[1], output_filename=sys.argv[2])