# -*- coding: utf-8 -*-
import sys
import re
import numpy as np
import operator
import math

def read_model(filename):
    model_type = ''
    model = {}
    model['class_pr'] = {}
    curr_class = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.split(',')
            if line[0] == '#class':
                curr_class = line[1].replace('\n','')
                model[curr_class]={}
                model['class_pr'][curr_class] = line[2].replace('\n','')
            elif line[0] == '#model_type':
                model_type = line[1].replace('\n','')
            else:
                term = line[0]
                pr = float(line[1].replace('\n',''))
                model[curr_class][term]=pr
    f.close()        
    return model, model_type

def multinomial_classify(model, test_filename):
    stop_words = np.loadtxt('stoplist.txt',dtype=str)
    f = open(test_filename)
    data = f.read()
    f.close()
    exp = r'\w+(\.?\w+)*'
    tokens = re.finditer(exp,data)
    terms = {}
    for token in tokens:
        if token:
            token = token.group()
            if token not in stop_words:
                token = token.replace('\n','')
                if terms.has_key(token):
                    terms[token] += 1
                else:
                    terms[token] = 1
    predictions = {}
    for c in model:
        if c != 'class_pr':
            predictions[c] = float(model['class_pr'][c])
            for term in terms:
                if model[c].has_key(term):
                    predictions[c] += float(model[c][term])
    pclass = max(predictions.iteritems(),key=operator.itemgetter(1))[0]
    return pclass, predictions

def binomial_classify(model, test_filename):
    stop_words = np.loadtxt('stoplist.txt',dtype=str)
    f = open(test_filename)
    data = f.read()
    f.close()
    exp = r'\w+(\.?\w+)*'
    tokens = re.finditer(exp,data)
    terms = {}
    for token in tokens:
        if token:
            token = token.group()
            if token not in stop_words:
                token = token.replace('\n','')
                if terms.has_key(token):
                    terms[token] += 1
                else:
                    terms[token] = 1
    predictions = {}
    for c in model:
        if c != 'class_pr':
            predictions[c] =  float(model['class_pr'][c])
            for term in terms:
                if model[c].has_key(term):
                    predictions[c] += math.log(float(model[c][term]),10)
                    del model[c][term]
            for term in model[c]:
                predictions[c] += math.log(float(1-model[c][term]))
    pclass = max(predictions.iteritems(),key=operator.itemgetter(1))[0]
    return pclass, predictions         
    
def main(model_filename, test_filename):
    model, model_type = read_model(model_filename)
    if model_type == 'multinomial':
        pclass, predictions = multinomial_classify(model, test_filename)
    
    if model_type == 'binomial':
        pclass, predictions = binomial_classify(model, test_filename)
    
    print pclass
    return
    
if __name__=='__main__':
    if len(sys.argv) != 3:
        print "usage: python test.py <model_filename> <test_filename>"
    else:
        main(model_filename=sys.argv[1], test_filename=sys.argv[2])