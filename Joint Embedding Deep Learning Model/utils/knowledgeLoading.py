import os
import json
import numpy as numpy#import gensim

class InputRawKnowledge(object):
    def __init__(self, unique_id, candidateEntities):
        self.unique_id = unique_id
        self.candidateEntities = candidateEntities

def read_rawKnowledge(input_path):
    rawKnowledge=[]
    for file in os.listdir(input_path):
        unique_id = int(file.replace('.txt',''))
        candidateEntities = []
        candidateEntities.append([])
        with open('%s/%s' % (input_path, file), "r", encoding='utf-8') as reader:
            while True:
                line=reader.readline().strip()
                if len(line)<1:
                    break
                if line=='Null':
                    candidateEntities.append([])
                else:
                    candidateEntities.append([w.split(',') for w in line.split('\t')])
        candidateEntities.append([])
        rawKnowledge.append(
            InputRawKnowledge(unique_id=unique_id, candidateEntities=candidateEntities))
    uniqie_rawKnowledge={}
    for f in rawKnowledge:
        uniqie_rawKnowledge[f.unique_id]=f.candidateEntities
    return uniqie_rawKnowledge




#rawKnowledge = read_rawKnowledge("../data/trainDataKnowledge")
#print(rawKnowledge[1])
#print(len(rawKnowledge))
