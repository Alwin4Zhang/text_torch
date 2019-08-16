import json
import sklearn
from bidict import bidict
import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split
import pickle
import os
from os.path import join as join_path,dirname

ENTITY = 'ENTITY'
CONCEPT = 'CONCEPT'
POS = 'N'
O = 'O'

# 标签简化，可以减少存储和提高后续匹配效率
LABELS = {'ENTITY_B':'A',
'ENTITY_M':'B',
'ENTITY_E':'C',
'CONCEPT_B':'D',
'CONCEPT_M':'E',
'CONCEPT_E':'F',
'S_CC':'G',
'ENTITY_FB1':'H',
'ENTITY_FB2':'I',
'ENTITY_TB1':'J',
'ENTITY_TB2':'K',
'CONCEPT_FB1':'L',
'CONCEPT_FB2':'M',
'CONCEPT_TB1':'N',
'CONCEPT_TB2':'Z',
'ENTITY_CONCEPT_S':'P',
'CONCEPT_ENTITY_S':'Q',
'ENTITY_ENTITY_S':'R',
'CONCEPT_CONCEPT_S':'S ',
'O':'O'}

BI_LABELS = bidict(LABELS)

def labeled_2_conll(line, is_print=False):
    conll_format = []
    if line:
        try:
            item = json.loads(line)
            text = item.get('text')
            labels = item.get('labels')
            if text and labels:
                labels = sorted(labels, key=lambda x: x[0])
                if is_print:
                    print(text)
                    print('======================')
                text_len = len(text)
                conll_format = [(x, POS, O) for x in text]
                for label in labels:
                    start = label[0]
                    end = label[1]
                    type_ = label[2]
                    if is_print:
                        print(text[start: end], type_)
                    
                    if start > 2:
                        offset = start-2
                        if O == conll_format[offset][2]:
                            conll_format[offset] = [text[offset], POS, type_+'_FB2']
                            
                    if start > 1:
                        offset = start-1
                        if O == conll_format[offset][2]:
                            conll_format[offset] = [text[offset], POS, type_+'_FB1']   
                        else:
                            temp = conll_format[offset][2]
                            temp = temp[0: temp.rindex('_')]
                            conll_format[offset] = [text[offset], POS, temp+'_'+type_+'_S']
                            
                    if end < text_len:
                        if O == conll_format[end][2]:
                            conll_format[end] = [text[end], POS, type_+'_TB1']
                            
                    if end + 1 < text_len and O == conll_format[end+1][2]:
                        conll_format[end+1] = [text[end+1], POS, type_+'_TB2']
                    for i in range(start+1, end-1):
                        conll_format[i] = [text[i], POS, type_+'_M']
                    conll_format[start] = [text[start], POS, type_+'_B']
                    conll_format[end-1] = [text[end-1], POS, type_+'_E']
        except ValueError:
            pass
    if is_print:
        print('======================')
    
    return conll_format

train_file = '/Users/bello/Desktop/train_corpus.txt'
test_file = '/Users/bello/Desktop/test_corpus.txt'


corpus = []
for line in open(train_file,'r'):
    # line = line.strip()
    conll_format = labeled_2_conll(line)
    corpus.append(conll_format)

for line in open(test_file,'r'):
    # line = line.strip()
    conll_format = labeled_2_conll(line)
    corpus.append(conll_format)

all_words = []
for row in corpus:
    for tp in row:
        all_words.append(tp[0])     

# words
sr_all_words = pd.Series(all_words)
sr_all_words = sr_all_words.value_counts()
set_words = sr_all_words.index
set_ids = range(1, len(set_words)+1)
word2id = pd.Series(set_ids,index=set_words)
id2word = pd.Series(set_words,index=set_ids)
word2id["unknow"] = len(word2id)+1

# tags 
tags = list(LABELS.keys())
tag_ids = range(len(tags))
tag2id = pd.Series(tag_ids,index=tags)
id2tag = pd.Series(tags,index=tag_ids)

max_len = 60
def X_padding(words):
    ids = list(word2id[words])
    if len(ids) >= max_len:  
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) 
    return ids

def y_padding(tags):
    ids = list(tag2id[tags])
    if len(ids) >= max_len: 
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) 
    return ids

x,y = [],[]
for row in corpus:
    row_words = list(map(lambda x:x[0],row))
    row_labels = list(map(lambda x:x[2],row))

    words_ids = X_padding(row_words)
    labels_ids = y_padding(row_labels)
    x.append(words_ids)
    y.append(labels_ids)

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,  test_size=0.2, random_state=43)

print(y_test,y_train)

save_dir = join_path(dirname(__file__),'Bellodata.pkl')
with open(save_dir, 'wb') as outp:
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)
    pickle.dump(x_train, outp)
    pickle.dump(y_train, outp)
    pickle.dump(x_test, outp)
    pickle.dump(y_test, outp)
    pickle.dump(x_valid, outp)
    pickle.dump(y_valid, outp)
print('** Finished saving the data.')