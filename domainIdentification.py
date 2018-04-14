import os
import csv
import collections
import random
import tensorflow as tf
import numpy as np
import tqdm
import re
import math
from tensorflow.contrib.tensorboard.plugins import projector
from gensim.parsing.porter import PorterStemmer

TEST_DIR=  "dataset/DATA/TEST"
TRAIN_DIR = "dataset/DATA/TRAIN"
VALID_DIR = "dataset/DATA/DEV"
TMP_DIR = "dataset/"
ANALOGIES_FILE = "dataset/eval/questions-words.txt"

domain_to_Labels={}

def get_latin(line):
    return ' '.join(''.join([i if ord(i) >=65 and ord(i) <=90 or  ord(i) >= 97 and ord(i) <= 122 else ' ' for i in line]).split())

p = PorterStemmer()

stoplist = set([w.rstrip('\r\n') for w in open("StopwordsM.txt")])

#BUILD THE SAME DICTIONARY OF WORDtoVECT
def build_wtnumb():
    dictionary = dict()
    reversed_dictionary = dict()
    with open(TMP_DIR + 'dictionaryBigBig1.tsv', 'r') as f:
        index=0
        AoA = [line.strip().split('\n') for line in f]
        #print(len(AoA))
        for a in AoA:
            if a[0] != "":
                dictionary[a[0]]=index
                index= index+1

    reversed_dictionary = {v: k for k, v in dictionary.iteritems()}
    return dictionary, reversed_dictionary

dictionary, reversed_dictionary=build_wtnumb()
#print(reversed_dictionary[0])
#print(len(dictionary))

#BUILT THE VECTOR RAPRESENTATION OF THE FINAL EMBEDDINGS MATRIX
def build_ntvector():
    dictionaryV = dict()
    reversed_dictionaryV = dict()
    with open(TMP_DIR + 'embeddingWBigBig1.tsv', 'r') as f:
        index=0
        text = f.read()
        AoA = text.strip().split('_')
        #print(len(AoA))
        for a in AoA:
            if a != "":
                #print("____\n"+a)
                a= a.replace('[', '').replace(']', '')
                a=a.split(' ')
                a=[float(x) for x in a if x]
                #print(len(a))
                dictionaryV[index]=a
                index= index+1
                c=len(a)

    #reversed_dictionaryV = {v: k for k, v in dictionaryV.iteritems()}
    return dictionaryV,c

dictionaryV,CENTROID_LENGHT  =build_ntvector()
#print(dictionaryV[0])

#UNIFY THE 2 DICTIONARY
def w2v(emb, dictw):
    final_dict= dict()
    for index in range(0,len(dictw)):
        w=dictw[index]
        v=emb[index]
        final_dict[w]=v
    return final_dict

dict= w2v(dictionaryV,reversed_dictionary)
#print(dict["UNK"])

#FUNCTION TO CALCULATE THE CENTROID AS SUM OF VECTOR DIVIDED BY NUMBER OF VECTOR
def calculateCentroid(directory,domain,file):
    #prepara il file da analizzare levando stopwords e separando le parole
    words = []
    if file.endswith(".txt"):
        with open(os.path.join(directory,domain, file)) as file:
            for line in file.readlines():
                # line=line.strip().lower()
                # line=p.stem_sentence(line)
                # line=line.lower()
                line = line.replace('+', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ')
                line = re.sub("(^|\W)\d+($|\W)", " ", line)
                line = get_latin(line)
                split = line.lower().strip().split()
                split = [word for word in split if (word not in stoplist) and (len(word) > 1)]
                words += split
    #print(words)
    # devo trasformare le parole in numeri con lo stesso dizionario utilizzato precedentemente il dizionario lo trovo inTMP_DIR + 'metadata.tsv'
    centroid=np.zeros(CENTROID_LENGHT)
    counter=0
    for wrd in words:
        if dict.has_key(wrd):
            counter=counter+1
            v=np.array(dict.get(wrd))
            centroid=centroid+v
    #print(centroid)
    #print(counter)
    if counter != 0:
        centroid=centroid/float(counter)
        return centroid
    else:
        return np.array([])
        print(directory+domain+file)


#FUNCTION TO TRANSFORM A DIRECTORY OF FILES IN A BATCH OF VECTORS AND A BATCH OF LABELS
def read_data(directory,max_number_of_file):
    i=0
    X=[]
    y=[]
    for domain in os.listdir(directory):
    #for dirpath, dnames, fnames in os.walk(directory):
        print(str(i)+": \""+domain+"\",")
        j=0
        for f in os.listdir(os.path.join(directory, domain)):
            j=j+1
            if True:#j<max_number_of_file:
                c=calculateCentroid(directory, domain,f)
                #c=c[:, np.newaxis]
                if c.size>0:
                    X.append(c)
                    #print(len(X))
                    y.append(i)
        i = i + 1
    return X,y


print("CREO TRAINING SET")

#TRAIN MY SVM MODEL
X,y = read_data(TRAIN_DIR,100)
print(len(X),len(y))

print("TRAINING MY MODEL")

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(np.array(X), np.array(y))

#TEST SET ANALISIS
print("CREO TEST SET")

files = []
X_test = []
for file in os.listdir(TEST_DIR):
    c = calculateCentroid(TEST_DIR, "", file)
    files.append(str(file))
    X_test.append(c)

dictn_Domain={0: "ANIMALS",
1: "RELIGION_MYSTICISM_AND_MYTHOLOGY",
2: "MATHEMATICS",
3: "WARFARE_AND_DEFENSE",
4: "TEXTILE_AND_CLOTHING",
5: "ROYALTY_AND_NOBILITY",
6: "METEOROLOGY",
7: "HISTORY",
8: "BUSINESS_ECONOMICS_AND_FINANCE",
9: "PHYSICS_AND_ASTRONOMY",
10: "MEDIA",
11: "HERALDRY_HONORS_AND_VEXILLOLOGY",
12: "SPORT_AND_RECREATION",
13: "GEOLOGY_AND_GEOPHYSICS",
14: "LITERATURE_AND_THEATRE",
15: "POLITICS_AND_GOVERNMENT",
16: "MUSIC",
17: "FOOD_AND_DRINK",
18: "CHEMISTRY_AND_MINERALOGY",
19: "NUMISMATICS_AND_CURRENCIES",
20: "ENGINEERING_AND_TECHNOLOGY",
21: "HEALTH_AND_MEDICINE",
22: "ART_ARCHITECTURE_AND_ARCHAEOLOGY",
23: "BIOLOGY",
24: "GAMES_AND_VIDEO_GAMES",
25: "COMPUTING",
26: "LANGUAGE_AND_LINGUISTICS",
27: "EDUCATION",
28: "LAW_AND_CRIME",
29: "TRANSPORT_AND_TRAVEL",
30: "PHILOSOPHY_AND_PSYCHOLOGY",
31: "FARMING",
32: "GEOGRAPHY_AND_PLACES",
33: "CULTURE_AND_SOCIETY"}

y_test = svm_model_linear.predict(X_test)
with open('dataset/test_answers.tsv', 'w') as f:
    for i in range(len(X_test)):
        f.write(files[i].replace('test_', '').replace('.txt', '') + '\t' + dictn_Domain[y_test[i]]+ "\n")

#EVALUATION MODEL ANALISIS
print("CREO EVALUETING SET")

Xt,yt = read_data(VALID_DIR,100)
print(len(Xt),len(yt))

print("EVALUETING MY MODEL")

svm_predictions = svm_model_linear.predict(Xt)
accuracy = svm_model_linear.score(np.array(Xt), np.array(yt))
print("total model accuracy: "+str(accuracy))
recall=recall_score(yt, svm_predictions, average=None)
print("recall: "+str(recall))
f1=f1_score(yt, svm_predictions, average=None)
print("f1: "+str(f1))
precision=precision_score(yt, svm_predictions, average=None)
print("precision: "+str(precision))
print("Confusion matrix:")
cm = confusion_matrix(yt, svm_predictions)
for x in range(0,len(cm)):
    print(cm[x])