#Ken Bacchiani
#classify.py
#Code written for Introduction to Artificial intelligence course at the University of Wisconsin-Madison

import os
import math
from collections import Counter

#This function creates a dictionary for the bag of words given a filepath
#and a vocabulary
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # TODO: add your code here
    f = open(filepath, "r")
    for line in f:
        val = line.rstrip('\n')
        if(val in vocab):
            if val in bow.keys():
                bow[val] = bow[val] + 1
            else:
                bow[val] = 1
        else:
            if(None in bow.keys()):
                bow[None] = bow[None] + 1
            else:
                bow[None] = 1
    f.close()
    return bow

#Creates a list of dictionaries given a directory and vocab list.
# The dictionaries have a label for the year and a bag of words for the vocab of each file
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    dataset = []
    # TODO: add your code here
    sixteen = "2016/"
    twenty = "2020/"
    for filename in os.listdir(directory + twenty):
        toAdd = { 'label': '2020' , 'bow' : create_bow(vocab, directory  +  twenty + filename) }
        dataset.append(toAdd)
    for filename in os.listdir(directory + sixteen):
        toAdd = { 'label': '2016' , 'bow' : create_bow(vocab, directory +  sixteen + filename) }
        dataset.append(toAdd)
    return dataset

#Creates a vocabulary list from a directory, only if it comes up equal to or more than the cutoff
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """
    vocab = []
    # TODO: add your code here
    all = []
    sixteen = "2016/"
    twenty = "2020/"
    directorySixteen = directory + sixteen
    directoryTwenty = directory + twenty
    for filename in os.listdir(directorySixteen):
        f = open(directorySixteen + filename, "r")
        for line in f:
            all.append(line.rstrip('\n'))
        f.close()
    for filen in os.listdir(directoryTwenty):
        t = open(directoryTwenty + filen, "r")
        for lin in t:
            all.append(lin.rstrip('\n'))
    t.close()
    all.sort()
    if(cutoff == 1):
        vocab = list(dict.fromkeys(all))
        return(vocab)
    first = all[0]
    number = 1
    for i in range(len(all) - 1):
        if(all[i + 1] == all[i]):
            number = number + 1
        else:
            if number >= cutoff:
                vocab.append(all[i])
            number = 1
    vocab.sort()
    return vocab

#Calculates the prior probability of the trainingdata for a year
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    # TODO: add your code here
    count = len(training_data) + 2
    yLabel = 0
    for p in label_list:
        for i in range(len(training_data)):
            dic = training_data[i]
            if(dic['label'] == p):
                yLabel = yLabel + 1
        prob = (yLabel + smooth) / count
        logprob[p] = math.log(prob)
        yLabel = 0
    return logprob

#Calculates and returns the probability of a word given the training data and a label
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """
    
    smooth = 1 # smoothing factor
    word_prob = {}
    # TODO: add your code here
    totalCount = {}
    types = len(vocab)
    totalWords = 0
    for p in range(len(training_data)):
        dic = training_data[p]
        if(dic['label'] == label):
            words = dic['bow']
            for k in words.keys():
                totalWords = totalWords + words[k]
            totalCount = Counter(totalCount) + Counter(words)
    for key in totalCount.keys():
        word_prob[key] = float(math.log(totalCount[key] + smooth) - math.log(totalWords + (smooth * (types + 1))))
    word_prob[None] = float(math.log(totalCount[None] + smooth) - math.log(totalWords + (smooth * (types + 1))))
    for w in vocab:
        if w not in word_prob.keys():
            word_prob[w] = float(math.log(1) - math.log(totalWords + smooth * (types + 1)))
    return word_prob

    
#Returns a model which shows data of the training_directory
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    retval['vocabulary'] = vocab
    retval['log prior'] = prior(training_data, ['2020', '2016'])
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')
    return retval


#Classifies whether it came from 2016 or 2020, using the previous methods
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>, 
             'log p(y=2016|x)': <log probability of 2016 label for the document>, 
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    pDic = model['log prior']
    vocab = model['vocabulary']
    conditionalsSix = model['log p(w|y=2016)']
    conditionalsTwenty = model['log p(w|y=2020)']
    sixteen = pDic['2016']
    twenty = pDic['2020']
    file = open(filepath)
    bow = create_bow(model['vocabulary'], filepath)
    for key, val in conditionalsSix.items():
        if key in bow:
            sixteen = sixteen + (bow[key] * val)
    for key, val in conditionalsTwenty.items():
        if key in bow:
            twenty  = twenty + (bow[key] * val)
    retval['log p(y=2020|x)'] = twenty
    retval['log p(y=2016|x)'] = sixteen
    if(twenty > sixteen):
        retval['predicted y'] = '2020'
    else:
        retval['predicted y'] = '2016'
    file.close()
    return(retval)
    

model = train('./corpus/training/', 2)
print(classify(model, './corpus/test/2016/0.txt'))
