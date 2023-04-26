# -*- coding: utf-8 -*-
"""
textsummarizationRouge.py
There is such a bug that sometimes the summary of this program output may be an empty set
Sometimes,the commond ic may generate errors in this program, and you shold use print instead.
You should have the following refernec about extractive summarization.
Refernec 1, https://derwen.ai/docs/ptr/explain_summ/
Created on Wed Apr  5 16:16:47 2023
run abut 20 minutes on Apr.11
@author: Thinkbook15p
"""
from math import sqrt
import pytextrank
from icecream import ic
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from rouge import Rouge
import pandas as pd
from string import punctuation
from operator import itemgetter
from spacy.lang.en import English
import matplotlib.pyplot as plt
import numpy as np
from string import punctuation
from operator import itemgetter
import warnings
warnings.filterwarnings("ignore")

stopwords = list(STOP_WORDS)
punctuation = punctuation + '\n'

training_data = pd.read_csv('train.csv',dtype={'user_id': int})
validation_data = pd.read_csv('validation.csv',dtype={'user_id': int})
test_data = pd.read_csv('test.csv',dtype={'user_id': int})

X_train0 = training_data['article']
y_train0 = training_data['highlights']
X_test0 = test_data['article']
y_test0 = test_data['highlights']
length_train0 = int(len(X_train0) * 0.001) #if the percent is 0.01, take 10 minutes or more
length_test0 = int(len(X_test0) * 0.01)

X_train = X_train0[0:length_train0 - 1]
y_train = y_train0[0:length_train0 - 1]
X_test = X_test0[0:length_test0 - 1]
y_test = y_test0[0:length_test0 - 1]

# nlp = English()
# nlp.add_pipe('sentencizer')


nlp = spacy.load('en_core_web_sm')
# ner = nlp.get_pipe("ner")
# n_iter = 2
nlp.pipe_names
# nlp = spacy.load('en_core_web_sm')
# training data

# add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank")

# =============================================================================
# training data
# =============================================================================
print("===============training data======\n")
length = len(X_train)  # used as counter
rouge1_train = np.empty([length, 3])
for i in range(length):

    print("i: \n", i)

    text = X_train[i]
    # print(len(text))

    doc = nlp(text)


    for p in doc._.phrases:
        i1 = i

    sent_bounds = [[s.start, s.end, set([])] for s in doc.sents]
    # sent_bounds

    limit_phrases = 4

    phrase_id = 0
    unit_vector = []

    for p in doc._.phrases:

        unit_vector.append(p.rank)

        for chunk in p.chunks:

            for sent_start, sent_end, sent_vector in sent_bounds:
                if chunk.start >= sent_start and chunk.end <= sent_end:
                    sent_vector.add(phrase_id)
                    break

        phrase_id += 1

        if phrase_id == limit_phrases:
            break


    sum_ranks = sum(unit_vector)

    unit_vector = [rank/sum_ranks for rank in unit_vector]
    print(unit_vector)

    sent_rank = {}
    sent_id = 0

    for sent_start, sent_end, sent_vector in sent_bounds:
        sum_sq = 0.0
        # ic
        for phrase_id in range(len(unit_vector)):
            # ic(phrase_id, unit_vector[phrase_id])

            if phrase_id not in sent_vector:
                sum_sq += unit_vector[phrase_id]**2.0

        sent_rank[sent_id] = sqrt(sum_sq)
        sent_id += 1

    # from operator import itemgetter

    sorted(sent_rank.items(), key=itemgetter(1))

    limit_sentences = 2

    sent_text = {}
    sent_id = 0

    for sent in doc.sents:
        sent_text[sent_id] = sent.text
        sent_id += 1

    num_sent = 0

    for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):

        if len(sent_text[sent_id]):  # if sent_text[sent_id] is not null set
            print(sent_text[sent_id])
            # ic(sent_id, sent_text[sent_id])
            
        if num_sent == 0:
            sent_id_first=sent_id
            summary=sent_text[sent_id]
        # else:
            # summary = summary.join(sent_text[sent_id])
            
        num_sent += 1

        if num_sent == limit_sentences:
            break
        
    model_out = summary
    reference = y_train[i]
    rouge = Rouge()
    # if model_out:
    #     print(rouge.get_scores(model_out, reference))
    #     print(summary)
    if len(model_out):
        r_out = rouge.get_scores(model_out, reference)
        print(r_out)
        # buffer the rouge-1 parameters
        r_out1 = r_out[0]  # list to dict
        r_out2 = r_out1.get('rouge-1')
        r_out3 = [r_out2.get('r'), r_out2.get('p'), r_out2.get('f')]
        rouge1_train[i, :] = r_out3

        if sum(r_out3) < 1e-10:
            rouge1_train[i, :] = rouge1_train[i-1, :]  # r_out is null set

f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
ax1.hist(rouge1_train[:,0])
ax1.set_title('Train rouge-1,r')
ax2.hist(rouge1_train[:,1])
ax2.set_title('Train rouge-1,p')
ax3.hist(rouge1_train[:,2])
ax3.set_title('Train rouge-1,f')
plt.show()

# =============================================================================
# test data
# =============================================================================
print("===============test data======\n")
length = len(X_test)  # used as counter
rouge1_test = np.empty([length, 3])

for i in range(length):

    # print("i: \n", i)
    text = X_test[i]
    # text = X_train[i]
    # print(len(text))

    doc = nlp(text)

    # examine the top-ranked phrases in the document
    # for phrase in doc._.phrases:
    #     print(phrase.text)
    #     print(phrase.rank, phrase.count)
    #     print(phrase.chunks)

    # from icecream import ic

    for p in doc._.phrases:
        i1 = i
        # ic(p.rank, p.count, p.text)
        # ic(p.chunks)

    sent_bounds = [[s.start, s.end, set([])] for s in doc.sents]
    # sent_bounds

    limit_phrases = 4

    phrase_id = 0
    unit_vector = []

    for p in doc._.phrases:
        # ic(phrase_id, p.text, p.rank)

        unit_vector.append(p.rank)

        for chunk in p.chunks:
            # ic(chunk.start, chunk.end)

            for sent_start, sent_end, sent_vector in sent_bounds:
                if chunk.start >= sent_start and chunk.end <= sent_end:
                    # ic(sent_start, chunk.start, chunk.end, sent_end)
                    sent_vector.add(phrase_id)
                    break

        phrase_id += 1

        if phrase_id == limit_phrases:
            break

    # for sent in doc.sents:
    #     ic(sent)

    sum_ranks = sum(unit_vector)

    unit_vector = [rank/sum_ranks for rank in unit_vector]
    print(unit_vector)

    sent_rank = {}
    sent_id = 0

    for sent_start, sent_end, sent_vector in sent_bounds:
        # ic(sent_vector)
        sum_sq = 0.0
        # ic
        for phrase_id in range(len(unit_vector)):
            # ic(phrase_id, unit_vector[phrase_id])

            if phrase_id not in sent_vector:
                sum_sq += unit_vector[phrase_id]**2.0

        sent_rank[sent_id] = sqrt(sum_sq)
        sent_id += 1

    # from operator import itemgetter

    sorted(sent_rank.items(), key=itemgetter(1))

    limit_sentences = 2

    sent_text = {}
    sent_id = 0

    for sent in doc.sents:
        sent_text[sent_id] = sent.text
        sent_id += 1

    num_sent = 0

    for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):

        if len(sent_text[sent_id]):  # if sent_text[sent_id] is not null set
            print(sent_text[sent_id])
            # ic(sent_id, sent_text[sent_id])
        # Ordering top n sentences in origi order
        if num_sent == 0:
            sent_id_first=sent_id
            summary=sent_text[sent_id]
        # else:
        #     summary = summary.join(sent_text[sent_id])

        num_sent += 1

        if num_sent == limit_sentences:
            break

    model_out = summary
    # model_out = sent_text[sent_id_first]  # Here there is a question, how to select sent_id
    # model_out = sent_text[sent_id]  #
    reference = y_test[i]
    # reference = y_train[i]
    rouge = Rouge()
    if model_out:
        print(rouge.get_scores(model_out, reference))

    reference = y_test[i]
    rouge = Rouge()
    if len(model_out):
        r_out = rouge.get_scores(model_out, reference)
        print(r_out)
        # buffer the rouge-1 parameters
        r_out1 = r_out[0]  # list to dict
        r_out2 = r_out1.get('rouge-1')
        r_out3 = [r_out2.get('r'), r_out2.get('p'), r_out2.get('f')]
        rouge1_test[i, :] = r_out3

        if sum(r_out3) < 1e-10:
            rouge1_test[i, :] = rouge1_test[i-1, :]  # r_out is null set

#
# Creates two subplots and unpacks the output array immediately

f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
ax1.hist(rouge1_test[:,0])
ax1.set_title('Test rouge-1,r')
ax2.hist(rouge1_train[:,1])
ax2.set_title('Test rouge-1,p')
ax3.hist(rouge1_train[:,2])
ax3.set_title('Test rouge-1,f')
plt.show()