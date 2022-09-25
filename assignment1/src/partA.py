import nltk
from nltk.book import *
from nltk.corpus import brown
from urllib import request
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import itertools
import pandas as pd


# Exercise 1.A
genres = ['news','religion','government','fiction','romance']
pronouns = ['he','she','her','him']
cfd = nltk.ConditionalFreqDist((genre,word)for genre in genres
                               for word in brown.words(categories=genre))
cfd.tabulate(samples=pronouns)

# 1.C
genders = ['Male', 'Female']
cfd = nltk.ConditionalFreqDist()
for word in brown.words():
    word = word.lower()
    if word == 'he' or word == 'him':
        cfd[genders[0]][word] += 1
    if word == 'she' or word == 'her':
        cfd[genders[1]][word] += 1
cfd.tabulate(samples = pronouns)

def summationn(gender):
    sum_frequency = 0
    for i in cfd[gender]:
        sum_frequency += cfd[gender][i]
    return sum_frequency

male_relative_frequency = cfd['Male']['him']/(summationn('Male'))
female_relative_frequency = cfd['Female']['her']/(summationn('Female'))

print(f'Relative frequency of him: {male_relative_frequency:3.5f}')
print(f'Relative frequency of her: {female_relative_frequency:3.5f}')



#1.D

tag_brown = [x for x in brown.tagged_words(tagset='universal')]

PP_pron = ['she','he','her','him']
POSS_pron = ['her','his','hers']
Tot_pron = set(PP_pron + POSS_pron )

cfd_new = nltk.ConditionalFreqDist()

def new_CFD(pronLst, pronType):
    for i in range(len(tag_brown)):
        word = tag_brown[i][0].lower()
        if word in pronLst and tag_brown[i][1] == pronType:
            cfd_new[tag_brown[i][1]][word] += 1
new_CFD(PP_pron,'PRON')
new_CFD(POSS_pron,'DET')

cfd_new.tabulate(samples = Tot_pron)

#1.E
def summation(typPron,lstPron):
    sum_frequency = 0
    for word in lstPron:
        sum_frequency += cfd_new[typPron][word]
    return sum_frequency


#'d_her' stands for determiner her, 
#'p_her' stands for pronoun her,
#'n_she' stands for noun she.
d_her = cfd_new['DET']['her']/summation('DET',POSS_pron) 
p_her = cfd_new['PRON']['her']/summation('DET',PP_pron)
n_she = cfd_new['PRON']['she']/summation('PRON',PP_pron)


d_him = cfd_new['DET']['him']/summation('DET',POSS_pron) 
n_he = cfd_new['PRON']['he']/summation('PRON',PP_pron)

print(f'Relative frequency of him as possessive pronoun: {d_him:3.5f}')
print(f'Relative frequency of her as possessive pronoun: {d_her:3.5f}')
print(f'Relative frequency of her as objective pronoun: {p_her:3.5f}')

#1.F

xticks = cfd_new.keys()

x = np.arange(0,len(cfd_new.keys()))
h = np.zeros(len(cfd_new.keys()))

for i, p in enumerate(Tot_pron):
    y = []
    for j, genre in enumerate(['DET','PRON']):
        y.append(cfd_new[genre][p])
    plt.bar(x,y, label = p, bottom = h)
    h += y
plt.xticks(np.arange(0, len(xticks)), xticks)
plt.legend()
plt.show()

#2.A
url = "http://www.gutenberg.org/files/74/74-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf-8')

# We will split the data to a list by following command.
raw_data_list = raw.split()

#2.B

# Marking where start and end will be so that 
# we'll slice later in raw_data_list.

Remove_start = [i for i in range(len(raw_data_list)) if raw_data_list[i]== "1876."]
Remove_end = [i for i in range(len(raw_data_list)) if raw_data_list[i]=="***"]

#Slicing so that we exclude preamble,appendix and copyrights. 
raw_data_list = raw_data_list[Remove_start[0]+1:Remove_end[2]]

#2.C

#Initiate nltk for all tokens.
tokens = nltk.word_tokenize(raw)

#Removing all dots.
tokens = [token.replace('.','') for token in tokens]

#Removing all underscores
tokens = [token.replace('_','')for token in tokens]

#Remove all punctuations
Pat_punc = re.compile('^\&$-+')
tokens = [token for token in tokens if not Pat_punc.match(token)]

#Removing empty lines
tokens = [token for token in tokens if token != '']
tokens = [token.lower() for token in tokens]

#2.D

Frq_dist = nltk.FreqDist(tokens)
twenty_most_freq_words = Frq_dist.most_common(21)

for i, (word, freq) in enumerate(twenty_most_freq_words):
    print(f'Pos: {i} , Token: " {word} " , Freq: {freq} ')
    

#2.E

Lst = []
print(len(set(Frq_dist.values())))
Count = Counter(Frq_dist.values())
for i in range(1,11):
    Lst.append([i,Count[i]])
    
#11-50 times occurence 
occur_11_50 = [key for key in Count.keys() if 11 <= key <= 50]
sum_11_50 = sum(Count[key] for key in occur_11_50)
Lst.append(['11-50',sum_11_50])

#50-100 times occurence 
occur_50_100 = [key for key in Count.keys() if 50 <= key <= 100]
sum_50_100 = sum(Count[key] for key in occur_50_100)
Lst.append(['50-100',sum_50_100])

# 100+ times occurence 
occur_100more = [key for key in Count.keys() if key > 100]
sum_100more = sum(Count[key] for key in occur_100more)
Lst.append(['100+',sum_100more])

for freq, words in Lst:
    print(f'Frequency: {freq}, number of words: {words}')


#2.F
for index, (word,freq) in enumerate(twenty_most_freq_words, 1):
    print(f'Rank: {index}, r*n: {(index*freq)}')


#2.G

x = list(range(1,len(Frq_dist)+1))
y= list(Frq_dist.values())
y.sort(reverse=True)
plt.plot(x,y)
plt.title('Ziph law')
plt.xlabel('Frequency')
plt.ylabel('Rank')
plt.show()

x1 = np.log(x)
y1 = np.log(y)
plt.plot(x1,y1)
plt.title('Ziph law')
plt.xlabel('Log[Frequency]')
plt.ylabel('Log[Rank]')

plt.show()

#Exercise