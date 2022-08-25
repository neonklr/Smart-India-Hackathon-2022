import pybktree
import json 

from . import constants 

def levenshtein(s, t):  # distance func 
    m, n = len(s), len(t)
    d = [range(n+1)]  
    d += [[i] for i in range(1,m+1)]
    for i in range(0,m):
        for j in range(0,n):
            cost = 1
            if s[i] == t[j]: cost = 0

            d[i+1].append( min(d[i][j+1]+1, # deletion
                               d[i+1][j]+1, # insertion
                               d[i][j]+cost) #substitution
                           )
    return d[m][n]

def get_data():
    # with open("test.json") as file:
    with open(constants.SANSKRIT_DICT_PATH) as file:
        data = json.load(file) 

    data = set(data.keys())
    return data  

def _autocorrect_word(word, MAX_WORDS = constants.MAX_WORDS_TO_PREDICT):
    word = word.strip() 
    predictions = sorted(tree.find(word, constants.MAX_ERROR))[:MAX_WORDS]
    # predictions is like:  [ (distance_1: int, word_1: str), (distance_2: int, word_2: str) .. ]   ..can also be empty 
    
    if predictions:
        return predictions[0][1]   # nearest valid word
    return word 

def autocorrect(text):
    return ' '.join(map(_autocorrect_word, text.split()))


data = get_data()
tree = pybktree.BKTree(levenshtein, data) 
