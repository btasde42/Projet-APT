from collections import Counter, defaultdict
from itertools import islice
import sys
import time
import numpy as np

def compute_ngrams (sentence, n): # OK
	"""la fonction qui renvoie la liste de ngrams à partir d'une phrase"""
	sentence = sentence.lower()
	ngrams = zip(*[sentence.split()[i:] for i in range(n)])
	return list(ngrams)


def proba_n_gram(l,n):
    """
    Cette fonction renvoie des probablites pour n-grams
    Renvoie une dictionnaire
    """

    dict_model= defaultdict(lambda: Counter()) #on crée une dict de dict, le couche interieur est un counter
    for i in compute_ngrams(l,n):
    	dict_model[i[:n-1]][i[-1]]+=1
    
    for key, value in dict_model.items():
    	somme=float(sum(value.values()))
    	for w3 in dict_model[key]:
    		dict_model[key][w3] /= somme

    return dict_model

def main():
	#corpus_eng=open('corpus.tc.en').read()
	#corpus_de=open('corpus.tc.de').read()
	test='Feronikel was privatised five years ago , and is still in business , but operates amid concerns for workers &quot; safety . &#91; Reuters &#93; on paper at least , it looks like a great idea still business in for workers &quot;'
	print(proba_n_gram(test,3))

if __name__ == '__main__':

    main()