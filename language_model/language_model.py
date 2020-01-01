from collections import Counter, defaultdict
from itertools import islice
import sys
import time
import numpy as np

def compute_ngrams (sentence, n): # OK
	"""la fonction qui renvoie la liste de ngrams à partir d'une phrase"""
	sentence = sentence.lower()
	ngrams = zip(*[sentence.split()[i:] for i in range(n)])
	return [" ".join(ngram) for ngram in ngrams]


def proba_n_gram(l):
    """
    Cette fonction renvoie des probablites pour n-grams
    Renvoie une dictionnaire
    """

    dict_model= {Counter} #on crée une dict de dict, le couche interieur est un counter
    for i in compute_ngrams(l,3): #on se base par exemple sur les trigrams
    	t_elt=tuple(i.split(' '))
    	print(t_elt)
    	dict_model[t_elt]+=1

    for k,v in dict_model.items():
    	print(v.values())
    	total_count=float(sum(v.values()))
    	for v in dict_model[k]:
    		dict_model[k][v] /= total_count
    return dict_model

def main():
	#corpus_eng=open('corpus.tc.en').read()
	#corpus_de=open('corpus.tc.de').read()
	test='Feronikel was privatised five years ago , and is still in business , but operates amid concerns for workers &quot; safety . &#91; Reuters &#93; on paper at least , it looks like a great idea still business in for workers &quot;'
	print(proba_n_gram(test))

if __name__ == '__main__':

    main()