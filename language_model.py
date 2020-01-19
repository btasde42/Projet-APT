from collections import Counter, defaultdict
from itertools import islice
import sys
import time
import numpy as np
import pickle
import json

MINPROBA =  1e-18

def proba_n_gram(l,n,MINPROBA=1e-18):
    """
    Cette fonction renvoie des probablites pour n-grams avec smoothing
    Renvoie une dictionnaire
    """
    ngrams=[]
    
    dict_model= Counter()
    all_words=Counter()
    for p in l:
        ngrams=list(zip(*[p.split()[i:] for i in range(n)]))
        print(ngrams)
        for i in ngrams:
            dict_model[i]+=1
            all_words[i[0]]+=1

    print(dict_model)

    for key, value in dict_model.items():
        dict_model[key]=(value+MINPROBA)/(all_words[key[0]]+MINPROBA)

    return dict_model


def calculate_proba_sentence(lang_model,p,n):

    ngrams_p = list(zip(*[p.split()[i:] for i in range(n)]))
    produit=1
    for (a,b) in ngrams_p:
        if (a,b) in lang_model:
            produit*=lang_model[(a,b)]
        else:
            produit*=produit #on evite l'abscence du ngram dans la modèle
        
    if len(ngrams_p)!=0:
        return produit/len(ngrams_p)
    else:
        return MINPROBA

def direct(corpus,lang):
    """la fonction qui renvoie la liste des phrases issues de la traduction directe d'une certaine langue"""
    return [elt for elt in corpus if elt["src_lang"] == elt["orig_lang"] and elt["src_lang"]== lang]

def indirect(corpus,lang):
    """la fonction qui renvoie la liste des phrases issues de la traduction INdirecte d'une certaine langue"""
    return [elt for elt in corpus if elt["src_lang"] != elt["orig_lang"] and elt["src_lang"]== lang]



def main1():

    #CREATION ET EXTRACTION DES MODELS DE LANGUES EN ET DE

    #corpus_eng=open('corpus.tc.en').readlines()
    #corpus_de=open('corpus.tc.de').readlines()

    json_file='da_newstest2016.json'
    corpus = json.load(open(json_file))
    #dict_en=proba_n_gram(corpus_eng,2)

    #pickle_out_en = open('en_lang_mod.pickle','wb')
    #pickle.dump(dict_en, pickle_out_en)
    #pickle_out_en.close()

    #dict_de=proba_n_gram(corpus_de,2)
    #pickle_out_de = open('de_lang_mod.pickle','wb')
    #pickle.dump(dict_de, pickle_out_de)
    #pickle_out_de.close()

    with open('de_lang_mod.pickle', 'rb') as handle:
        lang_mod_de = pickle.load(handle)
    #print(lang_mod_eng)
    #lang_mod_de='de_lang_mod.pickle'

    phrases_de_direct=direct(corpus,'de')
    phrases_de_indirect=indirect(corpus,'de')
    
    val_de_direct=[]
    val_de_indirect=[]
    for i in phrases_de_direct:
        calculate_proba_sentence(lang_mod_de,i['src'],2)
        val_de_direct.append(calculate_proba_sentence(lang_mod_de,i['src'],2))
            
    for i in phrases_de_indirect:
        calculate_proba_sentence(lang_mod_de,i['src'],2)
        val_de_indirect.append(calculate_proba_sentence(lang_mod_de,i['src'],2))
            
    print(sum(val_de_direct)/len(val_de_direct))
    print(sum(val_de_indirect)/len(val_de_indirect))

if __name__ == '__main__':

    main1()
