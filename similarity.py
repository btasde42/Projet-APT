import spacy
from spacy import displacy
from spacy.lang.fi import Finnish
from spacy.lang.de import German
from spacy.lang.cs import Czech
from spacy.lang.tr import Turkish
from spacy.lang.ro import Romanian
from spacy.lang.ru import Russian

import json
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
from collections import defaultdict, Counter, OrderedDict
from itertools import chain
import pandas as pd


def get_directions (corpus):
	"""args : le corpus entier
	return : liste des directions (une direction = une liste de dictionnaires, dont les clés sont : 
	src, hyp, réf, score)
	"""
	de_de_en = [elt for elt in corpus if elt["src_lang"] == "de" and elt["orig_lang"] == "de" and elt["tgt_lang"] == "en"]
	de_en_en = [elt for elt in corpus if elt["src_lang"] == "de" and elt["orig_lang"] == "en" and elt["tgt_lang"] == "en"]
	cs_cs_en = [elt for elt in corpus if elt["src_lang"] == "cs" and elt["orig_lang"] == "cs" and elt["tgt_lang"] == "en"]
	cs_en_en = [elt for elt in corpus if elt["src_lang"] == "cs" and elt["orig_lang"] == "en" and elt["tgt_lang"] == "en"]
	tr_en_en = [elt for elt in corpus if elt["src_lang"] == "tr" and elt["orig_lang"] == "en" and elt["tgt_lang"] == "en"]
	tr_tr_en = [elt for elt in corpus if elt["src_lang"] == "tr" and elt["orig_lang"] == "tr" and elt["tgt_lang"] == "en"]
	ro_ro_en = [elt for elt in corpus if elt["src_lang"] == "ro" and elt["orig_lang"] == "ro" and elt["tgt_lang"] == "en"]
	ro_en_en = [elt for elt in corpus if elt["src_lang"] == "ro" and elt["orig_lang"] == "en" and elt["tgt_lang"] == "en"]
	ru_en_en = [elt for elt in corpus if elt["src_lang"] == "ru" and elt["orig_lang"] == "en" and elt["tgt_lang"] == "en"]
	ru_ru_en = [elt for elt in corpus if elt["src_lang"] == "ru" and elt["orig_lang"] == "ru" and elt["tgt_lang"] == "en"]
	fi_fi_en = [elt for elt in corpus if elt["src_lang"] == "fi" and elt["orig_lang"] == "fi" and elt["tgt_lang"] == "en"]
	fi_en_en = [elt for elt in corpus if elt["src_lang"] == "fi" and elt["orig_lang"] == "en" and elt["tgt_lang"] == "en"]
	en =[elt for elt in corpus if elt["src_lang"] == "en" and elt["orig_lang"] == "en" and elt["tgt_lang"] == "ru"]
	en_en_ru = []
	[en_en_ru.append(elt) for elt in en if elt not in en_en_ru]
	print(len(en_en_ru))
	#en_en_ru = [elt for elt in corpus if elt["src_lang"] == "en" and elt["orig_lang"] == "en" and elt["tgt_lang"] == "ru"]
	en_ru_ru = [elt for elt in corpus if elt["src_lang"] == "en" and elt["orig_lang"] == "ru" and elt["tgt_lang"] == "ru"]
	return [de_de_en, de_en_en, cs_cs_en, cs_en_en, tr_tr_en, tr_en_en, ro_ro_en, ro_en_en, ru_ru_en, ru_en_en, fi_fi_en, fi_en_en, en_en_ru, en_ru_ru]

def directions_to_graph(directions):
	lengths = [len(direction) for direction in directions]
	lengths_direct = [lengths[i] for i in range(len(lengths)) if i%2==0]
	lengths_indirect = [lengths[i] for i in range(len(lengths)) if i%2!=0]
	labels = ['de', 'cs', 'tr', 'ro', 'ru', 'fi', 'en']
	df = pd.DataFrame({'direct' :lengths_direct, 'indirect' : lengths_indirect}, index = labels) 
	ax = df.plot.bar(rot=0)
	plt.legend()
	plt.show()
	
def compare_length(directions):
	"""la fonction qui compare la longueur des phrases sources issues de la traduction et écrites par un locuteru natif
	args : directions
	return : None (affiche des résultats)
	"""
	length = []
	for direction in directions:
		length = [len(phrase['src'].split()) for phrase in direction]
		print(mean(length))

def compare_length_to_graph(directions):
	"""la fonction qui compare la longueur des phrases sources issues de la traduction et écrites par un locuteru natif
	args : directions
	return : None (affiche des résultats)
	"""
	lengths = [mean([len(phrase['src'].split()) for phrase in direction]) for direction in directions]
	lengths_direct = [lengths[i] for i in range(len(lengths)) if i%2==0]
	lengths_indirect = [lengths[i] for i in range(len(lengths)) if i%2!=0]
	labels = ['de', 'cs', 'tr', 'ro', 'ru', 'fi', 'en']
	df = pd.DataFrame({'direct' :lengths_direct, 'indirect' : lengths_indirect}, index = labels) 
	ax = df.plot.bar(rot=0)
	plt.legend()
	plt.show()

def compare_heights(direction_direct, direction_indirect, nlp):
	"""la fonction qui compare la profondeur des arbres de dépendance des phrases sources issues de la traduction et écrites par un locuteur natif
	args: directions directe(dictionnaire), direction indirecte(dictionnaire), modèle de langue de Spacy
	return : 2 valeurs : moyenne des profondeurs des arbres de la direction directe et indirecte (float)
	"""
	heights_direct = []
	heights_indirect = []
	for phrase in direction_direct:
		doc = nlp(phrase["src"])
		root = [token for token in doc if token.head == token][0]
		heights_direct.append(get_height(root))
	for phrase in direction_indirect:
		doc = nlp(phrase["src"])
		root = [token for token in doc if token.head == token][0]
		heights_indirect.append(get_height(root))
	print("DIRECT : ", mean(heights_direct))
	print("INDIRECT : ", mean(heights_indirect))
	return mean(heights_direct), mean(heights_indirect)

def get_height(root):
    """la fonction qui retourne la profondeur de l'arbre de dépendance
    args : root(de l'arbre de Spacy)
    return : la profondeur (int)
    """
    if list(root.children):
    	return 1 + max(get_height(x) for x in root.children)
    return 1

def compute_lexical_richness (direction, nlp):
	"""la fonction qui calcule la richesse lexicale de la phrase;
	retourne le nombre de tokens uniques/nombre de tokens tota
	args : direction (dictionnaire), le modèle de langue de Spacy (ex. German())
	return : int """
	lemmas = defaultdict(int)
	tokens = defaultdict(int)
	all_tokens = defaultdict(int)
	for phrase in direction:
		doc = nlp(phrase["src"])
		for token in doc:
			lemmas[token.lemma_] +=1
			tokens[token] +=1
			#print("LEMMAS : ", len(lemmas))
			#print("TOKENS : ", sum(tokens.values()))
		all_tokens["lemmas"]+=len(lemmas)
		all_tokens["tokens"]+=sum(tokens.values())
	lemmas = defaultdict(int)
	tokens = defaultdict(int)
	print(all_tokens)
	return all_tokens["lemmas"]/all_tokens["tokens"]

def lexical_richness_to_graph(directions):
	de_direct = compute_lexical_richness(directions[0], German())
	de_indirect = compute_lexical_richness(directions[1], German())
	cs_direct = compute_lexical_richness(directions[2], Czech())
	cs_indirect = compute_lexical_richness(directions[3], Czech())
	tr_direct = compute_lexical_richness(directions[4], Turkish())
	tr_indirect = compute_lexical_richness(directions[5], Turkish())
	ro_direct = compute_lexical_richness(directions[6], Romanian())
	ro_indirect = compute_lexical_richness(directions[7], Romanian())
	ru_direct = compute_lexical_richness(directions[8], Russian())
	ru_indirect = compute_lexical_richness(directions[9], Russian())
	fi_direct = compute_lexical_richness(directions[10], Finnish())
	fi_indirect = compute_lexical_richness(directions[11], Finnish())
	en_direct = compute_lexical_richness(directions[12], English())
	en_indirect = compute_lexical_richness(directions[13], English())
	direct = [de_direct, cs_direct, tr_direct, ro_direct, ru_direct, fi_direct, en_direct]
	indirect = [de_indirect, cs_indirect, tr_indirect, ro_indirect, ru_indirect, fi_indirect, en_indirect]
	labels = ['de', 'cs', 'tr', 'ro', 'ru', 'fi', 'en']
	df = pd.DataFrame({'direct' : direct, 'indirect' : indirect}, index = labels) 
	ax = df.plot.bar(rot=0)
	plt.title('richesse lexicale')
	plt.show()

def compare_lexical_richness(directions):
	"""la fonction qui met en évidence les différence de la richersse lexicale entre la direction directe et indirecte pour chaque langue;
	prend en argument la liste des directions obtenue avec la fonction get_directions(corpus)
	ne renvoie rien mais fait des prints"""

	print("DE direct : ", compute_lexical_richness(directions[0], German()))
	print("DE indirect : ", compute_lexical_richness(directions[1], German()))

	print("CS direct : ", compute_lexical_richness(directions[2], Czech()))
	print("Cs indirect : ", compute_lexical_richness(directions[3], Czech()))

	print("TR direct : ", compute_lexical_richness(directions[4], Turkish()))
	print("TR indirect : ", compute_lexical_richness(directions[5], Turkish()))

	print("RO direct : ", compute_lexical_richness(directions[6], Romanian()))
	print("RO indirect : ", compute_lexical_richness(directions[7], Romanian()))

	print("RU direct : ", compute_lexical_richness(directions[8], Russian()))
	print("RU indirect : ", compute_lexical_richness(directions[9], Russian()))

	print("FI direct : ", compute_lexical_richness(directions[10], Finnish()))
	print("FI indirect : ", compute_lexical_richness(directions[11], Finnish()))

	print("LEN English direct : ", len(directions[12]))
	print("EN direct : ", compute_lexical_richness(directions[12], English()))
	print("LEN English indirect : ", len(directions[13]))
	print("EN indirect : ", compute_lexical_richness(directions[13], English()))

def pos_trigrams(direction, nlp) : # CORRECT!
# NEW FUNCTION TO REPLACE ALL PREVIOUS FUNCTIONS WITH TRIGRAMS!
	""" la fonction qui renvoie les trigrammes des PoS avec leur fréquence
	args : direction, modèle de langue
	return : dictionnaire (clé : PoS trigramme (tuple), valeur : fréquence)
	"""
	all_trigrams = defaultdict(float)
	for phrase in direction:
		phrase = phrase["src"].lower()
		doc = nlp(phrase)
		pos = [token.pos_ for token in doc]
		for i in range(len(pos)-2):
			all_trigrams[(pos[i], pos[i+1], pos[i+2])] +=1
	for trigram in all_trigrams:
		all_trigrams[trigram] = all_trigrams[trigram]/sum(all_trigrams.values())
	return all_trigrams

def pos_position(direction, nlp): # CORRECT !
	"""la fonction qui analyse la fréquence de l'apparition des tokens aux positions initiales (1 ou 2 mot de la phrase) 
	ou finales (le dernier ou l'avant dernier mot dechaque position
	args : direction, modèle de langue (ex : spacy.load("en_core_web_sm") pour l'anglais), 
			occurrences : un dictionnaire (key : PoS, value : occurrence du PoS dans le corpus)
	retourne un OrderedDict pour une direction donnée
	key : position (1,2,penultimate,last)
		key : part of speech
		value : frequency
	"""
	positions = defaultdict(lambda : defaultdict(float))
	for phrase in direction:
		phrase = phrase["src"].lower()
		doc = nlp(phrase)
		for token in doc:
			if token.i == 0:
				positions["1"][token.pos_] +=1
			if token.i == 1:
				positions["2"][token.pos_] +=1
			if token.i == len(phrase.split())-1:
				positions["last"][token.pos_] +=1
			if token.i == len(phrase.split())-2:
				positions["penultimate"][token.pos_] +=1
	for key, value in positions.items():
		for elt in value:
			positions[key][elt] = positions[key][elt]/len(direction)
		positions[key] = list(OrderedDict(sorted(value.items(), key=lambda x: float(x[1]), reverse = True)).items())[:3] # y a-t-il moyen de faire plus simple?
	# NOUVEAU AJOUT
	# MODIFICATION DES VALEURS
	new_positions = defaultdict(lambda : defaultdict(float))
	for key, value in positions.items():
		for i in range(len(value)):
			new_positions[key][positions[key][i][0]] = positions[key][i][1]
	print(new_positions)
	return new_positions

def positionnal_token_frequency(direction_directe, direction_indirecte, nlp): # CORRECT  ! 
	"""la fonction qui met en évidence la fréquence de l'apparition des tokens aux positions initiales (1 ou 2 mot de la phrase) 
	ou finales (le dernier ou l'avant dernier mot de la phrase) pour deux directions passées en arguments
	ne retourne rien, mais fait des prints
	"""
	pos_position(direction_directe, nlp)
	pos_position(direction_indirecte, nlp)

def dependent_head_relation_position_direction (direction ,nlp): # CORRECT
	"""args : direction, modèle de langue de spacy (ex. spacy.load("en_core_web_sm"))
	return : dictionnaire dans dictionnaire
	clé : tuple (PoS du head, PoS du dépendant, relation de dépendance)
		clé : 'right', 'left'
			valeur : fréquence
	"""
	all_dep = defaultdict(lambda : defaultdict(float))
	for phrase in direction:
		phrase = phrase["src"]
		phrase = phrase.lower()
		doc = nlp(phrase)
		for token in doc:
			if len(list(token.lefts)) != 0:
				for t in token.lefts:
					all_dep[(token.pos_, t.pos_, t.dep_)]["left"] +=1
			if len(list(token.rights)) != 0:
				for t in token.rights:
					all_dep[(token.pos_, t.pos_, t.dep_ )]["right"] +=1
	sum_dep = 0 # la somme de toutes les dépendances
	for key, value in all_dep.items():
		sum_dep += sum(all_dep[key].values())
	for key, value in all_dep.items():
		for elt in value:
			all_dep[key][elt] = all_dep[key][elt]/sum_dep
	return all_dep

def dict_to_graph(D):
	""" Transforms a normal key:value dict to tuple
	Pour la fonction pos_trigrams,"""
	plt.bar(range(len(D)), list(D.values()), align='center')
	plt.xticks(range(len(D)), list(D.keys()))
	plt.show()


def dict_of_dict_to_graph(dictt, name):
	"""Transfome une dictionnaire de dictionnaire en graph, l'axis X est le deuxième clé et axis y=valeir """
	pd.DataFrame(dictt).T.plot(kind='bar')
	plt.title(name)
	plt.show()

def main():
	corpus = json.load(open("da_newstest2016.json"))

	directions = get_directions(corpus)

	en_en = directions[12]
	en_ru = directions[13]
	de_de = directions[0]
	de_en = directions[1]

	nlp_en = spacy.load("en_core_web_sm")
	nlp_de = spacy.load("de_core_news_sm")

	#print("COMPARAISON DE LA PROFONDEUR DES ARBRES DE DÉPENDANCE")
	#compare_heights(de_de, de_en, nlp_de)
	#compare_heights(en_en, en_ru, nlp_en)

	#print("COMPARAISON DE LA RICHESSE LEXICALE : ")
	#compare_lexical_richness(directions)

	#print("COMPARAISON DES TRIGRAMMES DE POS : ")
	#print(pos_trigrams(en_en, nlp_en))
	#dict_to_graph(pos_trigram(de_de, de_en, nlp_de))
	#dict_of_dict_to_graph(dependent_head_relation_position_direction(de_de, nlp_de).keys[:10])
	#print("Positionnal token frequency")
	#positionnal_token_frequency(en_en, en_ru, nlp_en)
	#positionnal_token_frequency(de_de, de_en, nlp_de)
	#print(pos_position(de_de, nlp_de))
	dict_of_dict_to_graph(pos_position(de_de, nlp_de), "DE_DE positional frequency")
	dict_of_dict_to_graph(pos_position(de_en, nlp_de), "DE_EN positional frequency")
	dict_of_dict_to_graph(pos_position(en_en, nlp_en), "EN_EN positional frequency")
	dict_of_dict_to_graph(pos_position(en_ru, nlp_de), 'EN_RU positional frequency')
	
	
	#print(dependent_head_relation_position_direction(de_de, nlp_de))


	#print(pos_trigrams(phrase, spacy.load("en_core_web_sm")))


if __name__ == '__main__':
	main()
