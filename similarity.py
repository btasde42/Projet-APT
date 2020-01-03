import spacy
from spacy import displacy
from spacy.lang.fi import Finnish
from spacy.lang.de import German
from spacy.lang.cs import Czech
from spacy.lang.tr import Turkish
from spacy.lang.ro import Romanian
from spacy.lang.ru import Russian
from spacy.lang.fi import Finnish
from spacy.lang.en import English
import json
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
from collections import defaultdict


def get_directions (corpus):
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
	en_en_ru = [elt for elt in corpus if elt["src_lang"] == "en" and elt["orig_lang"] == "en" and elt["tgt_lang"] == "ru"]
	en_ru_ru = [elt for elt in corpus if elt["src_lang"] == "en" and elt["orig_lang"] == "ru" and elt["tgt_lang"] == "ru"]
	return [de_de_en, de_en_en, cs_cs_en, cs_en_en, tr_tr_en, tr_en_en, ro_ro_en, ro_en_en, ru_ru_en, ru_en_en, fi_fi_en, fi_en_en, en_en_ru, en_ru_ru]

def compute_lexical_richness (direction, nlp):
	"""la fonction qui calcule la richesse lexicale de la phrase;
	retourne le nombre de tokens uniques/nombre de tokens total"""
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

def pos_trigrams (phrase, nlp): 
	"""la fonction qui renvoie la liste de trigrams de PoS à partir d'une phrase"""
	phrase = phrase.lower()
	doc = nlp(phrase)
	pos = [token.pos_ for token in doc]
	ngrams = zip(*[pos[i:] for i in range(3)])
	return [" ".join(ngram) for ngram in ngrams]

def pos_trigrams_direction(direction, nlp):

	all_trigrams = []
	for phrase in direction:
		phrase = phrase["src"]
		pos = pos_trigrams(phrase, nlp)
		for p in pos:
			if p not in all_trigrams:
				all_trigrams.append(p)
	return all_trigrams

corpus = json.load(open("da_newstest2016.json"))

directions = get_directions(corpus)

pos_trigrams_indirect = pos_trigrams_direction(directions[12], spacy.load("en_core_web_sm"))
pos_trigrams_direct = pos_trigrams_direction(directions[11], spacy.load("en_core_web_sm"))
print("LEN INDIRECT : ", len(pos_trigrams_indirect))
print("LEN DIRECT : ", len(pos_trigrams_direct))
difference = 0
for elt in pos_trigrams_indirect:
	if elt not in pos_trigrams_direct:
		difference+=1
print(difference)

#COMPARER LEXICAL RICHNESS
#print("DE direct : ", compute_lexical_richness(directions[0], German()))
#print("DE indirect : ", compute_lexical_richness(directions[1], German()))

#print("CS direct : ", compute_lexical_richness(directions[2], Czech()))
#print("Cs indirect : ", compute_lexical_richness(directions[3], Czech()))

#print("TR direct : ", compute_lexical_richness(directions[4], Turkish()))
#print("TR indirect : ", compute_lexical_richness(directions[5], Turkish()))

#print("RO direct : ", compute_lexical_richness(directions[6], Romanian()))
#print("RO indirect : ", compute_lexical_richness(directions[7], Romanian()))

#print("RU direct : ", compute_lexical_richness(directions[8], Russian()))
#print("RU indirect : ", compute_lexical_richness(directions[9], Russian()))

#print("FI direct : ", compute_lexical_richness(directions[10], Finnish()))
#print("FI indirect : ", compute_lexical_richness(directions[11], Finnish()))

#print("EN direct : ", compute_lexical_richness(directions[12], English()))
#print("EN indirect : ", compute_lexical_richness(directions[13], English()))