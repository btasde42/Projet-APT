import spacy
import json
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean

def get_tree (phrase):
	nlp = spacy.load("en_core_web_sm")
	doc = nlp(phrase)
	root = [token for token in doc if token.head == token][0]
	print("ROOT = ", root)
	for token in doc:
		print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
			token.shape_, token.is_alpha, token.is_stop)
		#print("ROOT = ", token.head)

def get_height(root):
    """
    
    """
    if list(root.children):
    	return 1 + max(get_height(x) for x in root.children)
    return 1

#def get_root (phrase, nlp):
#	doc = nlp(phrase)
#	return [token for token in doc if token.head == token][0]

def get_heights_and_roots_for_direction (direction, nlp):
	"""la fonction qui renvoie 4 listes : 
	les profondeurs des arbres des hypothèses et des références, les racines des arbres des hypothèses et des références"""
	hyp = []
	ref = []
	roots_hyp = []
	roots_ref = []
	for phrase in direction:
		doc_hyp = nlp(phrase["hyp"])
		doc_ref = nlp(phrase["ref"])
		root_hyp = [token for token in doc_hyp if token.head == token][0]
		root_ref = [token for token in doc_ref if token.head == token][0]
		hyp.append(get_height(root_hyp))
		ref.append(get_height(root_ref))
		roots_hyp.append(root_hyp)
		roots_ref.append(root_ref)
	return hyp, ref, roots_hyp, roots_ref

def number_of_different_roots(direction, nlp):
	"""la fonction qui renvoie le % de têtes différentes entre l'hypothèse et la référence"""
	hyp, ref, roots_hyp, roots_ref = get_heights_and_roots_for_direction(direction, nlp)
	count = 0
	for i in range(len(roots_hyp)):
		if str(roots_ref[i]) != str(roots_hyp[i]):
			count+=1
	print("DIFFERENT ROOTS : ", count)
	print("TOTAL LENGTH : ", len(roots_hyp))
	return count/len(roots_hyp)

def mean_height (direction, nlp):
	"""la fonction qui retourne la profondeur moyenne des arbres de dépendance des hypothèses et des références pour une direction"""
	hyp_height, ref_height, roots_hyp, roots_ref = get_heights_and_roots_for_direction(direction, nlp)
	return mean(hyp_height), mean(ref_height)

def mean_difference_height (direction, nlp):
	""" la fonction qui renvoie l'écant moyen entre la profondeur de l'arbre de dépendance de l'hypothèse de celui de la référence
	pour chaque direction"""
	hyp_height, ref_height, roots_hyp, roots_ref = get_heights_and_roots_for_direction(direction, nlp)
	return mean([hyp_height[i]-ref_height[i] for i in range(len(hyp_height))])

def mean_difference_length (direction):
	"""la fonction qui calcule l'écart moyen entre les longueur des hypoyhèses et des références"""
	return mean([len(phrase['hyp'].split()) - len(phrase['ref'].split()) for phrase in direction])

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
	return [de_de_en, de_en_en, cs_cs_en, cs_en_en, tr_tr_en, tr_en_en, ro_ro_en, ro_en_en, ru_ru_en, ru_en_en, fi_fi_en, fi_en_en]

# CHARGEMENT DU CORPUS ET DES DIRECTIONS

corpus = json.load(open("da_newstest2016.json"))

nlp = spacy.load("en_core_web_sm")
directions = get_directions(corpus)

#############################################################
# ON CALCULE LA PROFONDEUR MOYENNE DES ARBRES DE DÉPENDANCE DE L'HYPOTHÈSE ET DE LA RÉFÉRENCE
#count = 0
#for direction in directions:
#	count+=1
#	if count%2 == 0:
#		print("INDIRECT :")
#	else: 
#		print("DIRECT : ")
#	hyp_mean, ref_mean = mean_height(direction, nlp)
#	print("HYP", hyp_mean, "MEAN", ref_mean)
############################################################

# ON CALCULE l'écant moyen entre la profondeur de l'arbre de dépendance 
# de l'hypothèse de celui de la référence pour chaque direction
count = 0
for direction in directions :
	count+=1
	if count%2 == 0:
		print("INDIRECT :")
	else: 
		print("DIRECT : ")
	print(mean_difference_length(direction))
############################################################

# ON CALCULE L'ÉCART MOYEN ENTRE LA LONGUEUR DES HYPOTHÈSES ET DES RÉFÉRENCES POUR TOUTES LES DIRECTIONS

############################################################
# ON CALCULE LE NOMBRE DE ROOTS DIFFERENTS DANS CHAQUE DIRECTION

#de_de_roots = number_of_different_roots(de_de_en, nlp)

#de_en_roots = number_of_different_roots(de_en_en, nlp)

#cs_cs_roots = number_of_different_roots(cs_cs_en, nlp)

#cs_en_roots = number_of_different_roots(cs_en_en, nlp)

#tr_tr_roots = number_of_different_roots(tr_tr_en, nlp)

#tr_en_roots = number_of_different_roots(tr_en_en, nlp)

#ro_ro_roots = number_of_different_roots(ro_ro_en, nlp)

#ro_en_roots = number_of_different_roots(ro_en_en, nlp)

#ru_ru_roots = number_of_different_roots(ru_ru_en, nlp)

#ru_en_roots = number_of_different_roots(ru_en_en, nlp)

#fi_fi_roots = number_of_different_roots(fi_fi_en, nlp)

#fi_en_roots = number_of_different_roots(fi_en_en, nlp)

#names = ['de_de', 'de_en', 'cs_cs', 'cs_en', 'tr_tr', 'tr_en', 'ro_ro', 'ro_en', 'ru_ru', 'ru_en', 'fi_fi', 'fi_en']
#values = [de_de_roots, de_en_roots, cs_cs_roots, cs_en_roots, tr_tr_roots, tr_en_roots, ro_ro_roots, ro_en_roots, 
#ru_ru_roots, ru_en_roots, fi_fi_roots, fi_en_roots]

#plt.figure(figsize=(9, 3))

#plt.subplot(131)
#plt.bar(names, values)
#plt.show()
#sns.distplot(hyp, label = "hyp")
#sns.distplot(ref, label = "ref")
#plt.legend()
#plt.show()


#get_tree(phrase)
#doc = nlp(phrase)
#root = [token for token in doc if token.head == token][0]
#print(get_height(root))