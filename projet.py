from collections import defaultdict, Counter
import json
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice
import math
import numpy as np
import pandas as pd
import statistics

def all_lang(corpus):
	"""la fonction qui calcule le nombre de langues utilisées dans le corpus"""
	return set([elt[key] for elt in corpus for key in elt.keys() if key == "src_lang" or key == "tgt_lang" or key == "orig_lang"])


def compute_directions (corpus):
	# la fonction qui calcule le nombre d'exemples pour chaque direction
	dico = defaultdict(int)
	for elt in corpus:
		dico[elt["src_lang"], elt["orig_lang"], elt["tgt_lang"]] +=1
	return dico

def direct(corpus):
	"""la fonction qui renvoie la liste des phrases issues de la traduction directe"""
	return [elt for elt in corpus if elt["src_lang"] == elt["orig_lang"]]

def indirect(corpus):
	"""la fonction qui renvoie la liste des phrases issues de la traduction INdirecte"""
	return [elt for elt in corpus if elt["src_lang"] != elt["orig_lang"]]

def score_eng(corpus):
	"""la fonction qui compare le score de traductions depuis et vers anglais"""
	dico_score=defaultdict(int)
	count_src=0
	count_tgt=0
	for elt in corpus:
		if elt["src_lang"]=="en":
			dico_score['src_lang']+=elt['score']
			count_src+=1
		if elt['tgt_lang']=='en':
			dico_score['tgt_lang']+=elt['score']
			count_tgt+=1	
	return (dico_score['src_lang']/count_src,dico_score['tgt_lang']/count_tgt) #on calcule le moyenne des scores pour voir à quel point ils sont differents

def distance_edition(ph1,ph2):
	"""la fonction qui calcule la distance d'édition entre deux phrases en forme d'un tableau """
	len_ph1=len(ph1)+1
	len_ph2=len(ph2)+1 #on prend les long des phrases pour créer le table
	table=np.zeros((len_ph1,len_ph2)) #on crée une table avec les zéros dedans

	for i in range(len_ph1): #on met tout d'abord indice pour chaque chr
		table[i,0]=i
	for j in range(len_ph2):
		table[0,j]=j

	for x in range(1,len_ph1): #on a deux boucles pour comparer deux phrases chr par chr
		for y in range(1,len_ph2):
			if ph1[x-1]==ph2[y-1]: #si les chr à meme indice sont les memes:
				table[x,y]=min(table[x-1,y]+1,table[x-1,y-1],table[x,y-1]+1)
			else:
				table[x,y]=min(table[x-1,y]+1,table[x-1,y-1]+1,table[x,y-1]+1)


	return(table[len(ph1)-1,len(ph2)-1])

def distance_edition_directions(corpus):
	dict_distance=defaultdict()
	directions=compute_directions(corpus)
	for i in directions.keys():
		for elt in corpus:
			if (elt["src_lang"], elt["orig_lang"], elt["tgt_lang"])==i:
				if i in dict_distance:
					dict_distance[i]+=distance_edition(elt['hyp'],elt['ref'])
				else:
					dict_distance[i]=distance_edition(elt['hyp'],elt['ref'])
		dict_distance[i]=dict_distance[i]/directions[i]
	return dict_distance

def length_score_impact(direction):
	"""la fonction qui fait un plot de la dépendance du score de la longueur de la phrase
	prend en arg une direction """
	length_score = [(len(elt["src"].split()), elt["score"]) for elt in direction]
	length, score = map(list, zip(*length_score))
	data = pd.DataFrame(0.0, columns=['X', 'Y'], index=np.arange(len(score)))
	data.X = length
	data.Y = score
	data['X_bins'] = pd.cut(data.X, bins = [0,10,20,30,40,50]) # intervals
	# for each bin, calculate the mean of Y
	result = data.groupby('X_bins')['Y'].mean()
	# do the plot
	result.plot()
	plt.show()

def compute_ngrams (sentence, n): # OK
	"""la fonction qui renvoie la liste de ngrams à partir d'une phrase"""
	sentence = sentence.lower()
	ngrams = zip(*[sentence.split()[i:] for i in range(n)])
	return [" ".join(ngram) for ngram in ngrams]

def compute_bleu_sentence (hypothesis, reference): 
	"""la fonction qui calcule le score bleu phrase par phrase
	BLEU-4
	weights = 0.25 pour chaque ngram"""
	clipped_precision = []
	BP = 1
	count = 0
	for i in range(1,5):
		hyp_ngrams = Counter(compute_ngrams(hypothesis, i))
		ref_ngrams = Counter(compute_ngrams(reference,i))
		count = sum(hyp_ngrams.values())
		for key in hyp_ngrams.keys():
			if key in ref_ngrams.keys() :
				if hyp_ngrams[key] > ref_ngrams[key]:
					hyp_ngrams[key] = ref_ngrams[key]
			else:
				hyp_ngrams[key] = 0
		if sum(hyp_ngrams.values()) == 0:
			clipped_precision.append((sum(hyp_ngrams.values())+1)/(count + 1))	
		else:
			clipped_precision.append(sum(hyp_ngrams.values())/count)
	bleu_score = [0.25 * math.log(precision) for precision in clipped_precision]
	bleu_score = math.exp(math.fsum(bleu_score))

	if len(hypothesis.split())<=len(reference.split()):
		BP = math.exp(1-len(hypothesis.split())/len(reference.split()))
	
	return BP*bleu_score

#original = "It is a guide to action which ensures that the military alwasy obeys the command of the party"
#machine_translated = "It is the guiding principle which guarantees the military forces alwasy being under the command of the party"
#print(compute_ngrams_2(machine_translated, 4))
#print(compute_bleu_sentence_2(machine_translated, original))

def compute_bleu_corpus(corpus) : # la dernière version; MAIS le résultat ne correspond pas au résultat NLTK!
	"""la fonction qui calcule le score bleu pour tout le corpus en utilisant le "micro_average precision"
	""" 
	hypothesis = ""
	reference = ""
	numerators = defaultdict(int)
	denominators = defaultdict(int)
	clipped_precision = defaultdict(float)
	hyp_lengths = 0
	ref_lengths = 0
	BP = 1
	for sentence in corpus:
		hypothesis = sentence["hyp"].lower()
		reference = sentence["ref"].lower()
		hyp_lengths += len(hypothesis.split())
		ref_lengths += len(reference.split())
		for i in range(1,5):
			hyp_ngrams = Counter(compute_ngrams(hypothesis, i))
			ref_ngrams = Counter(compute_ngrams(reference,i))
			for key in hyp_ngrams.keys():
				if key in ref_ngrams.keys() :
					if hyp_ngrams[key] > ref_ngrams[key]:
						hyp_ngrams[key] = ref_ngrams[key]
				else:
					hyp_ngrams[key] = 0
			numerators[i] += sum(hyp_ngrams.values())
			denominators[i] += sum(ref_ngrams.values())
			if numerators[i] == 0:
				for j in numerators:
					numerators[j] +=1
					denominators[j] +=1
	for j in range(1,5):
		clipped_precision[j] = numerators[j]/denominators[j]

	if hyp_lengths<=ref_lengths:
		BP = math.exp(1-hyp_lengths/ref_lengths)

	bleu_score = [0.25 * math.log(precision) for precision in clipped_precision.values()]
	bleu_score = BP * math.exp(math.fsum(bleu_score))
	
	return BP*bleu_score

# score standartization 

def to_z_score (scores):
	"""la fonction qui transforme les scores DA ou BLEU en z-scores pour comparer ensuite aux scores BLEU
	cela fonctionne pour une direction
	prend en arguments une liste de scores pour une direction"""
	mean = statistics.mean(scores)
	standart_dev = statistics.stdev(scores)
	return [(score - mean)/standart_dev for score in scores]

def convert_scale (scores, scale): 
	""" la fonction qui permet de standartiser les scores càd de convertir les valeurs aux mêmes limites (ici 0-1)
	prend en args la liste des scores et le "scale" (range) des scores (ex. 1 pour le score BLEU et 4 pour le score DA)"""
	minimum = min(scores)
	return [((x-minimum)/scale)*1 for x in scores]
def main():
	json_file="da_newstest2016.json"
	corpus = json.load(open(json_file))

	#SCORE ENG

	#print(score_eng(corpus))
	""" resultat:(scr:-0.22951067088406937, tgt:-0.03456279280542703)"""
	
	#DISTANCE D'EDITION
	scores=distance_edition_directions(corpus)
	z_score=[()]
	scaled_score=[()]
	for k,v in scores.items():
		z_score.append((k,to_z_score(v)))
		scaled_score.append((k,convert_scale(v,1)))

	print(z_score)
	print(scaled_score)

	
if __name__ == '__main__':

    main()
