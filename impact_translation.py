from collections import defaultdict, Counter
import json
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice
import statistics
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
	len_ph1=len(ph1)
	len_ph2=len(ph2) #on prend les long des phrases pour créer le table
	len_sum=float(len(ph1)+len(ph2))
	table=[] 

	for i in range(len_ph1+1): #on met tout d'abord indice pour chaque chr
		table.append([i])
	del table[0][0]
	for j in range(len_ph2+1):
		table[0].append(j)

	for y in range(1,len_ph2+1): #on a deux boucles pour comparer deux phrases chr par chr
		for x in range(1,len_ph1+1):
			if ph1[x-1]==ph2[y-1]: #si les chr à meme indice sont les memes:
				table[x].insert(y,table[x-1][y-1])
			else:
				mini=min(table[x-1][y]+1,table[x][y-1]+1,table[x-1][y-1]+2)
				table[x].insert(y,mini)
	dist=table[-1][-1]
	rat=(len_sum-dist)/len_sum
	return rat

def distance_edition_directions(corpus):
	"""
	separe direct-indirect
	conserver le pair de langue concerné
	"""
	dict_distance=defaultdict(list)
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
	prend en arguments une liste de scores pour une direction
	ARGS:
		scores:list
	"""
	mean = statistics.mean(scores)
	standart_dev = statistics.stdev(scores)
	return [(score - mean)/standart_dev for score in scores]

def convert_scale (scores, scale): 
	""" la fonction qui permet de standartiser les scores càd de convertir les valeurs aux mêmes limites (ici 0-1)
	prend en args la liste des scores et le "scale" (range) des scores (ex. 1 pour le score BLEU et 4 pour le score DA)"""
	minimum = min(scores)
	return [((x-minimum)/scale)*1 for x in scores]

def compare_methode_phrase(corpus):
	"""la fonction pour la comparaison entre Bleu score et la distance d'édition au niveu de phrase
	"""
	sentences_score_bleu=[]
	sentences_score_DA=[]
	sentences_score_DIST=[]

	direction_score_bleu={}
	direction_score_bleu_z={}
	direction_score_bleu_scale={}

	direction_score_DIST={}
	direction_score_DIST_z={}
	direction_score_DIST_scale={}
	
	direction_score_DA={}
	direction_score_DA_z={}
	direction_score_DA_scale={}
	for direction in get_directions(corpus):
		for phrase in direction:
			score=phrase['score'] #on compte en meme temps le DA pour la comparaison
			hyp = phrase["hyp"]
			ref = phrase["ref"]
			sentences_score_bleu.append(compute_bleu_sentence(hyp, ref))
			sentences_score_DIST.append(distance_edition(hyp,ref))
			sentences_score_DA.append(score)
		direction_score_bleu[(direction[0]["src_lang"],direction[0]["orig_lang"],direction[0]["tgt_lang"])]=sentences_score_bleu
		direction_score_DIST[(direction[0]["src_lang"],direction[0]["orig_lang"],direction[0]["tgt_lang"])]=sentences_score_DIST
		direction_score_DA[(direction[0]["src_lang"],direction[0]["orig_lang"],direction[0]["tgt_lang"])]=sentences_score_DA

	for k,v in direction_score_bleu.items():
		direction_score_bleu_z[k]=to_z_score(v)	
		direction_score_bleu_scale[k]=convert_scale(v, 1)
	
	for k,v in direction_score_DIST.items():
		direction_score_DIST_z[k]=to_z_score(v)
		direction_score_DIST_scale[k]=convert_scale(v,1)

	for k,v in direction_score_DA.items():
		direction_score_DA_z[k]=to_z_score(v)
		direction_score_DA_scale[k]=convert_scale(v,1)

	directions=compute_directions(corpus).keys()

	df1 = pd.DataFrame({'DA':statistics.mean(direction_score_DA_z[directions[0]]),'Bleu':statistics.mean(direction_score_bleu_z[directions[0]]),'Distance_edit':statistics.mean(direction_score_DIST_z[directions[0]])})
	ax=df1.plot.bar(rot=0);
	plt.show()

def compare_methode_corpus(corpus):
	"""la fonction pour la comparaison entre Bleu score et la distance d'édition au niveu des directions entieres
	"""
	list_DA_corpus=[]
	list_bleu_corpus=[]
	list_DIST_corpus=[]

	list_bleu_corpus_z=[]
	list_DIST_corpus_z=[]
	list_bleu_corpus_scale=[]
	list_DIST_corpus_scale=[]
	list_DA_corpus_z=[]
	list_DA_corpus_scale=[]

	scores_z=[]
	scores_scale=[]
	for k,v in distance_edition_directions(corpus).items():
		list_DIST_corpus.append([k,v])

	for i in get_directions(corpus):
		list_bleu_corpus.append([(i[0]["src_lang"],i[0]["orig_lang"],i[0]["tgt_lang"]),compute_bleu_corpus(i)])
		list_DA_corpus.append([(i[0]["src_lang"],i[0]["orig_lang"],i[0]["tgt_lang"]),sum([j['score'] for j in i])/len(i)]) #on calcule la moyenne des scores de DA pour chaque direction 

	list_bleu_corpus=sorted(list_bleu_corpus,key=lambda x: x[0]) #on enumere les deux listes pour qu'ils soient alignés
	list_DIST_corpus=sorted(list_DIST_corpus,key=lambda x: x[0])
	list_DA_corpus=sorted(list_DA_corpus,key=lambda x: x[0])
	
	for i in range(len(list_bleu_corpus)): #calcule des scores ajoustés
		list_bleu_corpus_z=to_z_score([i[1] for i in list_bleu_corpus])
		list_DIST_corpus_z=to_z_score([i[1] for i in list_DIST_corpus])
		list_DA_corpus_z=to_z_score([i[1] for i in list_DA_corpus])

		list_bleu_corpus_scale=convert_scale([i[1] for i in list_bleu_corpus],1)
		list_DIST_corpus_scale=convert_scale([i[1] for i in list_DIST_corpus],1)
		list_DA_corpus_scale=convert_scale([i[1] for i in list_DA_corpus],1)

	scores_z=list(zip(sorted(compute_directions (corpus).keys()),list_DA_corpus_z,list_bleu_corpus_z,list_DIST_corpus_z)) #on assemble les noms de directions et les deux scores de direction
	scores_scale=list(zip(sorted(compute_directions (corpus).keys()),list_DA_corpus_scale,list_bleu_corpus_scale,list_DIST_corpus_scale))

	
	columns=sorted(compute_directions (corpus).keys())

	#df1 = pd.DataFrame({'DA':list_DA_corpus_z,'Bleu':list_bleu_corpus_z,'Distance_edit':list_DIST_corpus_z}, index=columns)
	#ax=df1.plot.bar(rot=0);
	#plt.show()

	df2 = pd.DataFrame({'DA':list_DA_corpus_scale,'Bleu':list_bleu_corpus_scale,'Distance_edit':list_DIST_corpus_scale}, index=columns)
	ax=df2.plot.bar(rot=0);
	plt.show()


def main():
	json_file="da_newstest2016.json"
	corpus = json.load(open(json_file))
	compare_methode_phrase(corpus)
	"""
	[(('cs', 'cs', 'en'), -1.1858783621173596, -0.20590678364696838, -0.2910927752246894), (('cs', 'en', 'en'), 1.3515052965610204, 0.5711734830982574, 0.9991951640195246), (('de', 'de', 'en'), -0.6429112695433022, 0.5009100943511744, 0.11804422414690613), (('de', 'en', 'en'), 1.2014997888333605, 1.6551547074642587, 1.488366938893235), (('en', 'en', 'ru'), -1.7554917817673616, -0.9695684294220609, -0.5953978616838416), (('en', 'ru', 'ru'), 1.627883445694669, -0.12114133603470499, 0.35145532551463576), (('fi', 'en', 'en'), 0.6103531192654893, -0.4008427798430632, -0.049138115530800386), (('fi', 'fi', 'en'), -0.8520313733217793, -0.9065129953685516, -0.6219144918460731), (('ro', 'en', 'en'), 0.5793914585047785, 0.5046778088694648, 0.3765311533289205), (('ro', 'ro', 'en'), -0.304173660907757, 1.823381711094755, 1.185681238597858), (('ru', 'en', 'en'), 0.43749431152828033, -0.13989576500675385, 0.30208600204591707), (('ru', 'ru', 'en'), -0.1275356765496593, 0.46555996564667274, 0.4677419043227074), (('tr', 'en', 'en'), -0.5029412256792026, -1.5153077072286891, -1.8826974892201216), (('tr', 'tr', 'en'), -0.4371640705011766, -1.2616819739737937, -1.8488612173641714)]
	
	[(('cs', 'cs', 'en'), 0.09060835471837525, 0.09612717528827569, 0.10281618209540655), (('cs', 'en', 'en'), 0.4942297419100713, 0.1531750432630048, 0.18616758118985577), (('de', 'de', 'en'), 0.1769780806187663, 0.1480167903820127, 0.12924605118253651), (('de', 'en', 'en'), 0.4703683794754412, 0.2327534606042746, 0.21776762222962676), (('en', 'en', 'ru'), 0.0, 0.040064409818740016, 0.08315835694144535), (('en', 'ru', 'ru'), 0.53819318888872, 0.10235006920431519, 0.1443241909306089), (('fi', 'en', 'en'), 0.37633473265633455, 0.08181632034777422, 0.11844622775383529), (('fi', 'fi', 'en'), 0.14371336478123165, 0.04469350443214526, 0.08144540733068895), (('ro', 'en', 'en'), 0.3714096641026473, 0.14829338996844432, 0.14594406605165855), (('ro', 'ro', 'en'), 0.23086104113357053, 0.2451035257180028, 0.19821440791354095), (('ru', 'en', 'en'), 0.3488380978732702, 0.10097324851573933, 0.1411349786173678), (('ru', 'ru', 'en'), 0.25895882919708124, 0.14542162800107777, 0.15183619561170136), (('tr', 'en', 'en'), 0.19924310273179446, 0.0, 0.0), (('tr', 'tr', 'en'), 0.20970626880747836, 0.018619450222717965, 0.0021857916466192506)]
	"""
	#SCORE ENG

	#print(score_eng(corpus))
	
	""" resultat:(scr:-0.22951067088406937, tgt:-0.03456279280542703)"""



if __name__ == '__main__':

    main()
