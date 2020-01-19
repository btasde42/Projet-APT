import numpy as np
import json
from methodes_in_direct import*
from language_model import*

def get_directions(corpus):
	"""args : le corpus entier
	return : liste des directions (une direction = une liste de dictionnaires, dont les clés sont : 
	src, hyp, réf, score)
	"""
	deuch = [elt for elt in corpus if elt["src_lang"] == "de"]
	#de_en_en = [elt for elt in corpus if elt["src_lang"] == "de" and elt["orig_lang"] == "en" and elt["tgt_lang"] == "en"]
	eng =[elt for elt in corpus if elt["src_lang"] == "en"]
	return [deuch,eng]

def data_prepare(file,lang_corp,lang,model,nlp):
	"""
	Cette fonction prepare les vecteurs des phrases et leurs label,
	vecteurs de phrases sont composées de trois indicateurs
	"""

	#vector_phrase=[proba_model,lexical_richnes_phrase,pos_position]
	#labels={'direct':1,'indirect':-1}
	label=0
	data=[]
	specificities=[]
	pos_proba=1
	dict_pos=pos_position(lang_corp,nlp)
	for i in get_directions(file)[lang]:
		if i["src_lang"] == i["orig_lang"]:
			label=1 #direct
		else:
			label=-1 #indirect
		specificities.append(calculate_proba_sentence(model,i['ref'],2))
		specificities.append(compute_lexical_richness_phrase(i['ref'],nlp))
		doc = nlp(i['ref'])
		pos_proba=1
		for token in doc:
			if token in dict_pos['1']:
				pos_proba*=dict_pos['1'][token]/sum(dict_pos['1'].values())
			if token in dict_pos['2']:
				pos_proba*=dict_pos['2'][token]/sum(dict_pos['2'].values())
			if token in dict_pos['penultimate']:
				pos_proba*=dict_pos['penultimate'][token]/sum(dict_pos['penultimate'].values())
			if token in dict_pos['last']:
				pos_proba*=dict_pos['last'][token]/sum(dict_pos['lang_mod_engst'].values())
		specificities.append(pos_proba)
		observation=np.array([float(o) for o in specificities+[1.0]])
	data.append((label,observation))

	return data

def classify(o,w):
    
    if o.dot(w)>=0:
        return 1
    return -1

def test_classification(c,w):
    n_correct=0.0
    for example in c:
        y,o=example
        y_hat=classify(o,w)
        
        if y_hat==y:
            n_correct+=1
    
    return n_correct/len(c) #donne le taux de reussite

def main():
	json_file='da_newstest2016.json'
	corpus = json.load(open(json_file))
	nlp_en = spacy.load("en_core_web_sm")
	with open('en_lang_mod.pickle', 'rb') as handle:
		lang_mod_eng = pickle.load(handle)

	data=data_prepare(corpus,get_directions(corpus)[1],1,lang_mod_eng,nlp_en)
	train,test=data[:len(data)//2], data[len(data)//2:] #on divise en deux le data pour en extraire train et test
	w=np.ones(len(test[0][1]))
	print(len(data))

if __name__ == '__main__':
	main()
