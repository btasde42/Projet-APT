
import json
import numpy as np


def score_eng(array):
	score_target=0
	score_scr=0
	count_target=0
	count_scr=0
	for i in array:
		if i['scr_lang']=='en':
			count_scr+=1
			score_scr+=i['score']


		if i['tgt_lang']=='en':
			count_target+=1
			score_target+=i['score']

	return(score_scr/count_scr,score_target/count_target)

def distance_edition(ph1,ph2):
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

	#print(table)
	return(table[len(ph1)-1,len(ph2)-1])


def main():
	json_file="da_newstest2016.json"
	with open(json_file, 'r') as f:
		array = json.load(f)

	#SCORE ENG
	
	print(score_eng(array))
	
	#DISTANCE D'EDITION
	total_dist=0
    
    
	for i in array:
		p1=i['hyp']
		p2=i['ref']
		total_dist+=distance_edition(p1,p2)
	#print(total_dist/len(array))

if __name__ == '__main__':

    main()
