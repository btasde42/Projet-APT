> # Tests
> 1. 'projet.py' calcule le score BLEU sur le corpus et phrase par phrase, la distance d'édition entre les hypothèses de traduction et les références et compare les résultats en standartisant les scores
> - 'all_lang(corpus) ' la fonction qui calcule le nombre de langues utilisées dans le corpus
> - 'compute_directions(corpus)' la fonction qui calcule le nombre d'exemples pour chaque direction
> - 'get_directions(corpus)' retourne la liste des directions
> - 'score_eng(corpus)' la fonction qui compare le score de traductions depuis et vers anglais
> - 'distance_edition(phrase1, phrase2)' la fonction qui calcule la distance d'édition entre deux phrases en forme d'un tableau
> - 'distance_edition_directions(corpus)'
> - 'length_score_impact(direction)' la fonction qui fait un plot de la dépendance du score de la longueur de la phrase
> - 'compute_ngrams (sentence, n)' la fonction qui renvoie la liste de ngrams à partir d'une phrase
> - 'compute_bleu_sentence (hypothesis, reference)' la fonction qui calcule le score bleu phrase par phrase
> - 'compute_bleu_corpus(corpus)' la fonction qui calcule le score bleu pour tout le corpus en utilisant le "micro_average precision"
> - 'to_z_score (scores)' la fonction qui transforme les scores DA ou BLEU en z-scores pour comparer ensuite aux scores BLEU
> - 'convert_scale (scores, scale)' la fonction qui permet de standartiser les scores càd de convertir les valeurs aux mêmes limites
> - 'compare_methode_phrase(corpus)' la fonction pour la comparaison entre Bleu score et la distance d'édition au niveau de phrase
> - 'compare_methode_corpus(corpus)' la fonction pour la comparaison entre Bleu score et la distance d'édition au niveau des directions entieres

> 2. 'similarity.py' analyse les paramètres permettant de distinguer les phrases sources issues de la traduction ou écrites par un locuteur natif : 
> - 'compare_length(directions)' la fonction qui compare la longueur des phrases sources issues de la traduction et écrites par un locuteru natif
> - 'compare_length_to_graph(directions)' la fonction qui compare la longueur des phrases sources issues de la traduction et écrites par un locuteru natif et fait un plot pour visualiser la différence
> - 'compare_heights(direction_direct, direction_indirect, language_model)' la fonction qui compare la profondeur des arbres de dépendance des phrases sources issues de la traduction et écrites par un locuteur natif
> - 'get_height(root)' la fonction qui retourne la profondeur de l'arbre de dépendance
> - 'compute_lexical_richness (direction, nlp)' la fonction qui calcule la richesse lexicale de la phrase;
	retourne le nombre de tokens uniques/nombre de tokens total
> - 'lexical_richness_to_graph(directions)' visualise la différence de la richesse lexicale des phrases sources de toutes les directions
> - 'pos_trigrams(direction, nlp)' la fonction qui renvoie les trigrammes des PoS avec leur fréquence
> - 'pos_position(direction, nlp)' la fonction qui analyse la fréquence de l'apparition des tokens aux positions initiales (le 1er ou le 2ième mot de la phrase) ou finales (le dernier ou l'avant dernier mot)
> - 'dependent_head_relation_position_direction (direction ,nlp)' la fonction qui analyse le positionnement du gouverneur par rapport au dépendant et retourne un dictionnaire dans dictionnaire (clé : tuple (PoS du head, PoS du dépendant, relation de dépendance), clé : 'right', 'left' , valeur : fréquence)
> - 'dict_to_graph(D)' visualise un dictionnaire sur un graphique
> - 'dict_of_dict_to_graph(dictt, name)' visualise un dictionnaire dans un dictionnaire sur un graphique

