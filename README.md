# Predilex ENS challenge

This project contains code for an online [Machine Learning Competition](https://challengedata.ens.fr/challenges/24) and was done in collaboration with [Mohamed Bejaoui](https://github.com/mohamedbejaoui). The context of the Predilex challenge is in relation with french jurisprudence texts. Those texts are judiciary texts that describe how a victim was involved in an accident and if the victim eventually recovered from the accident (consolidation).

The goal of the challenge is to leverage the resources of Machine Learning to create a system that will be able to automatically retrieve the gender of the victim, the date of accident and the date of consolidation from these texts.

***Summary***
1. Data Description
2. Our Approach
3. Results

1. Data Description

- The challenge provides us with a corpus of 770 texts for training. An extract of a text is displayed below:

```bash
Mademoiselle Laetitia X... représentée par Me NARRAN, avoué assistée de Me Jacques BERTRAND, avocat APPELANTE d’un jugement du Tribunal de Grande Instance d’AGEN en date du 08 Janvier 2002 D’une part, ET : S.A.R.L. AMBIANCE CLUB enseigne “L’ECLIPSE” agissant poursuites et diligences de son gérant Y... Maxime D. Jean Vaillant Z... 47330 CASTILLONNES représentée par Me Jean Michel BURG, avoué: assistée de la SCP BRIAT-MERCIER, avocats UNION DES MUTUELLES ACCIDENTS ELEVES prise en la personne de son représentant légal actuellement en fonctions domicilié en cette qualité au siège 62 rue Louis Bouilhet 76044 ROUEN CEDEX représentée par la SCP VIMONT J. ET E., avoués assistée de la SCP DELMOULY - GAUTHIER - THIZY, avocats CAISSE PRIMAIRE D’ASSURANCE MALADIE DE LOT ET GARONNE prise en la personne de son représentant légal actuellement en fonctions domicilié en cette qualité au siège 2 rue Diderot 47000 AGEN représentée par Me Jean Michel BURG, avoué INTIMEES D’autre part, a rendu l’arrêt contradictoire suivant après que la cause ait été débattue et plaidée en audience publique, le 24 Mars 2003, devant Jean-Louis BRIGNOL, Président de Chambre, Catherine LATRABE et Christian COMBES, Conseillers, assistés de Monique FOUYSSAC, Greffière, et qu’il en ait été délibéré par les magistrats du siège ayant assisté aux débats, les parties ayant été avisées de la date à laquelle l’arrêt serait rendu.
```

- The labels for each text is 3 strings: victim's gender ('male' or 'female'), accident and consolidation dates in the format 'dd/mm/yy'.

2. Our Approach

Instead of using one Machine Learning algorithm to extract all 3 labels at the same, we trained 3 different ML algorithms to extract each 3 of the labels

- For victim gender, we consider the whole text as a bag of words (TD-IDF) and use an ensemble method (xgboost) to predict if a bag of words (text) correspond to 'male' or 'female'.

- For dates extraction (both accident and consolidation), we can't directly use a supervised learning method and need to transform the data a little bit.
    - We first build a function ***F*** that return all sentences of the text with a date in it.
    - Then we label each sentence as ***True*** if this sentence is related to an accident/consolidation else ***False***.
    - Here, we created a dataset that can fuel a supervised learning approach and train a classifier to separate each sentence into either of both categories
    - Finally to extract the accident/consolidation date of a text, we use the function ***F*** to return all sentences with a date in it, then use the classifier to make a prediction on each of the sentences. The sentence with the highest prediction score for category ***True*** would be chosen as the one that contain the accident/consolidation date of this text.

3. Results 

As of March 19th 2020, we results on the public leaderboard of the challenge is displayed below.

<img src="https://user-images.githubusercontent.com/34350063/79867527-c3d95500-83de-11ea-893b-0823cf1513e0.png" width="400" height="500">