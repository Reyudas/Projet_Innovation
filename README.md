### Projet d'innovation

## COLIN Théo 
## VOUGEOT Valentin

## Installation du projet

```
python -m pip install -r requirements.txt
```

## Programme d'analyse et filtrage des données

```
python dataAnalysis.py
```

# Ce programme affiche des graphes d'analyses ainsi que des informations utiles et filtre les données
# Attention, il lis le fichier ./data/AshleyMadison.txt et le transforme en PasswordFiltered.txt

## Programme RNN pytorch 

# Train le modèle

```
python passwordGeneration.py -d ./data/PasswordFiltered.txt
```
Une fois entrainé un fichier ./models/rnn.pt sera créé.

# Testé le modele 

```
python passwordGeneration.py -d ./data/PasswordFiltered.txt -te test --n -1 --p 5

```

## Programme modèle Tensorflow

```
python TFModel.py

```


## Le fichier projet_d'innovation.py est un export du code provenant du google collab, se référer au notebook Projet d'innovation.ipynb 