# Pipeline d’analyse d’image par photogrammétrie

<br>

## Description
Ce projet à pour objetctif de mesurer les traits architecturaux de couverts végétaux dans un contexte d'étude des mélanges variétaux de blé.
Différents génotypes sont cultivés dans des bacs séparés, en culture pure ou en mélange.
Plusieurs dispositifs d'acqisition d'image, manuel ou automatique, permettent une collecte de données régulière à différents stades de croissance.

Le projet comporte deux méthodes, s'adaptant chacune à un dispositif d'aquisition.
Ici, on se propose de traiter les images par photogrammétrie à l'aide notamment de la librairie Python openCV.

<br>

## Installation

1. **Clonez le dépôt :**
    ```bash
    git clone https://github.com/aloeb-gh/Photogrammetrie_openCV_Mobidiv.git
    ```

2. **Accédez au répertoire du projet :**
    ```bash
    cd Photogrammetrie_openCV_Mobidiv
    ```

3. **Créez et activez un environnement virtuel :**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4. **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

<br>

## Utilisation
**Lancer le pipeline :**

```bash
python3 photogrammetrie_openCV.py 
```

**Selectionner votre dossier :**

Dans l'interface qui apparait, sélectionner au choix : 
- dossier *plot* contenant les images à traiter.
- dossier *Session* contenant plusieurs dossiers *plot*.
- dossier *racine* contenant plusieurs dossiers *Session*.

**Choisir le nombre de zones :**

Pour calculer localement les hauteur des plantes, la région du bac est découpée en zones de même taille.
Par défault, le nombre de zone est fixé à 100.
Augmenter ce nombre permet une meilleure résolution et d'avantage de données de hauteur.
Cependant, un nombre de zone trop important peut faire apparaitre des valeurs aberrantes.
Pour une répartition égale des zones, il est préférable de choisir une puissance (81, 100, 169, 225 etc.).


**Choisir le seuil de filtre des petits objets :**

Les petits objets (cailloux etc.) peuvent perturber la détection du bac.
Il est nécessaire de filtrer ces derniers par leur taille.
Par défault, la valeur du seuil est fixé à 300 (pixels).
Il est souvent nécessaire d'augmenter cette valeur (ex : 2000, 5000) lorsque les plantes occupent peu de surface (notamment lors des premiers stades de croissance).

**Outputs :**

- fichier .csv comprenant la hauteur de chaque zone (dans le dossier sélectionné).
- représentation graphique des hauteur de chaque zone (dans le dossier *plot*).
- représentation graphique des contours du bac détecté sur chaque image (dans le dossier *plot*).

<br>

## Ressources utiles

Suivi du projet : 
https://aloeb.notion.site/Suivi-du-projet-482f379e883b4974b1b2b95aec96181d

Projet Mobidiv :
https://mobidiv.hub.inrae.fr/
