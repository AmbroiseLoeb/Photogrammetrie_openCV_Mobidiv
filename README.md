# Pipeline d’analyse d’image mesurant les traits architecturaux du couvert

## Description
Ce projet à pour objetctif de mesurer les hauteurs d'un couvert de plantes dans un contexte d'étude des mélanges variétaux de blé.
Des bacs séparés permettent un nombre important de cultures différentes, pures ou en mélange.
Plusieurs dispositifs d'acqisition d'image, manuel ou automatique, permettent une collecte de données régulière à différents stades de croissance.

Ce projet comporte deux méthodes, s'adaptant chacune à un dispositif d'aquisition. Ici, on se propose de traiter les images par photogrammétrie à l'aide notamment de la librairie Python openCV.

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



## Utilisation
**Lancer le pipeline :**

```bash
python3 photogrammetrie_openCV.py 
```

**Selectionner votre dossier :**

Dans l'interface qui apparait, sélectionner au choix : 
- dossier *plot* contenant les images à traiter
- dossier *Session* contenant plusieurs dossiers *plot*
- dossier *racine* contenant plusieurs dossiers *Session*

**Choisir le nombre de zones :**


**Choisir le seuil de filtre des petits objets :**


## Ressources utiles

Suivi du projet : 
https://aloeb.notion.site/Suivi-du-projet-482f379e883b4974b1b2b95aec96181d

Projet Mobidiv : https://mobidiv.hub.inrae.fr/
