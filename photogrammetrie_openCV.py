# Importation des bibliotheques
import carte_profondeur
import position_bac
import hauteurs_plantes
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import math
import numpy as np
import cv2 as cv
import csv
from itertools import zip_longest
from tqdm import tqdm
from matplotlib import pyplot as plt


def sauvegarder_image(image, path_dossier, nom_fichier):
    """Enregistre une image ou une figure dans un dossier."""

    def figure_to_numpy(fig):
        """Convertit une figure Matplotlib en tableau numpy."""

        fig.set_dpi(800)  # Résolution de l'image
        fig.canvas.draw()
        img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return cv.cvtColor(img_np, cv.COLOR_RGB2BGR)

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(path_dossier):
        os.makedirs(path_dossier)

    # Construire le chemin complet du fichier
    chemin_complet = os.path.join(path_dossier, nom_fichier)

    # Vérifier que l'image est un tableau numpy
    if isinstance(image, np.ndarray):
        cv.imwrite(str(chemin_complet), image)  # Enregistrer l'image
    else:
        image_np = figure_to_numpy(image)
        cv.imwrite(str(chemin_complet), image_np)  # Enregistrer l'image


def traiter_dossier_racine(racine_path):
    """Traite chaque dossier session du dossier racine."""

    sessionlist = os.listdir(racine_path)  # Liste des dossiers sessions dans le dossier racine
    for session in tqdm(sorted(sessionlist)):
        if session.find("Session") == 0:
            print(session)
            traiter_dossier_session(os.path.join(racine_path, session))


def traiter_dossier_session(session_path):
    """Traite chaque dossier du dossier session."""

    plotlist = os.listdir(session_path)  # Liste des dossiers plots dans le dossier session
    for plot in tqdm(sorted(plotlist)):
        if plot.find("uplot") == 0:
            print(plot)
            traiter_dossier_plot(os.path.join(session_path, plot), os.path.basename(session_path))


def traiter_dossier_plot(plot_path, session_name):
    """Traite par photogramétrie les images du dossier plot."""

    imglist = os.listdir(plot_path)  # Liste des images dans le dossier
    for file in imglist:
        if "_camera_1_2_RGB.jpg" in file:
            print(file)

            # Chargement des images gauche et droite
            left_path = (plot_path + "/" + file)
            id_image = left_path.split('camera_1')
            right_path = 'camera_2'.join(id_image)
            image_left = cv.imread(left_path)
            image_right = cv.imread(right_path)

            # Carte de profondeur
            depth_image = carte_profondeur.workflow_carte_profondeur(image_left, image_right)

            # Extraire la region du bac
            haut, bas, gauche, droite, image_left_bac, image_right_bac = position_bac.contour_bac(image_left, image_right, seuil_small_obj)
            image_cut = depth_image[haut:bas, gauche:droite]  # Carte de profondeur au dimensions du bac

            # Filtre des points aberrants
            mat_filtree = hauteurs_plantes.filtre_points_aberrants(image_cut)

            # Calcul des hauteurs locales
            carte_hauteur, profondeur_sol = hauteurs_plantes.carte_hauteur_absolue(mat_filtree, n_zones)  # Carte de hauteur relative au sol
            liste_hauteurs, grille_h, figure_h = hauteurs_plantes.hauteur_par_zone(carte_hauteur, n_zones)
            print(liste_hauteurs)

            # Enregistrement des fichiers
            sauvegarder_image(image_left_bac, plot_path, file.replace('RGB', 'bac'))  # Contours du bac détectés sur l'image de gauche
            sauvegarder_image(image_left_bac, plot_path, file.replace('camera_1', 'camera_2').replace('RGB', 'bac'))  # Contours du bac détectés sur l'image de droite
            sauvegarder_image(figure_h, plot_path, f"grille_hauteur_{os.path.basename(os.path.normpath(plot_path))}_{n_zones}z.jpg")  # Représentation graphique des hauteurs locales dans le bac

            # Export de la liste des hauteurs en csv
            with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([session_name] + [os.path.basename(plot_path)] + [str(h) for h in liste_hauteurs if not math.isinf(h)])


def main():
    """Fonction principale pour exécuter le pipeline."""
    global n_zones, csv_path, seuil_small_obj, folder_name

    # Interface utilisateur pour sélectionner un dossier
    root = tk.Tk()
    root.withdraw()
    selected_path = filedialog.askdirectory(initialdir="/home/loeb/Documents", title="Sélectionnez un dossier")

    # Interface utilisateur pour sélectionner le nombre de zones
    n_zones = simpledialog.askinteger("Nombre de zones", "Veuillez choisir un nombre de zones : \n (correspond au maillage utilisé lors de la reconnaissance du sol et des maximas locaux)", initialvalue=100, minvalue=1)
    print('nombre de zones =', n_zones)

    # Interface pour sélectionner le seuil du filtre des petits objets
    seuil_small_obj = simpledialog.askinteger("Seuil petis objets", "Veuillez choisir une taille limite : \n (choisisser une taille plus grande lors des premiers stades de croissance)", initialvalue=300, minvalue=50)
    print('seuil du filtre des petits objets =', seuil_small_obj, 'pixels')

    if selected_path:
        folder_name = os.path.basename(selected_path)
        csv_path = os.path.join(selected_path, f"hauteurs_opencv_{folder_name}_{n_zones}z.csv")  # Emplacement du fichier csv
        # Créer le fichier csv temporaire
        with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([' '] + ['n° zone'] + [n for n in range(1, n_zones + 1)])

        #  Définir le type de dossier sélectionné
        if "uplot" in os.path.basename(selected_path):
            print("Dossier plot sélectionné")
            traiter_dossier_plot(selected_path, "N/A")
        elif "Session" in os.path.basename(selected_path):
            print("Dossier session sélectionné")
            traiter_dossier_session(selected_path)
        else:
            print("Dossier racine sélectionné")
            traiter_dossier_racine(selected_path)

    # Passer du fichier csv en ligne au csv final en colonne
    with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'r') as csvfile_temp, open(csv_path, 'w',
                                                                                                       newline='') as csvfile_final:
        csv_reader = csv.reader(csvfile_temp)
        csv_writer = csv.writer(csvfile_final)
        data_transposed = list(zip_longest(*csv_reader, fillvalue=None))
        csv_writer.writerows(data_transposed)
    os.remove(os.path.basename(csv_path).replace(".csv", "_temporary.csv"))  # Supprimer le fichier csv temporaire


if __name__ == "__main__":
    main()
