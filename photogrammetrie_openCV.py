# importation des bibliotheques
import carte_profondeur
import position_bac
import hauteurs_plantes
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog
# import PySimpleGUI as sg
import os
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
import csv
from itertools import zip_longest
from tqdm import tqdm


def sauvegarder_image(image, path_dossier, nom_fichier):
    """ Enregistrer une image ou une figure dans un dossier. """

    def figure_to_numpy(fig):
        """Convertit une figure Matplotlib en tableau numpy."""
        fig.set_dpi(800)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(path_dossier):
        os.makedirs(path_dossier)

    # Construire le chemin complet du fichier
    chemin_complet = os.path.join(path_dossier, nom_fichier)

    # Vérifier que l'image est un tableau numpy
    if isinstance(image, np.ndarray):
        # Enregistrer l'image
        cv.imwrite(chemin_complet, image)
    else:
        image_np = figure_to_numpy(image)
        cv.imwrite(chemin_complet, image_np)


def traiter_dossier_racine(racine_path):
    sessionlist = os.listdir(racine_path)
    for session in tqdm(sorted(sessionlist)):
        if session.find("Session") == 0:
            print(session)
            traiter_dossier_session(os.path.join(racine_path, session))


def traiter_dossier_session(session_path):
    plotlist = os.listdir(session_path)
    for plot in tqdm(sorted(plotlist)):
        if plot.find("uplot") == 0:
            print(plot)
            traiter_dossier_plot(os.path.join(session_path, plot), os.path.basename(session_path))


def traiter_dossier_plot(plot_path, session_name):
    imglist = os.listdir(plot_path)
    for file in imglist:
        if "_camera_1_2_RGB.jpg" in file:
            print(file)

            # chargement des images gauche et droite
            left_path = (plot_path + "/" + file)
            id_image = left_path.split('camera_1')
            right_path = 'camera_2'.join(id_image)
            image_left = cv.imread(left_path)
            image_right = cv.imread(right_path)

            # carte de profondeur, avec suppression du capteur
            depth_image = carte_profondeur.workflow_carte_profondeur(image_left, image_right)

            # Extraire la region du bac
            haut, bas, gauche, droite, image_left_bac, image_right_bac = position_bac.contour_bac(image_left, image_right, seuil_small_obj)
            image_cut = depth_image[haut:bas, gauche:droite]

            # Filtre des points aberrants
            mat_filtree = hauteurs_plantes.filtre_points_aberrants(image_cut)

            # Calcul des hauteurs locales
            carte_hauteur, profondeur_sol = hauteurs_plantes.carte_hauteur_absolue(mat_filtree, n_zones)
            liste_hauteurs, grille_h, figure_h = hauteurs_plantes.hauteur_par_zone(carte_hauteur, n_zones)
            print(liste_hauteurs)

            # Stats hauteurs locales
            hauteur_moyenne = np.nanmean(liste_hauteurs)
            hauteur_mediane = np.nanmedian(liste_hauteurs)
            hauteur_min = np.nanmin(liste_hauteurs)
            hauteur_max = np.nanmax(liste_hauteurs)
            variance_hauteur = np.nanvar(liste_hauteurs)
            ecartype_hauteur = np.nanstd(liste_hauteurs)
            # print(hauteur_moyenne, hauteur_mediane, hauteur_min, hauteur_max, variance_hauteur, ecartype_hauteur)

            # Enregistrement des fichiers
            sauvegarder_image(image_left_bac, plot_path, file.replace('RGB', 'bac'))
            sauvegarder_image(image_left_bac, plot_path, file.replace('camera_1', 'camera_2').replace('RGB', 'bac'))
            sauvegarder_image(figure_h, plot_path, f"grille_hauteur_{folder_name}_{n_zones}z.jpg")

            # Export des hauteurs locales en csv
            with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([session_name] + [os.path.basename(plot_path)] + [str(h) for h in liste_hauteurs if not math.isinf(h)])


def main():
    global n_zones, csv_path, seuil_small_obj, folder_name

    # Interface utilisateur pour sélectionner un dossier
    root = tk.Tk()
    root.withdraw()
    selected_path = filedialog.askdirectory(initialdir="/home/loeb/Documents", title="Sélectionnez un dossier")

    # Interface pour sélectionner le nombre de zones
    n_zones = simpledialog.askinteger("Nombre de zones", "Veuillez choisir un nombre de zones : \n (correspond au maillage utilisé lors de la reconnaissance du sol et des maximas locaux)", initialvalue=100, minvalue=1)
    print('nombre de zones =', n_zones)

    # Interface pour sélectionner le seuil du filtre des petits objets
    seuil_small_obj = simpledialog.askinteger("Seuil petis objets", "Veuillez choisir une taille limite : \n (choisisser une taille plus grande lors des premiers stades de croissance)", initialvalue=300, minvalue=50)
    print('seuil du filtre des petits objets =', seuil_small_obj, 'pixels')

    if selected_path:
        folder_name = os.path.basename(selected_path)
        csv_path = os.path.join(selected_path, f"hauteurs_opencv_{folder_name}_{n_zones}z.csv")
        with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([' '] + ['n° zone'] + [n for n in range(1, n_zones + 1)])

        if "uplot" in os.path.basename(selected_path):
            print("Dossier plot sélectionné")
            traiter_dossier_plot(selected_path, "N/A")
        elif "Session" in os.path.basename(selected_path):
            print("Dossier session sélectionné")
            traiter_dossier_session(selected_path)
        else:
            print("Dossier racine sélectionné")
            traiter_dossier_racine(selected_path)

    # csv en ligne -> csv en colonne
    with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'r') as csvfile_temp, open(csv_path, 'w',
                                                                                                       newline='') as csvfile_final:
        csv_reader = csv.reader(csvfile_temp)
        csv_writer = csv.writer(csvfile_final)
        data_transposed = list(zip_longest(*csv_reader, fillvalue=None))
        csv_writer.writerows(data_transposed)
    os.remove(os.path.basename(csv_path).replace(".csv", "_temporary.csv"))


if __name__ == "__main__":
    main()

# plt.ion()  # active l'affichage automatique
# plt.figure() and plt.imshow(image_left)
# plt.figure() and plt.imshow(image_right)
# plt.figure() and plt.imshow(depth_image, cmap='jet', vmin=1200, vmax=2000) and plt.colorbar()
# plt.figure() and plt.imshow(image_cut, cmap='jet', vmin=1200, vmax=1900)
# plt.figure() and plt.imshow(mat_filtree, cmap='jet', vmin=1200, vmax=1900) and plt.colorbar()
# plt.figure() and plt.imshow(carte_hauteur) and plt.colorbar()
# plt.figure() and plt.imshow(grille_h) and plt.colorbar()
# plt.figure() and plt.imshow(image_left_bac)
# plt.figure() and plt.imshow(image_right_bac)

