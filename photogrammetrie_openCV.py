# importation des bibliotheques
import carte_profondeur
import position_bac
import hauteurs_plantes
from pathlib import Path
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


def main():
    n_plot = 0
    # PATH
    PATH = "/home/loeb/Documents/Comparaison_mesures"
    #PATH = "/home/loeb/Documents/Literal_mobidiv_2023"
    n_zones = 10**2
    print('nombre de zones =', n_zones)
    csv_path = PATH + "/" + "hauteurs_opencv" + str(n_zones) + ".csv"
    sessionlist = os.listdir(PATH)
    for session in tqdm(sorted(sessionlist)):
        if session.find("Session") == 0:
            print(session)
            plotlist = os.listdir(PATH + "/" + session)
            for plot in tqdm(sorted(plotlist)):
                if plot.find("uplot_7_1") == 0:
                    print(plot)
                    imglist = os.listdir(PATH + "/" + session + "/" + plot)
                    for file in imglist:
                        if "_camera_1_2_RGB.jpg" in file:
                            print(file)

                            # chargement des images gauche et droite
                            left_path = (PATH + "/" + session + "/" + plot + "/" + file)
                            id_image = left_path.split('camera_1')
                            right_path = 'camera_2'.join(id_image)
                            image_left = cv.imread(left_path)
                            image_right = cv.imread(right_path)

                            # carte de profondeur, avec suppression du capteur
                            depth_image = carte_profondeur.workflow_carte_profondeur(image_left, image_right)

                            # Extraire la region du bac
                            haut, bas, gauche, droite, image_left_bac, image_right_bac = position_bac.contour_bac(image_left, image_right)
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
                            print(hauteur_moyenne, hauteur_mediane, hauteur_min, hauteur_max, variance_hauteur, ecartype_hauteur)


                            # Enregistrement des fichiers
                            sauvegarder_image(image_left_bac, PATH + "/" + session + "/" + plot, file.replace('RGB', 'bac'))
                            sauvegarder_image(image_left_bac, PATH + "/" + session + "/" + plot, file.replace('camera_1', 'camera_2').replace('RGB', 'bac'))
                            sauvegarder_image(figure_h, PATH + "/" + session + "/" + plot, 'grille_hauteur.jpg')



                            # Export des hauteurs locales en csv
                            with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'a', newline='') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                if n_plot == 0:
                                    csv_writer.writerow([' '] + ['n° zone'] + [n for n in range(1, n_zones + 1)])
                                    n_plot += 1
                                csv_writer.writerow([session] + [plot] + [str(h) for h in liste_hauteurs if not math.isnan(h)])

    # csv en ligne -> csv en colonne
    with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'r') as csvfile_temp, open(csv_path, 'w', newline='') as csvfile_final:
        csv_reader = csv.reader(csvfile_temp)
        csv_writer = csv.writer(csvfile_final)
        data_transposed = list(zip_longest(*csv_reader, fillvalue=None))
        csv_writer.writerows(data_transposed)
    os.remove(os.path.basename(csv_path).replace(".csv", "_temporary.csv"))


if __name__ == "__main__":
    main()



#plt.ion()  # active l'affichage automatique
# plt.figure() and plt.imshow(image_left)
# plt.figure() and plt.imshow(image_right)
# plt.figure() and plt.imshow(depth_image, cmap='jet', vmin=1200, vmax=2000) and plt.colorbar()
# plt.figure() and plt.imshow(image_cut, cmap='jet', vmin=1200, vmax=1900)
# plt.figure() and plt.imshow(mat_filtree, cmap='jet', vmin=1200, vmax=1900) and plt.colorbar()
# plt.figure() and plt.imshow(carte_hauteur) and plt.colorbar()
# plt.figure() and plt.imshow(grille_h) and plt.colorbar()
# plt.figure() and plt.imshow(image_left_bac)
# plt.figure() and plt.imshow(image_right_bac)

