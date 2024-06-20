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


def main():

    # PATH
    PATH = "/home/loeb/Documents/Comparaison_mesures"
    #PATH = "/home/loeb/Documents/Literal_mobidiv_2023"
    n_zones = 100
    print('nombre de zones =', n_zones)
    csv_path = PATH + "/" + "hauteurs_opencv" + str(n_zones) + ".csv"
    sessionlist = os.listdir(PATH)
    for session in tqdm(sorted(sessionlist)):
        if session.find("Session") == 0:
            print(session)
            plotlist = os.listdir(PATH + "/" + session)
            for plot in tqdm(sorted(plotlist)):
                if plot.find("uplot_30_1") == 0:
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
                            haut, bas, gauche, droite = position_bac.contour_bac(image_left, image_right)
                            image_cut = depth_image[haut:bas, gauche:droite]

                            # Filtre des points aberrants
                            mat_filtree = hauteurs_plantes.filtre_points_aberrants(image_cut)

                            # Calcul des hauteurs locales
                            carte_hauteur, profondeur_sol = hauteurs_plantes.carte_hauteur_absolue(mat_filtree, n_zones)
                            liste_hauteurs, z_mat = hauteurs_plantes.hauteur_par_zone(carte_hauteur, n_zones)
                            print(liste_hauteurs)

                            # Stats hauteurs locales
                            hauteur_moyenne = np.nanmean(liste_hauteurs)
                            hauteur_mediane = np.nanmedian(liste_hauteurs)
                            hauteur_min = np.nanmin(liste_hauteurs)
                            hauteur_max = np.nanmax(liste_hauteurs)
                            variance_hauteur = np.nanvar(liste_hauteurs)
                            ecartype_hauteur = np.nanstd(liste_hauteurs)
                            print(hauteur_moyenne, hauteur_mediane, hauteur_min, hauteur_max, variance_hauteur, ecartype_hauteur)


    """
                            # Export des hauteurs locales en csv
                            with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'a', newline='') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                csv_writer.writerow([session] + [plot] + [str(h) for h in liste_hauteurs])
    
    # csv en ligne -> csv en colonne
    with open(os.path.basename(csv_path).replace(".csv", "_temporary.csv"), 'r') as csvfile_temp, open(csv_path, 'w', newline='') as csvfile_final:
        csv_reader = csv.reader(csvfile_temp)
        csv_writer = csv.writer(csvfile_final)
        data_transposed = list(zip_longest(*csv_reader, fillvalue=None))
        csv_writer.writerows(data_transposed)
    os.remove(os.path.basename(csv_path).replace(".csv", "_temporary.csv"))
    """


if __name__ == "__main__":
    main()


# plt.figure() and plt.imshow(image_left)
# plt.figure() and plt.imshow(image_right)
# plt.figure() and plt.imshow(depth_image, cmap='jet', vmin=1200, vmax=2000) and plt.colorbar()
# plt.figure() and plt.imshow(image_cut, cmap='jet', vmin=1200, vmax=1900)
# plt.figure() and plt.imshow(mat_filtree, cmap='jet', vmin=1200, vmax=1900) and plt.colorbar()
# plt.figure() and plt.imshow(carte_hauteur) and plt.colorbar()
# plt.figure() and plt.imshow(z_mat) and plt.colorbar()
