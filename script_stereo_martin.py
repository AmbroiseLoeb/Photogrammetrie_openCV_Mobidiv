# importation des bibliotheques
import os
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
import csv
from itertools import zip_longest


def raccourcir_image(image):
    hauteur, largeur = image.shape[:2]
    # Calculer les nouvelles dimensions
    nouvelle_largeur = int(largeur * 0.75)
    g = int((largeur - nouvelle_largeur) / 2)
    d = largeur - g
    # Recadrer l'image
    photo = image[:, g:d]

    return photo


def bord_bac(image):
    """ definir les contour du bac sur la photo, detecter et supprimer le capteur si besoin """

    # convertir en rgba (a = transparence)
    img_sans_capteur = cv.cvtColor(image, cv.COLOR_RGB2RGBA)

    # definir la colone et la ligne centrale
    height, width, color = img_sans_capteur.shape
    centre_colonne = width // 2
    centre_ligne = height // 2

    # masque des pixels verts
    hsv_img = cv.cvtColor(img_sans_capteur, cv.COLOR_BGR2HSV)
    lower_green = np.array([10, 20, 20])  # Valeurs min de teinte, saturation et valeur pour la couleur verte
    upper_green = np.array([100, 255, 255])  # Valeurs max de teinte, saturation et valeur pour la couleur verte
    mask_green = cv.inRange(hsv_img, lower_green, upper_green)
    img_without_green = cv.bitwise_and(img_sans_capteur, img_sans_capteur, mask=~mask_green)  # Appliquer le masque
    img_gray = cv.cvtColor(img_without_green, cv.COLOR_BGR2GRAY)  # Convertir en niveaux de gris

    # seuil de gris
    seuil_gris = max(np.mean(img_gray), 20)
    _, thresholded_img = cv.threshold(img_gray, seuil_gris, 255, cv.THRESH_BINARY)
    # plt.figure() and plt.imshow(thresholded_img)

    # Étiqueter les objets, calculer leur coordonnées et leur taille
    labels, nb_labels = ndi.label(thresholded_img)
    coordinates = ndi.center_of_mass(thresholded_img, labels, range(nb_labels + 1))
    sizes = ndi.sum(thresholded_img, labels, range(nb_labels + 1))
    # Supprimer les objets inférieurs à 300 pixels et le capteur
    seuil2 = 0

    filtered_image = np.zeros_like(thresholded_img)
    for label in range(1, nb_labels + 1):
        if 1200 <= coordinates[label][0] <= 2500 and 1500 <= coordinates[label][1] <= 3200:  # capteur
            if 20000 * 255 <= sizes[label] <= 80000 * 255:
                img_sans_capteur[labels == label] = (0, 0, 0, 0)
                seuil2 = 100000  # augmentation du seuil en cas de présence du capteur
                filtered_image[labels == label] = 255
            elif sizes[label] >= 5000 * 255:
                filtered_image[labels == label] = 255
        elif sizes[label] >= 300 * 255:
            filtered_image[labels == label] = 255
    # plt.figure() and plt.imshow(filtered_image)

    # filtered_image = thresholded_img
    # paramettres pour recherche des bords de bac
    largueur_min = 1000*0
    longueur_min = 1200*0
    nouvelle_longueur = 0
    nouvelle_largeur = 0
    seuil_bordure = 20000
    nouvelle_largeur_haut = centre_ligne
    nouvelle_largeur_bas = centre_ligne
    nouvelle_longueur_gauche = centre_colonne
    nouvelle_longueur_droite = centre_colonne
    n = 0

    # Recherche des bords de bac
    while nouvelle_longueur <= longueur_min or nouvelle_largeur <= largueur_min and n <= 2:
        if nouvelle_largeur <= largueur_min:
            colonne = centre_colonne
            for colonne in range(centre_colonne, width):
                if np.sum(filtered_image[1200:2500, colonne]) > max(seuil_bordure, seuil2):
                    break
            nouvelle_longueur_droite = colonne

            for colonne in range(centre_colonne, -1, -1):
                if np.sum(filtered_image[1200:2500, colonne]) > max(seuil_bordure, seuil2):
                    break
            nouvelle_longueur_gauche = colonne
            nouvelle_largeur = nouvelle_longueur_droite - nouvelle_longueur_gauche

        if nouvelle_longueur <= longueur_min:
            ligne = centre_ligne
            for ligne in range(centre_ligne, height):
                if np.sum(filtered_image[ligne, min(1800, nouvelle_longueur_gauche):min(3200, nouvelle_longueur_droite)]) > max(seuil_bordure, seuil2):
                    break
            nouvelle_largeur_bas = ligne

            for ligne in range(centre_ligne - 100, -1, -1):
                if np.sum(filtered_image[ligne, min(1800, nouvelle_longueur_gauche):min(3200, nouvelle_longueur_droite)]) > max(seuil_bordure, seuil2):
                    break
            nouvelle_largeur_haut = ligne
            nouvelle_longueur = nouvelle_largeur_bas - nouvelle_largeur_haut

        filtered_image[nouvelle_largeur_haut:nouvelle_largeur_bas, nouvelle_longueur_gauche:nouvelle_longueur_droite] = 0
        n = n+1

    # plt.figure() and plt.imshow(filtered_image)
    return (nouvelle_largeur_haut, nouvelle_largeur_bas, nouvelle_longueur_gauche, nouvelle_longueur_droite,
            img_sans_capteur)


def contour_bac(image1, image2):
    """ definir les contour finaux du bac, a partir des contours de bac de deux photos """

    # definir la colone et la ligne centrale
    height, width, color = image1.shape
    centre_colonne = width // 2
    centre_ligne = height // 2

    # charger les contours de bac des images 1 et 2
    haut_left, bas_left, gauche_left, droite_left = bord_bac(image1)[:-1]
    haut_right, bas_right, gauche_right, droite_right = bord_bac(image2)[:-1]

    # definir les nouveaux contours du bac
    '''
    h = max(haut_left, haut_right)
    b = min(bas_left, bas_right)
    g = max(gauche_left, gauche_right)
    d = min(droite_left, droite_right)
    '''

    h = min(haut_left, haut_right)
    b = max(bas_left, bas_right)
    g = min(gauche_left, gauche_right)
    d = max(droite_left, droite_right)


# ajouter un bord manquant

    if b - h > 1.5 * (d - g):
        if b - centre_ligne < centre_ligne - h:
            h = max(h, int(b - 1.2 * (d - g)))
        else:
            b = min(b, int(h + 1.2 * (d - g)))
    if d - g > 1 * (b - h):
        if d - centre_colonne < centre_colonne - g:
            g = max(g, int(d - 0.8 * (b - h)))
        else:
            d = min(d, int(g + 0.8 * (b - h)))

    return h, b, g, d


def carte_profondeur(image1, image2):
    """ cretation d'une carte de profondeur a partir de deux images """

    # chargement des parametres stereo
    stereo_path = "/home/loeb/Documents/PycharmProjects/Photogrammetrie_openCV_Mobidiv/calibration"
    Q = np.load(stereo_path + f"/Q.npy")
    FL = np.load(stereo_path + "/P1.npy")[0][0]
    T = np.load(stereo_path + "/T.npy")
    B = np.linalg.norm(T)
    mapx11 = np.load(stereo_path + "/mapx11.npy")
    mapx12 = np.load(stereo_path + "/mapx12.npy")
    mapx21 = np.load(stereo_path + "/mapx21.npy")
    mapx22 = np.load(stereo_path + "/mapx22.npy")

    # definition des parametres de disparite
    Dmax = 100 * 1000
    Dmin = .5 * 1000
    blockSize = 5
    MinDisp = int(np.floor(FL * B / Dmax))
    MaxDisp = int(np.ceil(FL * B / Dmin))
    numDisparities = MaxDisp - MinDisp
    if T[np.argmax(abs(T))] > 0:
        min_disp = - MaxDisp
    else:
        min_disp = MinDisp

    # conversion en niveaux de gris
    img_l = cv.cvtColor(image1, cv.IMREAD_GRAYSCALE + cv.IMREAD_IGNORE_ORIENTATION)
    img_r = cv.cvtColor(image2, cv.IMREAD_GRAYSCALE + cv.IMREAD_IGNORE_ORIENTATION)
    # rgb_l = cv.cvtColor(image1, cv.COLOR_BGR2RGB)

    # rectification des images (transformation de perspective)
    imglCalRect = cv.remap(img_l, mapx11, mapx12, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    imgrCalRect = cv.remap(img_r, mapx21, mapx22, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    # rgblCalRect = cv.remap(rgb_l, mapx11, mapx12, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    # reduction taille des images
    h_ori, w_ori = imglCalRect.shape
    isubsampling = 2
    imglCalRect = cv.resize(imglCalRect, (round(w_ori / isubsampling), round(h_ori / isubsampling)),
                            interpolation=cv.INTER_AREA)
    imgrCalRect = cv.resize(imgrCalRect, (round(w_ori / isubsampling), round(h_ori / isubsampling)),
                            interpolation=cv.INTER_AREA)

    # configuration de StereoSGBM (Stereo Semi-Global Block Matching ?)
    stereo = cv.StereoSGBM.create(minDisparity=round(min_disp / isubsampling),
                                  numDisparities=round(numDisparities / isubsampling),
                                  blockSize=5,
                                  uniquenessRatio=1,
                                  # preFilterCap=50,
                                  # disp12MaxDiff=10,
                                  P1=2 * blockSize ** 2,
                                  P2=32 * blockSize ** 2,
                                  mode=cv.StereoSGBM_MODE_HH4,
                                  speckleWindowSize=0,
                                  # speckleRange=2,
                                  )

    # calcul de la carte de disparite
    disparity = stereo.compute(imglCalRect, imgrCalRect).astype(np.float32) / 16
    disparity = cv.resize(disparity * isubsampling, (w_ori, h_ori), interpolation=cv.INTER_AREA)

    # affichage de la carte de disparite
    '''
    plt.figure() and plt.imshow(disparity, cmap='jet', vmin=np.nanquantile(disparity, 0.005), 
                                vmax=np.nanquantile(disparity, 0.995))
    '''

    # calcul de la carte de profondeur
    xyz_image = cv.reprojectImageTo3D(disparity, Q)
    x_image, y_image, z_image = cv.split(xyz_image)

    # affichage de la carte de profondeur
    '''
    plt.figure()
    plt.imshow(z_image, cmap='jet', vmin=800, vmax=1500)
    '''
    # calcul et affichage de la carte de profondeur 2
    '''
     Z = abs(FL * B / disparity)
     plt.figure() and plt.imshow(Z, cmap='jet', vmin=800, vmax=1500)
    '''

    return z_image


def filtre_points_aberrants(matrice):
    """ Supprime les points aberrants jusqu'à ce que la variation de la moyenne soit inférieure à seuil_stable_moy % """
    matrice_filtree = matrice.copy()  # Copie de la matrice pour éviter les modifications inattendues
    matrice_filtree[np.isinf(matrice_filtree)] = np.nan
    seuil_stable_moy = 0.0001

    while True:
        # Calculer la moyenne actuelle
        moyenne_actuelle = np.nanmean(matrice_filtree)

        # Trouver un seuil pour filtrer les points aberrants
        ecart_type = np.nanstd(matrice_filtree)
        limite_inf = moyenne_actuelle - 5 * ecart_type
        limite_sup = moyenne_actuelle + 4 * ecart_type

        # Remplacer les points aberrants par NaN
        nouvelle_matrice_filtree = matrice_filtree.copy()
        nouvelle_matrice_filtree[(matrice_filtree < limite_inf) | (matrice_filtree > limite_sup)] = np.nan

        # Calculer la nouvelle moyenne
        nouvelle_moyenne = np.nanmean(nouvelle_matrice_filtree)

        # Si la variation de la moyenne est inférieure à 10%, arrêter
        if abs(nouvelle_moyenne - moyenne_actuelle) / moyenne_actuelle < seuil_stable_moy:
            break

        # Mettre à jour la matrice filtrée
        matrice_filtree = nouvelle_matrice_filtree

    return matrice_filtree


def hauteur_locale(matrice, nombre_zones):
    # Taille des zones représentant n% de la matrice
    coeff = 1/math.sqrt(nombre_zones)
    zone_size = (int(matrice.shape[0] * coeff), int(matrice.shape[1] * coeff))
    # Calculer le nombre total de zones dans la matrice

    # Initialiser les listes pour stocker les résultats
    max_locals = []
    sol_locaux = []
    hauteur = []
    mat_sans_nan = matrice[~np.isnan(matrice)]
    sol_bac = np.median(np.sort(mat_sans_nan.flatten())[::-1][:int(mat_sans_nan.size * 0.02)])

    # Parcourir chaque zone
    for i in range(0, matrice.shape[0], zone_size[0]):
        for j in range(0, matrice.shape[1], zone_size[1]):
            # Extraire la zone actuelle
            zone = matrice[i:i + zone_size[0], j:j + zone_size[1]]

            # Calculer max_local et sol_local pour la zone
            zone_sans_nan = zone[~np.isnan(zone)]
            max_local = np.median(np.sort(zone_sans_nan.flatten())[:int(zone_sans_nan.size * 0.05)])
            sol_local = np.median(np.sort(zone_sans_nan.flatten())[::-1][:int(zone_sans_nan.size * 0.1)])

            if zone.shape[0]*zone.shape[1] <= 0.5 * zone_size[0]*zone_size[1]:
                hauteur.append(np.nan)
            else:
                # Ajouter les résultats à la liste
                max_locals.append(max_local)
                sol_locaux.append(sol_local)
                if sol_bac - 50 <= sol_local <= sol_bac + 50:
                    hauteur.append(sol_local - max_local)
                else:
                    hauteur.append(sol_bac - max_local)

    # Convertir les listes en tableaux numpy
    max_locals = np.array(max_locals)
    sol_locaux = np.array(sol_locaux)
    hauteur_a = np.array(hauteur)
    hauteur = hauteur_a[~np.isnan(hauteur_a)]

    mat_hauteur = matrice.copy()  # Copie de mat_filtree pour ne pas modifier l'original
    index = 0
    for i in range(0, mat_hauteur.shape[0], zone_size[0]):
        for j in range(0, mat_hauteur.shape[1], zone_size[1]):
            # Assigner la valeur de hauteur correspondante à chaque point de la zone
            mat_hauteur[i:i + zone_size[0], j:j + zone_size[1]] = hauteur_a[index]
            index += 1

    return hauteur, mat_hauteur


# PATH
PATH = "/home/loeb/Documents/Comparaison_mesures"
n_zones = 100
print('nombre de zones =', n_zones)
csv_path = PATH + "/" + "hauteurs_opencv" + str(n_zones) + ".csv"
sessionlist = os.listdir(PATH)
for session in sorted(sessionlist):
    if session.find("Session") == 0:
        print(session)
        plotlist = os.listdir(PATH + "/" + session)
        if not os.path.exists(PATH + "/" + session + "/" + "mask_z_map"):
            # Crée le fichier s'il n'existe pas
            os.makedirs(PATH + "/" + session + "/" + "mask_z_map")
        for plot in sorted(plotlist):
            if plot.find("uplot_7_1") == 0:
                print(plot)
                imglist = os.listdir(PATH + "/" + session + "/" + plot)
                for file in imglist:
                    if "_camera_1_2_RGB.jpg" in file:

                        # chargement des images gauche et droite
                        left_path = (PATH + "/" + session + "/" + plot + "/" + file)
                        id_image = left_path.split('camera_1')
                        right_path = 'camera_2'.join(id_image)

                        image_left = cv.imread(left_path)
                        image_right = cv.imread(right_path)
                        # plt.figure() and plt.imshow(image_left)

                        # carte de profondeur, avec suppression du capteur
                        depth_image1 = carte_profondeur(bord_bac(image_left)[4], image_right)
                        depth_image2 = carte_profondeur(image_left, bord_bac(image_right)[4])
                        depth_image = (depth_image1 * depth_image2) / depth_image1
                        # plt.figure() and plt.imshow(depth_image, cmap='jet', vmin=800, vmax=1500)

                        # Extraire la region du bac
                        haut, bas, gauche, droite = contour_bac(image_left, image_right)
                        image_cut = np.zeros_like(depth_image, dtype='float32')
                        image_cut[haut:bas, gauche:droite] = depth_image[haut:bas, gauche:droite]
                        # image_cut = depth_image[haut:bas, gauche:droite]
                        # plt.figure() and plt.imshow(image_cut, cmap='jet', vmin=800, vmax=2000)

                        print(file)

                        # Filtre des points aberrants
                        # plt.figure() and plt.imshow(image_cut[haut:bas, gauche:droite], cmap='jet', vmin=500, vmax=2000)
                        mat_filtree = filtre_points_aberrants(image_cut[haut:bas, gauche:droite])

                        '''
                        # Exporter z_map vers dosier mask_z_map
                        plt.figure() and plt.imshow(mat_filtree, cmap='jet', vmin=800, vmax=2000)
                        plt.savefig(PATH + "/" + session + "/mask_z_map/" +
                                    os.path.basename(file).replace("camera_1_2_RGB", plot + "_z_map"), dpi='figure')
                        plt.close()
                        '''

                        # Calcul des hauteurs locales
                        liste_hauteurs, z_mat = hauteur_locale(mat_filtree, n_zones)
                        print(liste_hauteurs)
                        plt.figure() and plt.imshow(z_mat, cmap='jet', vmin=0, vmax=1000)

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
