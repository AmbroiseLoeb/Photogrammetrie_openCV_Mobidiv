# importation des bibliotheques
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
from tqdm import tqdm


def contour_bac(image1, image2, seuil_small_obj=300):
    """ definir les contour finaux du bac, a partir des contours de bac de deux photos """

    def bord_bac(image, seuil):
        """ definir les contour du bac sur la photo """

        def label_objects(image):
            # Étiqueter les objets, calculer leur coordonnées et leur taille
            labels_, nb_labels_ = ndi.label(image)
            coordinates_ = np.array(ndi.center_of_mass(image, labels_, range(1, nb_labels_ + 1)))
            sizes_ = ndi.sum(image, labels_, range(nb_labels_ + 1))
            return labels_, nb_labels_, coordinates_, sizes_

        height, width = image.shape[:2]

        # masque des pixels verts
        hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower_green = np.array([10, 20, 20])  # Valeurs min de teinte, saturation et valeur pour la couleur verte
        upper_green = np.array([100, 255, 255])  # Valeurs max de teinte, saturation et valeur pour la couleur verte
        mask_green = cv.inRange(hsv_img, lower_green, upper_green)
        img_without_green = cv.bitwise_and(image, image, mask=~mask_green)  # Appliquer le masque
        # plt.figure() and plt.imshow(img_without_green)

        # masque des pixels non gris, seuil de gris
        img_gray = cv.cvtColor(img_without_green, cv.COLOR_BGR2GRAY)
        seuil_gris = max(np.mean(img_gray), 20)
        _, image_filtree = cv.threshold(img_gray, seuil_gris, 255, cv.THRESH_BINARY)
        # plt.figure() and plt.imshow(image_filtree)

        # Supprimer les petits objets
        labels, nb_labels, coordinates, sizes = label_objects(image_filtree)
        image_filtree2 = np.zeros_like(image_filtree)
        for label in (range(1, nb_labels + 1)):
            if sizes[label] >= seuil_small_obj * 255:
                image_filtree2[labels == label] = 255
        # plt.figure() and plt.imshow(image_filtree2)

        # Supprimer le capteur (même si en plusieurs morceaux)
        labels, nb_labels, coordinates, sizes = label_objects(image_filtree2)
        image_filtree3 = image_filtree2.copy()
        for i in (range(len(coordinates))):
            if 1200 <= coordinates[i][0] <= 2100 and 1800 <= coordinates[i][1] <= 2900:
                for j in range(i + 1, len(coordinates)):
                    if (abs(coordinates[i][0] - coordinates[j][0]) <= 300) and (
                            abs(coordinates[i][1] - coordinates[j][1]) <= 300):
                        # print(coordinates[i])
                        # print(coordinates[j])
                        # print(sizes[i])
                        # print(sizes[j])
                        if 2000 * 255 <= sizes[i] + sizes[j] <= 200000 * 255:  # capteur
                            # print(sizes[i] + sizes[j])
                            image_filtree3[labels == i + 1] = 0
                            image_filtree3[labels == j + 1] = 0
                            image_filtree3[int(coordinates[i][0] - 300): int(coordinates[i][0] + 300), int(coordinates[i][1] - 300): int(coordinates[i][1] + 300)] = 0
                            image_filtree3[int(coordinates[j][0] - 300): int(coordinates[j][0] + 300), int(coordinates[j][1] - 300): int(coordinates[j][1] + 300)] = 0
                    elif 2000 * 255 <= sizes[i] <= 200000 * 255:  # capteur
                        # print(sizes[i])
                        image_filtree3[labels == i + 1] = 0

        # Tracer des lignes entre les objets ayant les mêmes coordonnées horizontales
        image_with_lines = image_filtree3.copy()
        labels, nb_labels, coordinates, sizes = label_objects(image_filtree3)
        for i in (range(len(coordinates))):
            for j in range(i + 1, len(coordinates)):
                if abs(coordinates[i][0] - coordinates[j][0]) <= 100 and abs(
                        coordinates[i][1] - coordinates[j][1]) <= 1000:
                    cv.line(image_with_lines, (int(coordinates[i][1]), int(coordinates[i][0])),
                            (int(coordinates[j][1]), int(coordinates[j][0])), (255, 255, 255), 30)
            # Tracer une ligne si l'objet est proche du bord gauche ou droit de l'image
            '''
            if coordinates[i][1] <= 1500:
                cv.line(image_with_lines, (int(coordinates[i][1]), int(coordinates[i][0])), (0, int(coordinates[i][0])),
                        (255, 255, 255), 15)
            elif coordinates[i][1] >= width - 1500:
                cv.line(image_with_lines, (int(coordinates[i][1]), int(coordinates[i][0])),
                        (image_with_lines.shape[1] - 1, int(coordinates[i][0])), (255, 255, 255), 15)
            '''

        # Tracer des lignes entre les objets ayant les mêmes coordonnées verticales
        for i in (range(len(coordinates))):
            for j in range(i + 1, len(coordinates)):
                if abs(coordinates[i][1] - coordinates[j][1]) <= 100 and abs(
                        coordinates[i][0] - coordinates[j][0]) <= 1500:
                    cv.line(image_with_lines, (int(coordinates[i][1]), int(coordinates[i][0])),
                            (int(coordinates[j][1]), int(coordinates[j][0])), (255, 255, 255), 30)
                    # Tracer une ligne si l'objet est proche du bord supérieur ou inférieur de l'image
            '''
                if coordinates[i][0] <= 1500:
                    cv.line(image_with_lines, (int(coordinates[i][1]), int(coordinates[i][0])), (int(coordinates[i][1]), 0),
                            (255, 255, 255), 15)
                elif coordinates[i][0] >= image_with_lines.shape[0] - 1500:
                    cv.line(image_with_lines, (int(coordinates[i][1]), int(coordinates[i][0])),
                            (int(coordinates[i][1]), image_with_lines.shape[0] - 1), (255, 255, 255), 15)
            '''
        # plt.figure() and plt.imshow(image_with_lines)

        # paramettres pour la recherche des bords de bac
        largueur_min = 1600
        longueur_min = 1800
        nouvelle_longueur = 0
        nouvelle_largeur = 0
        seuil_bordure = 300 * 255
        centre_ligne = height // 2
        centre_colonne = width // 2
        nouvelle_largeur_haut = centre_ligne
        nouvelle_largeur_bas = centre_ligne
        nouvelle_longueur_gauche = centre_colonne
        nouvelle_longueur_droite = centre_colonne

        # Recherche des bords du bac
        image_with_lines = image_with_lines + image_filtree2
        while (nouvelle_longueur <= longueur_min) or (nouvelle_largeur <= largueur_min):
            if nouvelle_longueur <= longueur_min:
                ligne = centre_ligne
                for ligne in range(centre_ligne, height):
                    if np.sum(image_with_lines[ligne, 1800:2900]) > seuil_bordure:
                        # print(np.sum(image_with_lines[ligne, 1800:2900]))
                        break
                nouvelle_largeur_bas = ligne

                for ligne in range(centre_ligne, 0, -1):
                    if np.sum(image_with_lines[ligne, 1800:2900]) > seuil_bordure:
                        # print(np.sum(image_with_lines[ligne, 1800:2900]))
                        break
                nouvelle_largeur_haut = ligne
                nouvelle_longueur = nouvelle_largeur_bas - nouvelle_largeur_haut

            if nouvelle_largeur <= largueur_min:
                colonne = centre_colonne
                for colonne in range(centre_colonne, width):
                    if np.sum(image_with_lines[1200:2100, colonne]) > seuil_bordure:
                        # print(np.sum(image_with_lines[1200:2100, colonne]))
                        break
                nouvelle_longueur_droite = colonne

                for colonne in range(centre_colonne, 0, -1):
                    if np.sum(image_with_lines[1200:2100, colonne]) > seuil_bordure:
                        # print(np.sum(image_with_lines[1200:2100, colonne]))
                        break
                nouvelle_longueur_gauche = colonne
                nouvelle_largeur = nouvelle_longueur_droite - nouvelle_longueur_gauche

            # image_with_lines[nouvelle_largeur_haut:nouvelle_largeur_bas, 1800:2900] = 0
            # image_with_lines[1200:2100, nouvelle_longueur_gauche:nouvelle_longueur_droite] = 0
            image_with_lines[nouvelle_largeur_haut:nouvelle_largeur_bas, nouvelle_longueur_gauche:nouvelle_longueur_droite] = 0

        # plt.figure() and plt.imshow(image_with_lines)
        return nouvelle_largeur_haut, nouvelle_largeur_bas, nouvelle_longueur_gauche, nouvelle_longueur_droite

    # Definir la colone et la ligne centrale
    height, width, color = image1.shape
    centre_colonne = width // 2
    centre_ligne = height // 2

    # charger les contours de bac des images 1 et 2
    haut_1, bas_1, gauche_1, droite_1 = bord_bac(image1, seuil_small_obj)
    haut_2, bas_2, gauche_2, droite_2 = bord_bac(image2, seuil_small_obj)

    # afficher les contours du bac des images 1 et 2 (en rouge)
    cv.rectangle(image1, (gauche_1, haut_1), (droite_1, bas_1), (0, 0, 255), 20)
    cv.rectangle(image2, (gauche_2, haut_2), (droite_2, bas_2), (0, 0, 255), 20)

    h = min(haut_1, haut_2)
    b = max(bas_1, bas_2)
    g = min(gauche_1, gauche_2)
    d = max(droite_1, droite_2)

    # Ajouter un bord manquant

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

    return h, b, g, d, image1, image2
