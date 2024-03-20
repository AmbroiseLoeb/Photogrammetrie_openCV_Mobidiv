# importation des bibliotheques
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.ndimage as ndi

'''matplotlib.use('TkAgg')  # pour afficher les plt en popup'''

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

# chargement des images gauche et droite
left_path = ("/home/loeb/Documents/Literal_mobidiv_2023/Session 2023-06-22 06-32-00"
             "/uplot_100_1/uplot_100_camera_1_2_RGB.jpg")
id_image = left_path.split('camera_1')
right_path = 'camera_2'.join(id_image)

image_left = cv.imread(left_path)
image_right = cv.imread(right_path)


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
    filtered_image = np.zeros_like(thresholded_img)
    seuil2 = 0

    for label in range(1, nb_labels + 1):
        if 1200 <= coordinates[label][0] <= 2500 and 1500 <= coordinates[label][1] <= 3200:  # capteur
            if 20000 * 255 <= sizes[label] <= 60000 * 255:
                img_sans_capteur[labels == label] = (0, 0, 0, 0)
                seuil2 = 100000  # augmentation du seuil en cas de présence du capteur
        elif sizes[label] >= 300 * 255:
            filtered_image[labels == label] = 255
    # plt.figure() and plt.imshow(filtered_image)

    # paramettres pour recherche des bords de bac
    largueur_min = 1600
    longueur_min = 1800
    nouvelle_longueur = 0
    nouvelle_largeur = 0
    seuil_bordure = 20000
    nouvelle_largeur_haut = centre_ligne
    nouvelle_largeur_bas = centre_ligne
    nouvelle_longueur_gauche = centre_colonne
    nouvelle_longueur_droite = centre_colonne

    # Recherche des bords de bac
    while nouvelle_longueur <= longueur_min or nouvelle_largeur <= largueur_min:
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
                if np.sum(filtered_image[ligne, 1500:3200]) > max(seuil_bordure, seuil2):
                    break
            nouvelle_largeur_bas = ligne

            for ligne in range(centre_ligne - 100, -1, -1):
                if np.sum(filtered_image[ligne, 1500:3200]) > max(seuil_bordure, seuil2):
                    break
            nouvelle_largeur_haut = ligne
            nouvelle_longueur = nouvelle_largeur_bas - nouvelle_largeur_haut

        filtered_image[nouvelle_largeur_haut:nouvelle_largeur_bas, nouvelle_longueur_gauche:nouvelle_longueur_droite] = 0

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
    h = max(haut_left, haut_right)
    b = min(bas_left, bas_right)
    g = max(gauche_left, gauche_right)
    d = min(droite_left, droite_right)

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


# carte de profondeur, avec suppression du capteur
depth_image1 = carte_profondeur(bord_bac(image_left)[4], image_right)
depth_image2 = carte_profondeur(image_left, bord_bac(image_right)[4])
depth_image = (depth_image1 * depth_image2) / depth_image1

plt.figure() and plt.imshow(depth_image, cmap='jet', vmin=800, vmax=1500)

# Extraire la region du bac
haut, bas, gauche, droite = contour_bac(image_left, image_right)
image_cut = np.zeros_like(depth_image, dtype='float32')
image_cut[haut:bas, gauche:droite] = depth_image[haut:bas, gauche:droite]

plt.figure() and plt.imshow(image_cut, cmap='jet', vmin=800, vmax=1500)
plt.figure() and plt.imshow(image_left)
