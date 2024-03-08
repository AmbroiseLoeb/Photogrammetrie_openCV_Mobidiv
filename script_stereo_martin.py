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


def raccourcir_image(photo):
    hauteur, largeur = photo.shape[:2]
    # Calculer les nouvelles dimensions
    nouvelle_largeur = int(largeur * 0.75)
    gauche = int((largeur - nouvelle_largeur) / 2)
    droite = largeur - gauche
    # Recadrer l'image
    photo = photo[:, gauche:droite]

    return photo


# reconaissance du bac #---------------

# masque des pixels verts
hsv_img = cv.cvtColor(image_left, cv.COLOR_BGR2HSV)
lower_green = np.array([10, 20, 20])  # Valeurs minimales de teinte, saturation et valeur pour la couleur verte
upper_green = np.array([100, 255, 255])  # Valeurs maximales de teinte, saturation et valeur pour la couleur verte
mask_green = cv.inRange(hsv_img, lower_green, upper_green)
img_without_green = cv.bitwise_and(image_left, image_left, mask=~mask_green)  # Appliquer le masque
img_gray = cv.cvtColor(img_without_green, cv.COLOR_BGR2GRAY)  # Convertir en niveaux de gris

# seuil de gris
mean_intensity = np.mean(img_gray)
_, thresholded_img = cv.threshold(img_gray, 1.5*mean_intensity, 255, cv.THRESH_BINARY)
# thresholded_img = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 511, C = -mean_intensity)
plt.figure() and plt.imshow(thresholded_img)


# Étiqueter les objets
labels, nb_labels = ndi.label(thresholded_img)
# Calculer la taille de chaque objet
sizes = ndi.sum(thresholded_img, labels, range(nb_labels + 1))
# Supprimer les objets inférieurs à 50 pixels
filtered_image = np.zeros_like(thresholded_img)
for label in range(1, nb_labels + 1):
    if sizes[label] >= 300 * 255 :
        filtered_image[labels == label] = 255
plt.figure() and plt.imshow(filtered_image)


# Recherche des bords de bac
mean_center = np.mean(thresholded_img[1200:2400, 1500:3000])
mean_tot = np.mean(thresholded_img)
height, width = thresholded_img.shape
centre_colonne = width // 2
centre_ligne = height // 2

for colonne in range(centre_colonne + 100, width):
    if np.sum(thresholded_img[1200:2400:, colonne]) > 30000 and np.mean(thresholded_img[1200:2400:, colonne]) > 1.5*mean_tot:
        break
nouvelle_largeur_droite = colonne

for colonne in range(centre_colonne - 100, -1, -1):
    if np.sum(thresholded_img[1200:2400:, colonne]) > 30000 and np.mean(thresholded_img[1200:2400:, colonne]) > 1.5*mean_tot:
        break
nouvelle_largeur_gauche = colonne

for ligne in range(centre_ligne + 100, height):
    if np.sum(thresholded_img[ligne, 1500:3000]) > 30000 and np.mean(thresholded_img[ligne, 1500:3000]) > 1.5*mean_tot:
        break
nouvelle_hauteur_bas = ligne

for ligne in range(centre_ligne - 100, -1, -1):
    if np.sum(thresholded_img[ligne, 1500:3000]) > 30000 and np.mean(thresholded_img[ligne, 1500:3000]) > 1.5*mean_tot:
        break
nouvelle_hauteur_haut = ligne




'''
# blurred = cv.GaussianBlur(thresholded_img, (5, 5), 0)
# Détection des contours
edges = cv.Canny(thresholded_img, 0, 0, L2gradient=True)

# Détection des lignes à l'aide de la transformée de Hough
lines = cv.HoughLinesP(thresholded_img, 1, np.pi / 180, threshold=0, minLineLength=500, maxLineGap=20)

# Dessiner les lignes détectées sur une copie de l'image originale
lines_img = image_left.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
plt.figure() and plt.imshow(lines_img)
'''

# conversion en niveaux de gris
# img_l = cv.cvtColor(image_left, cv.COLOR_BGR2GRAY)
img_l = cv.cvtColor(image_left, cv.IMREAD_GRAYSCALE + cv.IMREAD_IGNORE_ORIENTATION)
img_r = cv.cvtColor(image_right, cv.IMREAD_GRAYSCALE + cv.IMREAD_IGNORE_ORIENTATION)
rgb_l = cv.cvtColor(image_left, cv.COLOR_BGR2RGB)

# rectification des images (transformation de perspective)
imglCalRect = cv.remap(img_l, mapx11, mapx12, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
imgrCalRect = cv.remap(img_r, mapx21, mapx22, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
rgblCalRect = cv.remap(rgb_l, mapx11, mapx12, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

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
plt.figure()
plt.imshow(rgblCalRect)
plt.figure()
plt.imshow(disparity, cmap='jet', vmin=np.nanquantile(disparity, 0.005), vmax=np.nanquantile(disparity, 0.995))

# calcul et affichage de la carte de profondeur
Z = abs(FL * B / disparity)
plt.figure()
plt.imshow(Z, cmap='jet', vmin=800, vmax=1500)
plt.colorbar()

# calcul et affichage de la carte de profondeur 2
xyz_image = cv.reprojectImageTo3D(disparity, Q)
x_image, y_image, z_image = cv.split(xyz_image)
#mask_distance = z_image > 1400  # Masque en fonction de la distance
#z_image[mask_distance] = np.nan
#mask_distance2 = z_image < 600
#z_image[mask_distance2] = np.nan

# Extraire la région du bac de l'image seuillée
image_cut = np.zeros_like(thresholded_img, dtype = 'float32')
#image_cut = cv.cvtColor(zeros_img, cv.COLOR_GRAY2BGR)
image_cut[nouvelle_hauteur_haut:nouvelle_hauteur_bas, nouvelle_largeur_gauche:nouvelle_largeur_droite] = z_image[nouvelle_hauteur_haut:nouvelle_hauteur_bas, nouvelle_largeur_gauche:nouvelle_largeur_droite]
plt.figure() and plt.imshow(image_cut, cmap='jet', vmin=800, vmax=1500)



# racourcir image
# z_image = raccourcir_image(z_image)

plt.figure()
plt.imshow(z_image, cmap='jet', vmin=800, vmax=1500)
plt.colorbar()
