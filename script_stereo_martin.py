# importation des bibliotheques
import numpy as np
import cv2 as cv
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')  # pour afficher les plt en popup

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
left_path = ("/home/loeb/Documents/Literal_mobidiv_2023/Session 2023-02-06 "
             "13-14-02/uplot_100_1/uplot_100_camera_1_2_RGB.jpg")
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


# reconaissance des contours du bac


# conversion en niveaux de gris
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
mask_distance = z_image > 1400  # Masque en fonction de la distance
z_image[mask_distance] = np.nan
mask_distance2 = z_image < 600
z_image[mask_distance2] = np.nan


# racourcir image
z_image = raccourcir_image(z_image)

plt.figure()
plt.imshow(z_image, cmap='jet', vmin=800, vmax=1500)
plt.colorbar()
