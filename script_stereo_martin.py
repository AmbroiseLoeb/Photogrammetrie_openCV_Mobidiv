        # importation des bibliotheques
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

        # chargement des parametres stereo
# stereo_path = r"C:\Users\loeb\PycharmProjects\stereo_imagerie\calibration"
stereo_path = r"C:\Users\loeb\PycharmProjects\stereo_imagerie\calibration"
Q = np.load(stereo_path + rf"\Q.npy") # Q.npy absent
FL = np.load(stereo_path + r"\P1.npy")[0][0]
T = np.load(stereo_path + r"\T.npy")
B = np.linalg.norm(T)
mapx11 = np.load(stereo_path + r"\mapx11.npy")
mapx12 = np.load(stereo_path + r"\mapx12.npy")
mapx21 = np.load(stereo_path + r"\mapx21.npy")
mapx22 = np.load(stereo_path + r"\mapx22.npy")

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
# left_path = r"C:\Users\loeb\PycharmProjects\stereo_imagerie\uplot_100_1\uplot_100_camera_1_1_RGB.jpg"
left_path=r"C:\Users\loeb\PycharmProjects\stereo_imagerie\uplot_100_1\uplot_100_camera_1_1_RGB.jpg"
id_image = left_path.split('camera_1')
right_path = 'camera_2'.join(id_image)

        # rotation et conversion en niveaux de gris
img_l = cv.imread(left_path, cv.IMREAD_GRAYSCALE + cv.IMREAD_IGNORE_ORIENTATION)
img_l = cv.rotate(img_l, cv.ROTATE_90_CLOCKWISE)
img_r = cv.imread(right_path, cv.IMREAD_GRAYSCALE + cv.IMREAD_IGNORE_ORIENTATION)
img_r = cv.rotate(img_r, cv.ROTATE_90_CLOCKWISE)
rgb_l = cv.cvtColor(cv.imread(left_path, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB)
rgb_l = cv.rotate(rgb_l, cv.ROTATE_90_CLOCKWISE)

        # rectification des images (transformation de perspective)
imglCalRect = cv.remap(img_l, mapx11, mapx12, cv.INTER_LINEAR, borderValue=np.nan)
imgrCalRect = cv.remap(img_r, mapx21, mapx22, cv.INTER_LINEAR, borderValue=np.nan)
rgblCalRect = cv.remap(rgb_l, mapx11, mapx12, cv.INTER_LINEAR, borderValue=np.nan)

        # reduction taille des images
h_ori, w_ori = imglCalRect.shape
isubsampling = 2
imglCalRect = cv.resize(imglCalRect, (round(w_ori / isubsampling), round(h_ori / isubsampling)), interpolation=cv.INTER_AREA)
imgrCalRect = cv.resize(imgrCalRect, (round(w_ori / isubsampling), round(h_ori / isubsampling)), interpolation=cv.INTER_AREA)

        # configuration de StereoSGBM (Stereo Semi-Global Block Matching ?)
blockSize = 5
stereo = cv.StereoSGBM_create(minDisparity=round(min_disp / isubsampling),
                              numDisparities=round(numDisparities / isubsampling),
                              uniquenessRatio=1,
                              # preFilterCap=50,
                              # disp12MaxDiff=1,
                              # disp12MaxDiff=10,
                              P1=2 * blockSize ** 2,
                              P2=32 * blockSize ** 2,
                              mode=cv.StereoSGBM_MODE_HH4,
                              # speckleWindowSize=100,
                              # speckleRange=2,
                              blockSize=blockSize
                              )


        # calcul de la carte de disparite
disparity = stereo.compute(imglCalRect, imgrCalRect).astype(np.float32) / 16
disparity = cv.resize(disparity * isubsampling, (w_ori, h_ori), interpolation=cv.INTER_AREA)

        # affichage des résultats (avec Matplotlib)
plt.figure()
plt.imshow(rgblCalRect)
plt.figure()
plt.imshow(disparity, cmap='jet', vmin=np.nanquantile(disparity, 0.005), vmax=np.nanquantile(disparity, 0.995))

        # calcul et affichage de la carte de profondeur
Z = abs(FL * B / disparity)
Z = cv.rotate(Z, cv.ROTATE_90_COUNTERCLOCKWISE)
plt.figure()
plt.imshow(Z, cmap='jet', vmin=500, vmax=2000)
plt.colorbar()
plt.figure()
plt.imshow(cv.rotate(rgblCalRect, cv.ROTATE_90_COUNTERCLOCKWISE))

        # affichage de la carte de disparite
# Masking based on distance
xyz_image = cv.reprojectImageTo3D(disparity, Q)
x_image, y_image, z_image = cv.split(xyz_image)
plt.figure()
plt.imshow(z_image, cmap='jet', vmin=np.nanquantile(z_image, 0.01), vmax=np.nanquantile(z_image, 0.9))

mask_distance = z_image > 3000
disparity[mask_distance] = np.nan
plt.figure()
plt.imshow(disparity, cmap='jet', vmin=np.nanquantile(disparity, 0.005), vmax=np.nanquantile(disparity, 0.995))
