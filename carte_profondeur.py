# importation des bibliotheques
from pathlib import Path
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def parametres_stereo():

    """ Chargement des parametres stereo """

    current_dir = os.path.dirname(os.path.abspath(__name__))  # Chemin absolu du répertoire où se trouve le script
    calibration_dir = os.path.join(current_dir, 'calibration')  # Dossier contenant les fichiers de rectification (calibration)

    # Fichiers de rectification
    Q = np.load(calibration_dir + f"/Q.npy")
    FL = np.load(calibration_dir + "/P1.npy")[0][0]
    T = np.load(calibration_dir + "/T.npy")
    B = np.linalg.norm(T)
    mapx11 = np.load(calibration_dir + "/mapx11.npy")
    mapx12 = np.load(calibration_dir + "/mapx12.npy")
    mapx21 = np.load(calibration_dir + "/mapx21.npy")
    mapx22 = np.load(calibration_dir + "/mapx22.npy")

    # Definition des parametres de disparite
    Dmax = 100 * 1000
    Dmin = .5 * 1000
    blockSize = 10
    MinDisp = int(np.floor(FL * B / Dmax))
    MaxDisp = int(np.ceil(FL * B / Dmin))
    numDisparities = MaxDisp - MinDisp
    if T[np.argmax(abs(T))] > 0:
        min_disp = - MaxDisp
    else:
        min_disp = MinDisp

    return Q, blockSize, mapx11, mapx12, mapx21, mapx22, numDisparities, min_disp


def workflow_carte_profondeur(image1, image2):
    """ Cretation d'une carte de profondeur a partir de deux images. """

    # parametres stereo
    Q, blockSize, mapx11, mapx12, mapx21, mapx22, numDisparities, min_disp = parametres_stereo()

    # conversion en niveaux de gris
    img_l = cv.cvtColor(image1, cv.IMREAD_GRAYSCALE + cv.IMREAD_IGNORE_ORIENTATION)
    img_r = cv.cvtColor(image2, cv.IMREAD_GRAYSCALE + cv.IMREAD_IGNORE_ORIENTATION)

    # rectification des images (transformation de perspective)
    imglCalRect = cv.remap(img_l, mapx11, mapx12, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    imgrCalRect = cv.remap(img_r, mapx21, mapx22, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    # reduction de la taille des images
    h_ori, w_ori = imglCalRect.shape
    isubsampling = 2
    imglCalRect = cv.resize(imglCalRect, (round(w_ori / isubsampling), round(h_ori / isubsampling)),
                            interpolation=cv.INTER_AREA)
    imgrCalRect = cv.resize(imgrCalRect, (round(w_ori / isubsampling), round(h_ori / isubsampling)),
                            interpolation=cv.INTER_AREA)

    # configuration de StereoSGBM (Semi-Global Block Matching)
    stereo = cv.StereoSGBM.create(minDisparity=round(min_disp / isubsampling),
                                  numDisparities=round(numDisparities / isubsampling),
                                  blockSize=blockSize,
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

    # calcul de la carte de profondeur
    xyz_image = cv.reprojectImageTo3D(disparity, Q)
    x_image, y_image, z_image = cv.split(xyz_image)
    # z_image = abs(FL * B / disparity)

    return z_image


# plt.figure() and plt.imshow(disparity, cmap='jet', vmin=np.nanquantile(disparity, 0.005), vmax=np.nanquantile(disparity, 0.995)) and plt.colorbar()
# plt.figure() and plt.imshow(z_image, cmap='jet', vmin=800, vmax=2000)
