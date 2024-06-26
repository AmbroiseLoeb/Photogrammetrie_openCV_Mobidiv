# importation des bibliotheques
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def parametres_stereo():
    """Charge les paramètres de rectification stéréo et calcule les paramètres de disparité."""

    # Trouver le dossier 'calibration'
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

    # Calcul des parametres de disparite
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
    """Génère une carte de profondeur à partir de deux images."""

    # Récuperation des parametres stereo
    Q, blockSize, mapx11, mapx12, mapx21, mapx22, numDisparities, min_disp = parametres_stereo()

    # Conversion des images en niveaux de gris
    img_l = cv.cvtColor(image1, cv.IMREAD_GRAYSCALE + cv.IMREAD_IGNORE_ORIENTATION)
    img_r = cv.cvtColor(image2, cv.IMREAD_GRAYSCALE + cv.IMREAD_IGNORE_ORIENTATION)

    # Rectification des images (transformation de perspective)
    imglCalRect = cv.remap(img_l, mapx11, mapx12, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    imgrCalRect = cv.remap(img_r, mapx21, mapx22, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    # Ajustement des images à leur taille initiale
    h_ori, w_ori = imglCalRect.shape
    isubsampling = 2
    imglCalRect = cv.resize(imglCalRect, (round(w_ori / isubsampling), round(h_ori / isubsampling)),
                            interpolation=cv.INTER_AREA)
    imgrCalRect = cv.resize(imgrCalRect, (round(w_ori / isubsampling), round(h_ori / isubsampling)),
                            interpolation=cv.INTER_AREA)

    # Configuration de StereoSGBM (Semi-Global Block Matching)
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

    # Calcul de la carte de disparité
    disparity = stereo.compute(imglCalRect, imgrCalRect).astype(np.float32) / 16
    disparity = cv.resize(disparity * isubsampling, (w_ori, h_ori), interpolation=cv.INTER_AREA)

    # Calcul de la carte de profondeur
    xyz_image = cv.reprojectImageTo3D(disparity, Q)
    x_image, y_image, z_image = cv.split(xyz_image)
    # z_image = abs(FL * B / disparity)

    return z_image

