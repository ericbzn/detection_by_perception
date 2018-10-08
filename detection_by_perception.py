#!/usr/bin/python  
'''
<detectection_by_perception.py This code uses the human perception principles to detect and recognize landing targets for a UAV application>
Copyright (C) <2018>  <Eric Bazan>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.


You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os
import numpy as np
import cv2
import scipy.stats as st
import time
import pygame
import pygame.camera
import pdb

from PIL import Image
from pygame.locals import *
from scipy.ndimage.filters import gaussian_laplace, gaussian_gradient_magnitude
from spectral import rx
from sklearn.cluster import AffinityPropagation, MeanShift, estimate_bandwidth, AgglomerativeClustering


def img_preparation(img):
    global img_color

    if len(img.shape) == 3:
        img_color = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_color = cv2.ctvColor(img, cv2.GRAY2BGR)
    noise = np.random.normal(0, noise_std, img.shape)
    img = img + noise
    return img


def find_cnts(img):
    """ Extract contours of the image (Marr-Hildreth with diff sigma)"""
    global cnts
    cnts = np.array([])
    # cnts = []
    # contours = []
    for s in sigma:
        l = gaussian_laplace(img, sigma=s, mode='constant')
        l = np.array(Image.fromarray(l).resize(scale * np.array(np.transpose(img).shape)))
        lp = np.array(l > 0, np.uint8)
        _, cnts_ii, hierarchy_ii = cv2.findContours(lp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts = np.append(cnts, cnts_ii)
        # cnts += cnts_ii
        # contours.append([cnts_ii, hierarchy_ii])
    # C = np.array([np.ravel(c).reshape([len(c), 2]) for c in cnts])
    return cnts


def biuld_multispace(cnts):
    """Extract some contours features (mean_int & circularity) and return a multivariable space"""
    # circularity = [4 * np.pi * np.divide(cv2.contourArea(c), cv2.arcLength(c, True) ** 2) for c in cnts]
    # mean_int = [np.mean(gg[np.ravel(c).reshape([len(c), 2])[:, 1], np.ravel(c).reshape([len(c), 2])[:, 0]]) for c in cnts]

    global circularity, cnts_area, approx
    circularity = np.zeros(len(cnts))
    mean_int = np.zeros(len(cnts))
    cnts_area = np.zeros(len(cnts))
    cnts_perim = np.zeros(len(cnts))
    approx = np.zeros(len(cnts))
    gg = gaussian_gradient_magnitude(img, sigma=1)
    gg = np.array(Image.fromarray(gg).resize(scale * np.array(np.transpose(img).shape)))
    for ii in range(len(cnts)):
        cnts_area[ii] = cv2.contourArea(cnts[ii])
        cnts_perim[ii] = cv2.arcLength(cnts[ii], True)
        approx[ii] = len(cv2.approxPolyDP(cnts[ii], 0.01*cnts_perim[ii], True))
        circularity[ii] = 4 * np.pi * np.divide(cnts_area[ii], cnts_perim[ii] ** 2)
        coor = cnts[ii].reshape((len(cnts[ii]), 2))
        mean_int[ii] = np.mean(gg[np.array(coor[:, 1]), np.array(coor[:, 0])])
    features = np.column_stack((mean_int, circularity))  # Mean intensity, circularity
    multi_space = np.reshape(features, (len(cnts), 1, features.shape[1]))
    return multi_space


def rx_detector(multispace):
    """Apply RX anomaly detector in the multispace to find the outliers"""

    global rx_labels
    rxval = rx(multispace)
    # Find the critical value (P) for 90% confidence.
    # Choose the threshold at .001 probability of occurrence in the background
    p = st.chi2.ppf(chi_squared, multispace.shape[2])
    rx_labels = []

    for ii in range(len(multispace)):
        if rxval[ii] > p and approx[ii] > 8:
            rx_labels.append(ii)
    img_rxcnts = np.ones((480*3, 640*3), dtype=np.uint8) * 255
    cv2.drawContours(img_rxcnts, np.take(cnts, rx_labels), -1, 0, 2)
    Img = Image.fromarray(img_rxcnts)
    # Img.show()
    return rx_labels


def dist_ellipse(cnts_points, f):
    return np.sqrt(np.sum(np.square(cnts_points - f), 1))


def grouping_features(cnts, rx_labels):
    """Compute Goodness index of a fitted ellipse per contour and the spacial position of contour centroids"""

    global H_mean, Cx, Cy, MA
    H_mean = np.zeros(len(rx_labels), dtype=float)
    Cx = np.zeros(len(rx_labels))
    Cy = np.zeros(len(rx_labels))
    MA = np.zeros(len(rx_labels))
    ma = np.zeros(len(rx_labels))
    angle = np.zeros(len(rx_labels))
    distance = np.zeros(len(rx_labels))
    rx_cnts = np.take(cnts, rx_labels)
    rx_area = np.take(cnts_area, rx_labels)
    for ii in range(len(rx_labels)):
        (Cx[ii], Cy[ii]), (ma[ii], MA[ii]), angle[ii] = cv2.fitEllipse(
            rx_cnts[ii])  # ellipse = [(Cx, Cy), (ma, MA), angle]
    ellipse_area = np.pi * MA * ma / 4
    diff_area = 1 - np.abs(rx_area - ellipse_area) / np.amax(np.column_stack((rx_area, ellipse_area)), axis=1)

    ## Goodness index
    # Focal linear eccentricity
    c = np.sqrt((MA / 2) ** 2 - (ma / 2) ** 2)
    cy = c * np.cos(np.pi / 180 * angle)
    cx = c * np.sin(np.pi / 180 * angle)

    # Focus coordinates F1 and F2
    F1 = np.array((Cx + cx, Cy - cy))
    F2 = np.array((Cx - cx, Cy + cy))

    for ii in range(len(rx_labels)):
        dd = np.abs(dist_ellipse(rx_cnts[ii], F1.T[ii]) + dist_ellipse(rx_cnts[ii], F2.T[ii]) - MA[ii])
        distance[ii] = dd.sum() / len(dd)
    goodness_indx = np.exp(-distance ** 2 / (2 * 1000 ** 2))  # sigma = 1000
    H_mean = 2 / (1. / goodness_indx + 1. / diff_area)  # st.hmean([goodness_indx, diff_area])
    X = np.column_stack((Cx / Cx.max(), Cy / Cy.max(), np.take(circularity, rx_labels), H_mean))  # , Jccrd_indx))
    return X


def affinity_clustering(features):
    """Compute clustering by Affinity Propagation between rx contours"""

    af = AffinityPropagation(preference=-0.02, damping=0.7).fit(
        features)  # -0.02/-0.03 and 0.8/0.7  -> good values for real images  0.7 01/06/2018/
    n_clusters = len(af.cluster_centers_)
    cluster_labels = af.labels_
    cluster_elements = []
    mean_similarity = []
    perceptive_clusters = []

    for ii in range(n_clusters):
        cluster_elements.append(np.asarray(np.where(cluster_labels == ii), dtype=np.int).ravel())
        mean_similarity.append((np.mean(np.take(H_mean, cluster_elements[ii]))))
        cluster_elements_centers = np.column_stack(
            (np.take(Cx, cluster_elements[ii]) / Cx.max(), np.take(Cy, cluster_elements[ii]) / Cy.max()))
        dist = dist_ellipse(cluster_elements_centers, (af.cluster_centers_[ii][0], af.cluster_centers_[ii][1]))
        dist = dist.sum() / len(dist)
        center_proximity_indx = np.exp(-dist ** 2 / (2 * 0.1 ** 2))  # sigma = 0.1
        if len(cluster_elements[ii]) >= 15 and mean_similarity[ii] >= 0.9 and center_proximity_indx >= 0.9:
            perceptive_clusters.append(cluster_elements[ii])
    return perceptive_clusters


def meanshift_clustering(features):
    """Compute clustering by Meanshift between rx contours"""

    bandwidth = estimate_bandwidth(features, quantile=0.02)  # quantile=0.015, 0.02   Real images,   0.9 synthetic images
    if bandwidth == 0:
        bandwidth = 0.01
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(features)
    n_clusters = len(ms.cluster_centers_)
    cluster_labels = ms.labels_
    cluster_elements = []
    mean_similarity = []
    perceptive_clusters = []

    for ii in range(n_clusters):
        cluster_elements.append(np.asarray(np.where(cluster_labels == ii), dtype=np.int).ravel())
        mean_similarity.append((np.mean(np.take(H_mean, cluster_elements[ii]))))
        cluster_elements_centers = np.column_stack(
            (np.take(Cx, cluster_elements[ii]) / Cx.max(), np.take(Cy, cluster_elements[ii]) / Cy.max()))
        dist = np.abs(dist_ellipse(cluster_elements_centers, (ms.cluster_centers_[ii][0], ms.cluster_centers_[ii][1])))
        dist = dist.sum() / len(dist)
        center_proximity_indx = np.exp(-dist ** 2 / (2 * 0.1 ** 2))  # sigma = 0.1
        if len(cluster_elements[ii]) >= 15 and mean_similarity[ii] >= 0.9 and center_proximity_indx >= 0.9:
            perceptive_clusters.append(cluster_elements[ii])
    return perceptive_clusters


def bin2dec(msg):
    msg_dec = np.right_shift(np.packbits(msg[0:5], axis=-1), 4).squeeze()
    return msg_dec


def hamming_syndrome(codeword):
    S = np.dot(codeword, H.transpose().astype(int)) % 2
    return np.array(S).ravel()


def error_correction(wrong_cw, pos):
    wrong_cw[pos] = wrong_cw[pos] ^ 1
    corrected_codeword = wrong_cw
    return corrected_codeword


def findError_raw(codeword):
    Pb_dec = bin2dec(np.dot(codeword, Rr).astype(int))
    msg_dec = bin2dec(codeword)
    if Pb_dec == ctrl_dec[msg_dec]:
        return msg_dec
    else:
        return None


def findError_hamming(codeword):
    syndrome = hamming_syndrome(codeword)
    if np.sum(syndrome) == 0:
        msg_dec = bin2dec(np.dot(codeword, R).astype(int))
        return msg_dec
    else:
        err = np.sum((H.transpose().astype(int) - syndrome) % 2, axis=1)
        pos = np.array(np.where(err == 0))
        if pos.size == 0:
            return None
        else:
            pos = int(pos[0])
            corrected_codeword = error_correction(codeword, pos)
            msg_dec = bin2dec(corrected_codeword)
            return msg_dec


def cluster_decoding(perceptive_clusters):
    M_Axes = []
    id_targets_raw = []
    id_targets_hamming = []
    img_cluster = np.ones((480 * 3, 640 * 3), dtype=np.uint8) * 255
    for ii in range(len(perceptive_clusters)):
        cv2.drawContours(img_cluster, np.take(cnts, np.take(rx_labels, perceptive_clusters[ii])), -1, 0, 2)
        M_Axes.append(np.take(MA, perceptive_clusters[ii]))
        MAx_cluster = AgglomerativeClustering(linkage='ward', n_clusters=10)
        MAx_cluster.fit(np.column_stack((M_Axes[ii], np.zeros(len(M_Axes[ii])))))
        MAx_labels = MAx_cluster.labels_
        cnts_same_diameter = []
        mean_MAxe = []
        for jj in range(10):
            cnts_same_diameter.append(np.asarray(np.where(MAx_labels == jj), dtype=np.int).ravel())
            mean_MAxe.append(-np.mean(np.take(M_Axes[ii], cnts_same_diameter[jj])))
        diameter = -np.sort(mean_MAxe)
        diameter = (diameter / diameter[0]) * DtcMax
        msg_rec = diameter[2:10] - dMsgC
        msg_rec = msg_rec / np.abs(msg_rec)
        msg_rec[msg_rec < 0] = 0
        id_targets_raw.append(findError_raw(np.array(msg_rec, dtype=int)))
        id_targets_hamming.append(findError_hamming(np.array(msg_rec, dtype=int)))
    Img = Image.fromarray(img_cluster)
    # Img.show()
    return np.array(id_targets_hamming), np.array(id_targets_raw)


def landing_target_detection(img):
    time_find_cnts = time.time()
    cnts = find_cnts(img)
    # print 'no_cnts:', len(cnts)
    print 'find_cnts Execution time: ', time.time() - time_find_cnts

    time_multispace = time.time()
    multispace = biuld_multispace(cnts)
    print 'multifeature_space Execution time: ', time.time() - time_multispace

    time_rx_detector = time.time()
    rx_labels = rx_detector(multispace)
    print 'rx_cnts:', len(rx_labels)
    print 'rx_detector Execution time: ', time.time() - time_rx_detector

    time_grouping_features = time.time()
    features = grouping_features(cnts, rx_labels)
    print 'cluster_features Execution time: ', time.time() - time_grouping_features

    time_affinity_clustering = time.time()
    perceptive_clusters = affinity_clustering(features)
    print 'af_clusters:', len(perceptive_clusters)
    print 'affinity_clustering Execution time: ', time.time() - time_affinity_clustering

    # time_meanshift_clustering = time.time()
    # perceptive_clusters = meanshift_clustering(features)
    # # print 'ms_clusters:', len(perceptive_clusters)
    # print 'meanshift_clustering Execution time: ', time.time() - time_meanshift_clustering

    time_cluster_decoding = time.time()
    id_targets, id_targets_raw = cluster_decoding(perceptive_clusters)
    print 'ID targets found:', id_targets
    print 'cluster_decoding Execution time: ', time.time() - time_cluster_decoding
    return perceptive_clusters, id_targets, id_targets_raw


def save_show_result_img(perceptive_clusters, id_targets, name):
    outdir = 'outdir/images'
    img_result_color = np.array(Image.fromarray(img_color).resize(scale * np.array(np.transpose(img).shape)))

    for ii in range(len(perceptive_clusters)):
        indx = np.argmax(np.take(MA, perceptive_clusters[ii]))
        x, y, w, h = cv2.boundingRect(np.take(cnts, np.take(rx_labels, perceptive_clusters[ii][indx])))
        cv2.rectangle(img_result_color, (x, y), (x + w, y + h), (0, 0, 255), 5)
        cv2.putText(img_result_color, str(id_targets[ii]), (x, y - 5 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                    3.5, (0, 0, 255), 10, cv2.LINE_AA)

    cv2.resize(img_result_color, (img_result_color.shape[0] / scale, img_result_color.shape[1] / scale))
    cv2.imwrite(os.path.join(outdir, str(name)), img_result_color)


def src_image():
    indir = 'indir/images'
    in_imgs = os.listdir(indir)

    for im_file in in_imgs:
        time_total = time.time()

        print '##############################', im_file, '##############################'
        global img
        img = img_preparation(cv2.imread(os.path.join(indir, im_file), cv2.CV_8S))

        perceptive_clusters, id_targets, id_targets_raw = landing_target_detection(img)
        save_show_result_img(perceptive_clusters, id_targets, im_file)

        print 'TOTAL EXECUTION TIME: ', time.time() - time_total


def show_result_video(perceptive_clusters, id_targets):
    img_result_color = np.array(Image.fromarray(img_color).resize(scale * np.array(np.transpose(img).shape)))
    for ii in range(len(perceptive_clusters)):
        indx = np.argmax(np.take(MA, perceptive_clusters[ii]))
        x, y, w, h = cv2.boundingRect(np.take(cnts, np.take(rx_labels, perceptive_clusters[ii][indx])))
        cv2.rectangle(img_result_color, (x, y), (x + w, y + h), (0, 0, 255), 5)
        cv2.putText(img_result_color, str(id_targets[ii]), (x, y - 5 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                    3.5, (0, 0, 255), 10, cv2.LINE_AA)
    return np.array(
        Image.fromarray(img_result_color).resize((img_result_color.shape[1] / scale, img_result_color.shape[0] / scale),
                                                 Image.ANTIALIAS))


def src_video_file():
    indir = 'indir/video'
    outdir = 'outdir/video'
    # in_video = os.listdir(indir)

    # for video_file in in_video:
#     print '##############################', video_file, '##############################'
    video_file = 'indoor_2.mp4'


    cap = cv2.VideoCapture(os.path.join(indir, video_file))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(outdir, video_file), fourcc, 30.0, (640, 480))
    # time_total = time.time()
    while (cap.isOpened()):
        print 'running'
        global img
        ret, img = cap.read()
        img = img_preparation(img)
        perceptive_clusters, id_targets, id_targets_raw = landing_target_detection(img)
        img_result = show_result_video(perceptive_clusters, id_targets)

        ########################################################
        out.write(img_result)
        # cv2.imshow('frame', img_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # print 'TOTAL EXECUTION TIME: ', time.time() - time_total


def src_video():
    DEVICE = '/dev/video0'  # 1 for build in camera    0 for others cameras
    SIZE = (640, 480)
    NUMBER_PROCESSES = 8
    T = 1

    pygame.init()
    pygame.camera.init()
    display = pygame.display.set_mode(SIZE, 0)
    camera = pygame.camera.Camera(DEVICE, SIZE)
    camera.start()
    capture = True
    while capture:
        global img
        screen = camera.get_image()
        img = pygame.image.tostring(screen, "RGB", False)
        img = np.array(Image.frombytes("RGB", SIZE, img))
        img = img_preparation(img)
        perceptive_clusters, id_targets, id_targets_raw = landing_target_detection(img)
        img_result = show_result_video(perceptive_clusters, id_targets)

        img = np.swapaxes(img_result, 0, 1)
        dis = pygame.surfarray.make_surface(img)

        pygame.display.flip()
        display.blit(dis, (0, 0))

        # Kills the program when click on the close window button (Doesn't work in toshiba PC)

        for event in pygame.event.get():
            if event.type == QUIT:
                capture = False

    camera.stop()
    pygame.quit()
    return

def f1_test(img_synthetic):
    global img
    noise = np.random.normal(0, noise_std, img_synthetic.shape)
    img = img_synthetic + noise
    perceptive_clusters, id_targets, id_targets_raw = landing_target_detection(img)

    return id_targets, id_targets_raw



# Parameters for multiscale contours and RX detector
global noise_std, scale, sigma
noise_std = 10
scale = 3
sigma = [3, 2, 1]
chi_squared = 0.9  # 0.9999999999 for f1-score test  , 0.9 for real images

# Parameters for Hamming code and decode
global G, H, R, Rr, ctrl_dec
G = np.concatenate((np.identity(4), 1 - np.identity(4)), axis=1)
H = np.concatenate((1 - np.identity(4), np.identity(4)), axis=1)
R = np.concatenate((np.identity(4), np.zeros((4, 4))), axis=1).T
Rr = np.concatenate((np.zeros((4, 4)), np.identity(4)), axis=1).T
ctrl_dec = np.array([0, 14, 13, 3, 11, 5, 6, 8, 7, 9, 10, 4, 12, 2, 1, 15], dtype=int)

# Parameters of target circle pattern
global res, NCircles, DmcMax, DtcMax, dMsgC
res = 1  # 10000  # resolution of the area to draw the contours
NCircles = 8
DmcMax = 330 * res
DtcMax = 470 * res
dMsgC = np.linspace(DmcMax, DmcMax / 10, NCircles)

if __name__ == '__main__':
    np.random.seed(1)
    src_image()
    # src_video_file()
    # src_video()
