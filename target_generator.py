import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools
import pdb

from scipy import fftpack
from PIL import Image
from scipy.signal import argrelextrema

# Parameters for Hamming code and decode
G = np.concatenate((np.identity(4), 1 - np.identity(4)), axis=1)
H = np.concatenate((1 - np.identity(4), np.identity(4)), axis=1)
R = np.concatenate((np.identity(4), np.zeros((4, 4))), axis=1).T
Rr = np.concatenate((np.zeros((4, 4)), np.identity(4)), axis=1).T
ctrl_dec = np.array([0, 14, 13, 3, 11, 5, 6, 8, 7, 9, 10, 4, 12, 2, 1, 15], dtype=int)

# Target area & resolution
res = 1
h, w = 640 * res, 480 * res     # 297, 210     640, 480
yy, xx = np.ogrid[-w / 2:w / 2, -h / 2:h / 2]
colors = {0: 'black', 1: 'white'}

# Diameter circles and frames declaration
# Target circle
DtcMax = 470 * res              #200     470
DtcMin = 430 * res              #190     450
dTc = np.linspace(DtcMax, DtcMin, 2)

# Message circles
DmcMax = 330 * res              #320
NCircles = 8
dMsgC = np.linspace(DmcMax, DmcMax/10, NCircles) #12 last value
MsgC_cy = 12 * res              #20

# k = (dMsgC[0] - dMsgC[1])/8
k = 10 * res                   #12 last value    (with 12, the system has the same performance,,, Hc o WOHc)  (with 5, hamming code works fine, without it no)

# Angle circle
# DacMax = 30 * res               #20        47
# dAc = np.linspace(DacMax, DacMax, 1)
Ac_cy = 186 * res               #175
# print('True thickness =', dMsgC[0] - dMsgC[2])

def ask4data():
    while True:
        n = int(input("Enter the number of targets you want (number between 1 and 16): "))
        if n <= 16 and n >= 1:
            num_targets = np.arange(n)
            return num_targets

        else:
            print("Target out of limits. Enter a valid target number")

def dec2bin(msg):
    # n = bin(msg).lstrip('-0b').zfill(4)
    # msg = [int(d) for d in str(n)]
    msg = [list(np.binary_repr(msg, 4))]
    return np.array(msg, dtype=int)

def hamming_encode(msg):
    C = np.dot(msg, G.astype(int)) % 2
    print C
    return np.array(C)[0]

def target_thickness(cont, k, msg):

    delta_D = [1] * len(msg)
    for i in range(len(msg)):
        if msg[i] == 0:
            delta_D[i] = -1
        else:
            delta_D[i] = 1
    delta_D = np.array(delta_D) * k
    new_cont = np.add(cont, delta_D)
    return new_cont

def print_target(dMsgC, a, res_circle):
    im = np.ones((w, h), dtype=np.bool)

    for i in range(len(dTc)):
        rad = dTc[i] / 2
        rad = rad * res_circle
        mask = ((xx / (rad * a)) ** 2 + (yy / rad) ** 2 <= 1)

        if colors[i % 2]:
            im = np.logical_xor(im, 1 - mask)
        else:
            im = np.logical_xor(im, mask)

    for i in range(len(dMsgC)):
        rad = dMsgC[i] / 2
        rad = rad * res_circle
        mask = ((xx / (rad * a)) ** 2 + ((yy - MsgC_cy * res_circle) / rad) ** 2 <= 1)
        if colors[i % 2]:
            im = np.logical_xor(im,  1 - mask)
        else:
            im = np.logical_xor(im, mask)

    dAc = np.linspace(dMsgC.min(), dMsgC.min(), 1)

    for i in range(len(dAc)):
        rad = dAc[i] / 2
        rad = rad * res_circle
        mask = ((xx / (rad * a)) ** 2 + ((yy + Ac_cy * res_circle) / rad) ** 2 <= 1)
        if colors[i % 2]:
            im = np.logical_xor(im, mask)
        else:
            im = np.logical_xor(im, 1 - mask)

    img = Image.fromarray(255 * np.uint8(im))
    return im, img


def detThickness(im):

    # subtract mean for the fft
    im_centered = im - im.mean()

    Xcorr = np.real(fftpack.ifft2(fftpack.fft2(im_centered) * np.conjugate(fftpack.fft2(im_centered))))
    det_thick = np.int(argrelextrema(Xcorr[0, :], np.greater)[0][0] * 5)

    if det_thick % 2 == 0:
        det_thick = det_thick + 1
    else:
        det_thick = det_thick

    # print('Detected thickness = ', det_thick)

    # plt.plot(Xcorr[:, 0])
    # plt.savefig('xcorr_'+str(num)+'.png')

    # plt.plot(Xcorr[0, :])
    # plt.savefig('xcorr_'+str(num)+'.png')

    return det_thick


def add_shade(im, alpha):
    im = np.array(im)#/np.max(im)
    m, n = im.shape
    im[0:m, 0:n/2] = im[0:m, 0:n/2] * alpha#* np.linspace(0, alpha*255, m/2)
    im_shade = im
    img_shade = Image.fromarray(im_shade.astype(np.uint8), mode='L')
    return im_shade, img_shade

def add_noise(im, mean, std):
    row, col = im.shape
    noise = np.random.normal(mean, std, (row, col))
    noise = noise.reshape(row, col)*res
    im_noisy = im + noise
    dst = np.zeros(shape=im_noisy.shape)
    dst = cv2.normalize(im_noisy, dst, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img_noisy = Image.fromarray(dst)
    return im_noisy, img_noisy

def create_pilot_target():
    outdir = 'normal'
    res_circle = 1
    a = 1
    im, img = print_target(dMsgC, a, res_circle)
    # detected_thickness = detThickness(im)
    img.save(os.path.join(outdir, 'pilot_target.png'))
    # img.show()
    return img

def create_normal_target(tar):
    outdir = 'normal'
    res_circle = 0.9
    a = 1
    msg_dec = int(tar)
    msg_bin = dec2bin(tar)
    codeword = hamming_encode(msg_bin)
    new_cont = target_thickness(dMsgC, k, codeword)
    im, img = print_target(new_cont, a, res_circle)
    # detected_thickness = detThickness(im)
    img.save(os.path.join(outdir, 'normal_target'+str(msg_dec)+'.png'))
    # img.show()
    return img

def create_resolution_target(tar, res_circle):
    outdir = 'resolution'
    a = 1
    msg_dec = int(tar)
    msg_bin = dec2bin(tar)
    codeword = hamming_encode(msg_bin)
    new_cont = target_thickness(dMsgC, (k * res_circle), codeword)
    im_scale, img_scale = print_target(new_cont, a, res_circle)
    # detected_thickness = detThickness(im_scale)
    img_scale.save(os.path.join(outdir, 'resolution'+str(res_circle)+'_target'+str(msg_dec)+'.png'))
    # img_scale.show()

    return img_scale

def create_ellipse_target(tar, major_axis):
    outdir = 'ellipse'
    res_circle = 0.5
    a = major_axis
    msg_dec = int(tar)
    msg_bin = dec2bin(tar)
    codeword = hamming_encode(msg_bin)
    new_cont = target_thickness(dMsgC, k, codeword)
    im_ellipse, img_ellipse = print_target(new_cont, a, res_circle)
    # detected_thickness = detThickness(im_ellipse)
    img_ellipse.save(os.path.join(outdir, 'maxis'+str(major_axis)+'_target'+str(msg_dec)+'.png'))
    # img_ellipse.show()

    return img_ellipse

def create_shade_target(tar, alpha):
    outdir = 'shade'
    res_circle = 0.9  # 0.9
    a = 1
    msg_dec = int(tar)
    msg_bin = dec2bin(tar)
    codeword = hamming_encode(msg_bin)
    new_cont = target_thickness(dMsgC, k, codeword)
    im, img = print_target(new_cont, a, res_circle)
    im_shade, img_shade = add_shade(img, alpha)
    # detected_thickness = detThickness(im)
    img_shade.save(os.path.join(outdir, 'shade'+str(alpha)+'_target'+str(msg_dec)+'.png'))
    # img_shade.show()

    return img_shade

def create_noise_target(tar, mean, std):
    outdir = 'noise'
    res_circle = 0.9
    a = 1
    msg_dec = int(tar)
    msg_bin = dec2bin(tar)
    codeword = hamming_encode(msg_bin)
    new_cont = target_thickness(dMsgC, k, codeword)
    im, img = print_target(new_cont, a, res_circle)
    im_noisy, img_noisy = add_noise(im, mean, std)
    # detected_thickness = detThickness(im_noisy)
    img_noisy.save(os.path.join(outdir, 'noise'+str(std)+'_target'+str(msg_dec)+'.png'))

    return img_noisy


if __name__ == '__main__':
    # targets = ask4data()
     ################## PILOT TARGET ###################
    img = create_pilot_target()
    targets = np.arange(16)  # Number of targets
    for t in targets:

        # # ################# NORMAL TARGET ###################
        img = create_normal_target(t)
        # # img.show()
        #
        # ################# RESOLUTION TARGET ###################
        # img_scale = create_resolution_target(t, 0.15)
        # # img_scale.show()
        #
        # ################# ELLIPSE TARGET ###################
        # img_ellipse = create_ellipse_target(t, 2.5)
        # # img_ellipse.show()
        #
        # ################## SHADE TARGET ###################
        # img_shade = create_shade_target(t, 0)
        # img_shade.show()

        ################ NOISE TARGET ###################
        # img_noisy = create_noise_target(t, 0, 0.08)
        # img_noisy.show()
