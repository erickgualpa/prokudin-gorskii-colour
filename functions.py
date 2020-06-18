import cv2 
from scipy import signal
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt

def show_image(image):
    cv2.imshow('figure',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def crop_in_3_channels(image):
    im_height = image.shape[0] 
    slice_size = (int) (im_height / 3)
    channels = {
            "R":image[:slice_size][:],
            "G":image[slice_size:2*slice_size][:],
            "B":image[2*slice_size:3*slice_size][:]
    }

    print("\nCROPPING RESULTS:")
    print("- Channel R: ", channels['R'].shape)
    print("- Channel G: ", channels['G'].shape)
    print("- Channel B: ", channels['B'].shape)

    return channels

## CORRELATION METHODS ##############################################
def convolve2d(image1, image2):
    return signal.convolve2d(np.float64(image1), np.float64(image2[::-1,::-1]), mode='same', boundary='fill', fillvalue=0)

def fftconvolve(image1, image2):
    f1 = fftpack.fft2(image1)
    f2 = fftpack.fft2(image2)

    c = f1 * np.conj(f2)
    ic = fftpack.ifft2(c)
    ic = fftpack.fftshift(ic)

    return np.absolute(ic)
 
def normxcorr(image1, image2):
    ims_len = image1.shape[0] * image2.shape[1]

    image1 = (image1 - np.mean(image1)) / (np.std(image1) * ims_len)
    image2 = (image2 - np.mean(image2)) / (np.std(image2) * ims_len)

    return signal.correlate(image1, image2, 'same')

def corr_phase_fourier(image1, image2):
    f1 = fftpack.fft2(image1)
    f2 = fftpack.fft2(image2)

    c = f1 * np.conj(f2)
    c = c / np.absolute(c)
    ic = fftpack.ifft2(c)
    ic = fftpack.fftshift(ic)
    return np.absolute(ic)
  
def correlate_edges(image1, image2):
    low_th = 50
    high_th = 100

    image1 = cv2.Canny(image1,low_th,high_th)
    image2 = cv2.Canny(image2,low_th,high_th)

    return signal.fftconvolve(np.float64(image1), np.float64(image2[::-1,::-1]), mode='same')
#####################################################################

def fit_axis_x(im1, im2):
    borderType = cv2.BORDER_CONSTANT
    x_diff = im1.shape[1] - im2.shape[1]
    if x_diff > 0:
        # Añadir 'x_diff' columnas a la derecha de la segunda imagen
        im2 = cv2.copyMakeBorder(im2, 0, 0, 0, np.absolute(x_diff), borderType, None, 0)

    elif x_diff < 0:
        # Añadir 'x_diff' columnas a la derecha de la primera imagen
        im1 = cv2.copyMakeBorder(im1, 0, 0, 0, np.absolute(x_diff), borderType, None, 0)

    return im1, im2

def fit_axis_y(im1, im2):
    borderType = cv2.BORDER_CONSTANT
    y_diff = im1.shape[0] - im2.shape[0]
    if y_diff > 0:
        # Añadir 'y_diff' filas debajo de la segunda imagen
        im2 = cv2.copyMakeBorder(im2, 0, np.absolute(y_diff), 0, 0, borderType, None, 0)

    elif y_diff < 0:
        # Añadir 'y_diff' filas debajo de la primera imagen
        im1 = cv2.copyMakeBorder(im1, 0, np.absolute(y_diff), 0, 0, borderType, None, 0)

    return im1, im2

def merge_preprocessing(channel1, channel2, index1, index2):
    borderType = cv2.BORDER_CONSTANT

    y_diff = index1[0] - index2[0]
    x_diff = index1[1] - index2[1]

    im_A = channel1
    im_B = channel2

    print("\nINDEXES DIFFERENCE:")
    print("- X axis: ", x_diff)
    print("- Y axis: ", y_diff)

    if y_diff > 0:
        # Añadir 'y_diff' filas sobre la segunda imagen
        im_B = cv2.copyMakeBorder(im_B, np.absolute(y_diff), 0, 0, 0, borderType, None, 0)
    elif y_diff < 0:
        # Añadir 'y_diff' filas sobre la primera imagen
        im_A = cv2.copyMakeBorder(im_A, np.absolute(y_diff), 0, 0, 0, borderType, None, 0)

    im_A,im_B = fit_axis_y(im_A, im_B)

    if x_diff > 0:
        # Añadir 'x_diff' columnas a la izquierda de la segunda imagen
        im_B = cv2.copyMakeBorder(im_B, 0, 0, np.absolute(x_diff), 0, borderType, None, 0)

    elif x_diff < 0:
        # Añadir 'x_diff' columnas a la izquierda de la primera imagen
        im_A = cv2.copyMakeBorder(im_A, 0, 0, np.absolute(x_diff), 0, borderType, None, 0)

    im_A,im_B = fit_axis_x(im_A, im_B)

    print("Channels new shape: ",im_A.shape, im_B.shape)

    return im_A, im_B

def merge_preprocessing_channelsRG_to_B(channelR, channelG, channelB, index1, index2):
    borderType = cv2.BORDER_CONSTANT

    y_diff = index1[0] - index2[0]
    x_diff = index1[1] - index2[1]

    im_A = channelR
    im_B = channelG
    im_C = channelB

    print("\nINDEXES DIFFERENCE:")
    print("- X axis: ", x_diff)
    print("- Y axis: ", y_diff)

    if y_diff > 0:
        # Añadir 'y_diff' filas sobre la tercera imagen
        im_C = cv2.copyMakeBorder(im_C, np.absolute(y_diff), 0, 0, 0, borderType, None, 0)
    elif y_diff < 0:
        # Añadir 'y_diff' filas sobre la primera y segunda imagen
        im_A = cv2.copyMakeBorder(im_A, np.absolute(y_diff), 0, 0, 0, borderType, None, 0)
        im_B = cv2.copyMakeBorder(im_B, np.absolute(y_diff), 0, 0, 0, borderType, None, 0)

    im_A,im_C = fit_axis_y(im_A, im_C)
    im_B,im_C = fit_axis_y(im_B, im_C)

    if x_diff > 0:
        # Añadir 'x_diff' columnas a la izquierda de la tercera imagen
        im_C = cv2.copyMakeBorder(im_C, 0, 0, np.absolute(x_diff), 0, borderType, None, 0)

    elif x_diff < 0:
        # Añadir 'x_diff' columnas a la izquierda de la primera y segunda imagen
        im_A = cv2.copyMakeBorder(im_A, 0, 0, np.absolute(x_diff), 0, borderType, None, 0)
        im_B = cv2.copyMakeBorder(im_B, 0, 0, np.absolute(x_diff), 0, borderType, None, 0)

    im_A,im_C = fit_axis_x(im_A, im_C)
    im_B,im_C = fit_axis_x(im_B, im_C)

    print("Channels new shape: ",im_A.shape, im_B.shape, im_C.shape)

    return im_A, im_B, im_C

def align_by_convolution(channels, mode=None):
    im_R = channels['R'].copy()
    im_G = channels['G'].copy()
    im_B = channels['B'].copy()

    #-- CONVOLUCIÓN DE LOS CANALES R Y G ----------------------------#
    print("\n---- CHANNELS R AND G CONVOLUTION")
    if mode=="fft":
        conv_itself = fftconvolve(im_R, im_R)
        conv_R_G = fftconvolve(im_G, im_R)
    elif mode=="norm":
        conv_itself = normxcorr(im_R, im_R)
        conv_R_G = normxcorr(im_G, im_R)
    elif mode=="phase":
        conv_itself = corr_phase_fourier(im_R, im_R)
        conv_R_G = corr_phase_fourier(im_G, im_R)
    elif mode=="edges":
        conv_itself = correlate_edges(im_R, im_R)
        conv_R_G = correlate_edges(im_G, im_R)
    else: # Convolución del espacio
        conv_itself = convolve2d(im_R, im_R)
        conv_R_G = convolve2d(im_G, im_R)

    
    plt.imshow(conv_itself, cmap='gray')
    plt.show()
    plt.imshow(conv_R_G, cmap='gray')
    plt.show() 
   
    
    # Obtener el índice de los valores máximos para cada señal
    index_conv_itself = np.unravel_index(np.argmax(conv_itself), conv_itself.shape)
    index_conv_R_G = np.unravel_index(np.argmax(conv_R_G), conv_R_G.shape)

    print("\nINDEXES:")
    print("Convolution itself index CH_R: ", index_conv_itself)
    print("Convolution index CH_R_G: ", index_conv_R_G)


    ch_R, ch_G = merge_preprocessing(im_R, im_G, index_conv_itself, index_conv_R_G)
    #----------------------------------------------------------------#

    rgg = np.dstack((ch_R, ch_G, ch_G))
    rgg_gray = cv2.cvtColor(rgg, cv2.COLOR_BGR2GRAY)

    #-- CONVOLUCIÓN DE LOS CANALES RG Y B ---------------------------#
    print("\n---- CHANNELS RG AND B CONVOLUTION")
    rgg_gray, im_B = fit_axis_x(rgg_gray, im_B)
    rgg_gray, im_B = fit_axis_y(rgg_gray, im_B)
    if mode=="fft":
        conv_itself = fftconvolve(rgg_gray, rgg_gray)
        conv_RG_B = fftconvolve(im_B, rgg_gray)
    elif mode=="norm":
        conv_itself = normxcorr(rgg_gray, rgg_gray)
        conv_RG_B = normxcorr(im_B, rgg_gray)
    elif mode=="phase":
        conv_itself = corr_phase_fourier(rgg_gray, rgg_gray)
        conv_RG_B = corr_phase_fourier(im_B, rgg_gray)
    elif mode=="edges":
        conv_itself = correlate_edges(rgg_gray, rgg_gray)
        conv_RG_B = correlate_edges(im_B, rgg_gray)
    else: # Convolución del espacio
        conv_itself = convolve2d(rgg_gray, rgg_gray)
        conv_RG_B = convolve2d(im_B, rgg_gray)

    plt.imshow(conv_itself, cmap='gray')
    plt.show()
    plt.imshow(conv_RG_B, cmap='gray')
    plt.show() 

    # Obtener el índice de los valores máximo para cada imagen
    index_conv_itself = np.unravel_index(np.argmax(conv_itself), conv_itself.shape)
    index_conv_RG_B = np.unravel_index(np.argmax(conv_RG_B), conv_RG_B.shape)

    print("\nINDEXES:")
    print("Convolution itself index CH_RG: ", index_conv_itself)
    print("Convolution index CH_RG_B: ", index_conv_RG_B)
    
    ch_R, ch_G, ch_B = merge_preprocessing_channelsRG_to_B(ch_R, ch_G, im_B, index_conv_itself, index_conv_RG_B)
    #----------------------------------------------------------------#

    return np.dstack((ch_R, ch_G, ch_B))
