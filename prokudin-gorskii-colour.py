import matplotlib.pyplot as plt
from functions import *
import os
import cv2
import numpy as np

working_image = 2 # Set working image
mode = "phase"    # "fft", "phase", "norm", "edges", None=(Space Convolution)


## 1 - LEER IMÁGENES ################################################
img_path = "./img/"
results_path = "./results/"
im = 1

img_set = []
large_set = []
for fold in os.listdir(img_path):
    img = cv2.imread(img_path + fold + "/" + str(im) + ".jpg", 0)
    large_img = cv2.imread(img_path + fold + "/" + str(im) + "large.tif", 0)
    img_set.append(img)
    large_set.append(large_img)
    im += 1
#####################################################################

## 2 - DIVIDIR IMÁGENES EN CANALES ##################################
im1 = img_set[working_image].copy()

ch = cropIn3Channels(im1)
cv2.imwrite(img_path + "im"+str(working_image+1)+"/r.jpg", ch['R'])
cv2.imwrite(img_path + "im"+str(working_image+1)+"/g.jpg", ch['G'])
cv2.imwrite(img_path + "im"+str(working_image+1)+"/b.jpg", ch['B'])

original = cv2.merge((ch['R'], ch['G'], ch['B']))
cv2.imwrite(img_path + "im"+str(working_image+1)+"/original.jpg", original)
#####################################################################

if mode == "fft":
	## 4 - CORRELACIÓN (Producto en el espacio de Fourier) ############## 
	merged = alignByConvolution(ch, mode="fft")
	cv2.imwrite(results_path + "im"+str(working_image+1)+"/product_espacio_fourier.jpg", merged)
	#####################################################################
elif mode == "phase":
	## 5 - CORRELACIÓN (De fase basada en Fourier) ######################
	merged = alignByConvolution(ch, mode="phase")
	cv2.imwrite(results_path + "im"+str(working_image+1)+"/correlacion_fase_fourier.jpg", merged)
	#####################################################################
elif mode == "norm":
	## 6 - CORRELACIÓN (Normalizada) ####################################
	merged = alignByConvolution(ch, mode="norm")
	cv2.imwrite(results_path + "im"+str(working_image+1)+"/correlacion_normalizada.jpg", merged)
	#####################################################################
elif mode == "edges":
	## 7 - CORRELACIÓN (Edges) ##########################################
	merged = alignByConvolution(ch, mode="edges")
	cv2.imwrite(results_path + "im"+str(working_image+1)+"/correlacion_edges.jpg", merged)
	#####################################################################
else: 
	## 3 - CORRELACIÓN (Convolución del espacio) ######################## 
	merged = alignByConvolution(ch)
	cv2.imwrite(results_path + "im"+str(working_image+1)+"/convolucio_espai.jpg", merged)
	#####################################################################









