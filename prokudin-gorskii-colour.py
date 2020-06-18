from functions import *
import cv2
import sys

## GET 'Input image' AND 'mode' FROM COMMAND LINE ARGUMENTS #########
input_filename = ''
mode = ''
try:
    input_filename = sys.argv[1]
except Exception:
    print("[ERROR]: Input image wasn't set")

try:
    mode = sys.argv[2]  # "fft", "phase", "norm", "edges", None=(Space Convolution)
except Exception:
    print("[ERROR]: Operation mode wasn't set")

working_im = cv2.imread(input_filename, 0)
results_path = "./results/"
#####################################################################
try:
    ## SPLIT WORKING IMAGE CHANNELS #####################################
    ch = crop_in_3_channels(working_im.copy())
    # original = cv2.merge((ch['R'], ch['G'], ch['B']))
    #####################################################################

    if mode == "fft":
        ## CORRELATION (Fourier space product) ##############################
        merged = align_by_convolution(ch, mode="fft")
        cv2.imwrite(results_path + "fourier_space_product.jpg", merged)
        #####################################################################
    elif mode == "phase":
        ## PHASE CORRELATION (Bases on Fourier) #############################
        merged = align_by_convolution(ch, mode="phase")
        cv2.imwrite(results_path + "fourier_phase_correlation.jpg", merged)
        #####################################################################
    elif mode == "norm":
        ## NORMALIZED CORRELATION ###########################################
        merged = align_by_convolution(ch, mode="norm")
        cv2.imwrite(results_path + "normalized_correlation.jpg", merged)
        #####################################################################
    elif mode == "edges":
        ## CORRELATION (Edge images) #########################################
        merged = align_by_convolution(ch, mode="edges")
        cv2.imwrite(results_path + "edges_correlation.jpg", merged)
        #####################################################################
    else:
        ## CORRELATION (Space convolution) ##################################
        merged = align_by_convolution(ch)
        cv2.imwrite(results_path + "space_correlation.jpg", merged)
        #####################################################################

except Exception as e:
    print('[ERROR]:', e)

print('[ERROR]: Finishing program')



