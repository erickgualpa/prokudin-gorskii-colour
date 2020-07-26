# Giving colour to the Prokudin-Gorskii photography collection 
This project tries to give colour to some images from the Prokudin-Gorskii collection. The samples from this collection are sets of three images, each one representing a channel from the RGB colorspace.

<p align="center">
  <img src="img/im4.jpg" height="450">
  <img src="results/normalized_correlation_2.jpg" height="450">
</p>

## Usage :pencil:

The program can be executed in 5 modes:

* Calculating **Correlation (fft)** between the Fast Fourier Transforms from the images channels
* Calculating **Phase Correlation (phase)** between the Fast Fourier Transforms from the images channels
* Calculating **Normalized Correlation (norm)** between the image channels
* Calculating **Correlation (edges)** between the Fast Fourier Transforms from the edges images channels
* Calculating **Correlation (default)** between the image channels

To execute it just run the prokudin-gorskii-colour.py file with a Python 3 version. Results will be saved in the **results** directory.

```console
erickgualpa@erickgm:~$ python prokudin-gorskii-colour.py ./img/im1.jpg fft
```

You can get sample images from the collection here: [Prokudin-Gorskii Photography Collection](https://www.loc.gov/exhibits/empire/gorskii.html)
