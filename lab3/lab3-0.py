import numpy as np
import matplotlib.pyplot as plt
import time
import pyximport
pyximport.install()
from lab3_0_convolve import convolve

def ex():
    kernel = [[-1, -1, -1],
              [-1, 8, -1], 
              [-1, -1, -1]]
    kernel = np.array(kernel, dtype=np.int32)
    image = plt.imread('lenna.png')
    start_time = time.time()

    pad_width = kernel.shape[0] // 2
    padded_image = np.pad(image, pad_width, mode='constant')

    output_image = convolve(padded_image, kernel)
    
    end_time = time.time()
    total_time = end_time - start_time
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original image - ex 3.0")
    plt.imshow(image, cmap='binary_r')
    plt.subplot(1, 2, 2)
    plt.title(f"Total time: {total_time}")
    plt.imshow(output_image, cmap='binary_r')
    plt.tight_layout()
    plt.show()

def main():
    ex()


if __name__ == "__main__":
    main()