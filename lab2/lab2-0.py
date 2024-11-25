import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor


def ex1():
    kernel = [[-1, -1, -1],
            [-1, 8, -1], 
            [-1, -1, -1]]
    kernel = np.array(kernel)
    image = plt.imread('lenna.png')

    pad_width = kernel.shape[0] // 2
    padded_image = np.pad(image, pad_width, mode='constant')
    output_image = np.zeros_like(image)
    
    start_time = time.time()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output_image[i, j] = np.sum(kernel * padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]])
    
    end_time = time.time()
    total_time = end_time - start_time

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original image - ex 2.0")
    plt.imshow(image, cmap='binary_r')
    plt.subplot(1, 2, 2)
    plt.title(f"Total time: {total_time}")
    plt.imshow(output_image, cmap='binary_r')
    plt.tight_layout()
    plt.show()


def main():
    ex1()


if __name__ == "__main__":
    main()