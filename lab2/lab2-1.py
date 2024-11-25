import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor


def convolve_pixel(args):
    i, j, kernel, padded_image = args
    print(i,j)
    return i, j, np.sum(kernel * padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]])

def convolve_batch(args):
    batch, kernel, padded_image = args
    results = []
    for i, j in batch:
        value = np.sum(kernel * padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]])
        results.append((i, j, value))
    return results

def ex2():
    kernel = [[-1, -1, -1],
              [-1, 8, -1], 
              [-1, -1, -1]]
    kernel = np.array(kernel)
    image = plt.imread('lenna.png')
    # image = np.random.random((5, 5))

    pad_width = kernel.shape[0] // 2
    padded_image = np.pad(image, pad_width, mode='constant')
    output_image = np.zeros_like(image)
    
    pixels = [(i, j) for i in range(image.shape[0]) for j in range(image.shape[1])]
    batch_size = len(pixels) // 4
    batches = [pixels[i:i + batch_size] for i in range(0, len(pixels), batch_size)]
    args = [(batch, kernel, padded_image) for batch in batches]

    start_time = time.time()
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(convolve_batch, args))
    
    for batch_results in results:
        for i, j, value in batch_results:
            output_image[i, j] = value

    end_time = time.time()
    total_time = end_time - start_time
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='binary_r')
    plt.title("Original image - ex 2.1")
    plt.subplot(1, 2, 2)
    plt.imshow(output_image, cmap='binary_r')
    plt.title(f"Total time: {total_time}")
    plt.tight_layout()
    plt.show()


def main():
    ex2()


if __name__ == "__main__":
    main()