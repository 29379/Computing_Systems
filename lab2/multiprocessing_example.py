import multiprocessing

def compute_square(number):
    return number * number

numbers = [1, 2, 3, 4]

if __name__ == "__main__":
    with multiprocessing.Pool(processes=2) as pool:  # Ustawienie liczby procesów na 2
        results = pool.map(compute_square, numbers)
    print("Wyniki:", results)
