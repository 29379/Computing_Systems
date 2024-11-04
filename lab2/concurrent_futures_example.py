from concurrent.futures import ProcessPoolExecutor

def compute_cube(number):
    return number * number * number

numbers = [1, 2, 3, 4]

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(compute_cube, numbers))
    print("Wyniki:", results)

