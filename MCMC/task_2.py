import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba


def task_2(path: str):
    with open(path, 'r') as text_file:
        text = text_file.readable()
    print(text)
    print(len(text))


if __name__ == '__main__':
    task_2(path="./Data/Mother of Learning - nobody103.txt")

    