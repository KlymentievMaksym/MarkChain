import random
import numpy as np
import numba
import collections
import chardet
import string
import tqdm

alphabet = list(string.ascii_lowercase)

def make_cipher_map():
    shuffled = alphabet.copy()
    random.shuffle(shuffled)
    return dict(zip(alphabet, shuffled))

def cipher(text, cipher_map: dict = None):
    if cipher_map is None:
        cipher_map = make_cipher_map()
    result = []
    for ch in text.lower():
        if ch in cipher_map:
            result.append(cipher_map[ch])
        else:
            result.append(ch)
    return ''.join(result)

def clean(text):
    result = []
    for ch in text.lower():
        if ch in alphabet:
            result.append(ch)
    return ''.join(result)

def load_text(path: str) -> str:
    with open(path, 'rb') as f:
        raw = f.read()
    enc = chardet.detect(raw)['encoding'] or 'utf-8'
    return raw.decode(enc, errors='ignore')

def get_bigrams(text):
    return [text[i:i+2] for i in range(len(text)-1)]

def receive_array_counts(counter: collections.Counter, learn_characters: dict):
    len_ch = len(learn_characters)
    counter_keys = sorted(list(counter.keys()))
    M = np.zeros((len_ch, len_ch), dtype=float)

    for bigram in counter_keys:
        M[learn_characters[bigram[0]], learn_characters[bigram[1]]] = counter[bigram]

    M += 1
    for index in range(M.shape[0]):
        M[index] = M[index] / M[index].sum()
    M = np.log(M)

    return M

@numba.jit()
def score(f: np.ndarray, M: np.array, text_indices: np.ndarray) -> float:
    score = 0
    for index in range(len(text_indices) - 1):
        score += M[f[text_indices[index]], f[text_indices[index+1]]]
    return score #/ len(text_indices)

@numba.jit()
def swap(f: np.ndarray) -> np.ndarray:
    f_ = f.copy()
    i, j = np.random.choice(len(f), 2, replace=False)
    f_[i], f_[j] = f_[j], f_[i]
    return f_


def task_2(path_to_learn: str, path_to_decipher: str, iterations: int = 10, every: int = 1, T = 1):
    # n = 100000

    text_to_learn = load_text(path_to_learn)
    text_to_deciphr = load_text(path_to_decipher)
    text_to_learn = clean(text_to_learn)
    text_to_decipher_r = cipher(text_to_deciphr)
    text_to_decipher = clean(text_to_decipher_r)

    learn_unique_character = np.array(sorted(list(set(text_to_learn))))
    learn_characters = {ch: i for i, ch in enumerate(learn_unique_character)}

    learn_bigrams = get_bigrams(text_to_learn)
    # decipher_bigrams = get_bigrams(text_to_decipher)
    text_indices = np.array([learn_characters[ch] for ch in text_to_decipher], dtype=int)  #[:n]

    counter = collections.Counter(learn_bigrams)
    M = receive_array_counts(counter, learn_characters)

    f = np.arange(len(learn_unique_character))
    # f = dict(zip(alphabet, alphabet))

    best_f = f.copy()
    best_score = score(f, M, text_indices)

    f_score = best_score
    for iteration in tqdm.tqdm(range(iterations)):
        f_ = swap(f)
        f_score_ = score(f_, M, text_indices)
        accept = f_score_ / f_score
        # if accept > 0 or np.random.rand() < np.exp(accept / T):
        if np.random.rand() < accept:
            f, f_score = f_, f_score_
        if f_score > best_score:
            best_score, best_f = f_score, f.copy()

        # T *= 0.999

        if (iteration + 1) % every == 0:
            print(f"[Best score] Iter {iteration + 1}: {best_score}")


    # F = {ch: learn_unique_character[f[i]] for i, ch in enumerate(learn_unique_character)}
    # best_F = {ch: learn_unique_character[best_f[i]] for i, ch in enumerate(learn_unique_character)}
    F = {str(ls): str(ls_) for ls, ls_ in zip(learn_unique_character, learn_unique_character[f])}
    best_F = {str(ls): str(ls_) for ls, ls_ in zip(learn_unique_character, learn_unique_character[best_f])}

    n = 100

    # print(f"[F] {F}")
    print(f"[Best F] {best_F}")
    print(f"[Best score] {best_score}")
    # print(f"[F] {cipher(text_to_decipher_r[:n], f)}")
    decode_map = {learn_unique_character[f[i]]: learn_unique_character[i] for i in range(len(best_f))}
    print(f"[Ciphered] {text_to_decipher_r[:n]}\n")
    print(f"[Decipher] {cipher(text_to_decipher_r[:n], best_F)}\n")
    print(f"[Real text] {text_to_deciphr[:n]}")



if __name__ == '__main__':
    task_2("./Data/TheWarOfTheWorlds.txt", "./Data/TheTimeMachine.txt", 1_000_000, every = 250_000)
