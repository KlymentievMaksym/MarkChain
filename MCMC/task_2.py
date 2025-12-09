import random
import numpy as np
import collections
import chardet
import string
from tqdm import trange

alphabet = list(string.ascii_lowercase)
alphabet_set = set(alphabet)
epsilon = 1e-12

def make_cipher_map():
    shuffled = alphabet.copy()
    random.shuffle(shuffled)
    return dict(zip(alphabet, shuffled))

def cipher(text, cipher_map: dict = None):
    if cipher_map is None:
        cipher_map = make_cipher_map()
    result = []
    for ch in text:
        low = ch.lower()
        if low in cipher_map:
            mapped = cipher_map[low]
            # preserve case
            result.append(mapped.upper() if ch.isupper() else mapped)
        else:
            result.append(ch)
    return ''.join(result)

def clean(text):
    return ''.join(ch for ch in text.lower() if ch in alphabet_set)

def load_text(path: str) -> str:
    with open(path, 'rb') as f:
        raw = f.read()
    enc = chardet.detect(raw)['encoding'] or 'utf-8'
    return raw.decode(enc, errors='ignore')

def get_bigrams(text):
    return [text[i:i+2] for i in range(len(text)-1)]

def receive_array_counts(counter: collections.Counter, learn_unique_character: list, smoothing: float = 1.0):
    len_ch = len(learn_unique_character)
    indexes = {char: i for i, char in enumerate(learn_unique_character)}
    counts = np.zeros((len_ch, len_ch), dtype=float)

    for bigram, cnt in counter.items():
        a, b = bigram[0], bigram[1]
        if a in indexes and b in indexes:
            counts[indexes[a], indexes[b]] = cnt

    counts += smoothing
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = counts / row_sums
    return probs

def log_score(mapping: dict, logM: np.array, ciphered_text: str, learn_indexes: dict):
    score = 0.0
    for i in range(len(ciphered_text) - 1):
        char_first = ciphered_text[i].lower()
        char_second = ciphered_text[i+1].lower()
        if char_first in mapping and char_second in mapping:
            char_first_mapped = mapping[char_first]
            char_second_mapped = mapping[char_second]
            if char_first_mapped in learn_indexes and char_second_mapped in learn_indexes:
                score += logM[learn_indexes[char_first_mapped], learn_indexes[char_second_mapped]]
            else:
                score += np.log(epsilon)
        else:
            score += np.log(epsilon)
    return score

def swap_mapping(mapping: dict):
    new_map = mapping.copy()
    a, b = random.sample(alphabet, 2)
    new_map[a], new_map[b] = new_map[b], new_map[a]
    return new_map

def find_best_score(f: dict, logM: np.array, ciphered_text: str, learn_indexes: dict, current_score: float):
    f_candidate = swap_mapping(f)
    score_candidate = log_score(f_candidate, logM, ciphered_text, learn_indexes)

    if score_candidate >= current_score:
        return f_candidate, score_candidate
    else:
        u = random.random()
        prob = np.exp(score_candidate - current_score)
        if u < prob:
            return f_candidate, score_candidate
        else:
            return f, current_score

def task_2(path_to_learn: str, path_to_decipher: str, iterations: int = 2000, report_every: int = 1000):
    raw_learn = load_text(path_to_learn)
    raw_decipher = load_text(path_to_decipher)

    text_to_learn = clean(raw_learn)
    plaintext_target = raw_decipher
    cleaned_plain_target = clean(plaintext_target)

    true_cipher_map = make_cipher_map()
    ciphered_target_full = cipher(plaintext_target, true_cipher_map)
    ciphered_clean_target = clean(ciphered_target_full)

    learn_bigrams = get_bigrams(text_to_learn)
    counter = collections.Counter(learn_bigrams)
    learn_unique_character = sorted(list(set(text_to_learn)))
    learn_indexes = {c: i for i, c in enumerate(learn_unique_character)}

    M = receive_array_counts(counter, learn_unique_character, smoothing=1.0)
    logM = np.log(M + epsilon)

    f = dict(zip(alphabet, alphabet))
    current_score = log_score(f, logM, ciphered_clean_target, learn_indexes)
    best_map = f.copy()
    best_score = current_score

    print(f"Start score: {current_score:.2f}, unique learn chars: {len(learn_unique_character)}\n")
    for it in trange(1, iterations + 1):
        f, current_score = find_best_score(f, logM, ciphered_clean_target, learn_indexes, current_score)
        if current_score > best_score:
            best_score = current_score
            best_map = f.copy()
        if it % report_every == 0 or it == 1:
            print(f"\niter {it}/{iterations} | curr {current_score:.2f} | best {best_score:.2f}\n")

    deciphered_full = []
    for char in ciphered_target_full:
        char_low = char.lower()
        if char_low in best_map:
            mapped = best_map[char_low]
            deciphered_full.append(mapped.upper() if char.isupper() else mapped)
        else:
            deciphered_full.append(char)
    deciphered_full_text = ''.join(deciphered_full)

    print("\n[Example outputs]\n")
    index = random.randint(0, len(ciphered_target_full) - 200)
    print("[Ciphered sample]\n", ciphered_target_full[index:index+200])
    print("\n[Deciphered sample]\n", deciphered_full_text[index:index+200])
    print("\n[True mapping]")
    print(f"{true_cipher_map}")
    print("\n[Recovered mapping]")
    print(f"{best_map}")
    return {
        "best_map": best_map,
        "best_score": best_score,
        "deciphered_text": deciphered_full_text,
        "ciphered_text": ciphered_target_full,
        "true_cipher_map": true_cipher_map
    }

if __name__ == '__main__':
    res = task_2("./Data/TheWarOfTheWorlds.txt", "./Data/TheTimeMachine.txt", iterations=1500, report_every=100)
    with open("./Data/Deciphered.txt", "w", encoding="utf-8") as out:
        out.write(res["deciphered_text"])
    print("\n[Saved] ./Data/Deciphered.txt")
