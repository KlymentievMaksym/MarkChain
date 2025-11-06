import random
import numpy as np
import collections
import chardet
import string
from tqdm import trange

alphabet = list(string.ascii_lowercase)
ALPH_SET = set(alphabet)
EPS = 1e-12

def make_cipher_map():
    shuffled = alphabet.copy()
    random.shuffle(shuffled)
    return dict(zip(alphabet, shuffled))

def cipher(text, cipher_map: dict = None):
    """Apply substitution cipher (lowercase only)."""
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
    """Return only lowercase letters (no spaces/punct)."""
    return ''.join(ch for ch in text.lower() if ch in ALPH_SET)

def load_text(path: str) -> str:
    with open(path, 'rb') as f:
        raw = f.read()
    enc = chardet.detect(raw)['encoding'] or 'utf-8'
    return raw.decode(enc, errors='ignore')

def get_bigrams(text):
    return [text[i:i+2] for i in range(len(text)-1)]

def receive_array_counts(counter: collections.Counter, learn_unique_character: list, smoothing: float = 1.0):
    """Build row-normalized bigram probability matrix with add-k smoothing."""
    len_ch = len(learn_unique_character)
    idx = {c: i for i, c in enumerate(learn_unique_character)}
    counts = np.zeros((len_ch, len_ch), dtype=float)

    for bigram, cnt in counter.items():
        a, b = bigram[0], bigram[1]
        if a in idx and b in idx:
            counts[idx[a], idx[b]] = cnt

    # additive smoothing:
    counts += smoothing
    # normalize rows to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    # avoid division by zero
    row_sums[row_sums == 0] = 1.0
    probs = counts / row_sums
    return probs

def log_score(mapping: dict, logM: np.array, ciphered_text: str, learn_idx: dict):
    """Return sum of log-probs of mapped bigrams under logM.
       mapping: cipher_char -> hypothesized_plain_char (lowercase)
       logM: matrix log-probabilities where logM[i,j] = log P(j | i) or similar depending on ordering
       learn_idx: mapping from character to row/col index in logM
    """
    s = 0.0
    for i in range(len(ciphered_text) - 1):
        a = ciphered_text[i].lower()
        b = ciphered_text[i+1].lower()
        if a in mapping and b in mapping:
            ma = mapping[a]
            mb = mapping[b]
            if ma in learn_idx and mb in learn_idx:
                s += logM[learn_idx[ma], learn_idx[mb]]
            else:
                s += np.log(EPS)
        else:
            s += np.log(EPS)
    return s

def random_key():
    perm = alphabet.copy()
    random.shuffle(perm)
    return dict(zip(alphabet, perm))

def swap_mapping(mapping: dict):
    """Return a new mapping with two source letters swapped (swap outputs)."""
    new_map = mapping.copy()
    a, b = random.sample(alphabet, 2)
    new_map[a], new_map[b] = new_map[b], new_map[a]
    return new_map

def find_best_score(f: dict, logM: np.array, ciphered_text: str, learn_idx: dict, current_score: float):
    """Propose swap, compute candidate score, accept/reject via MH. Return (f_new, score_new)."""
    f_candidate = swap_mapping(f)
    score_candidate = log_score(f_candidate, logM, ciphered_text, learn_idx)

    # acceptance probability (in log-space): accept with prob min(1, exp(score_candidate - current_score))
    if score_candidate >= current_score:
        return f_candidate, score_candidate
    else:
        u = random.random()
        prob = np.exp(score_candidate - current_score)
        if u < prob:
            return f_candidate, score_candidate
        else:
            return f, current_score

def task_2(path_to_learn: str, path_to_decipher: str, iterations: int = 20000, report_every: int = 2000):
    # Load texts
    raw_learn = load_text(path_to_learn)
    raw_decipher = load_text(path_to_decipher)

    # Cleaned training text for building bigram model (letters only)
    text_to_learn = clean(raw_learn)
    # For demonstration we cipher the second text to produce a 'ciphered' input
    plaintext_target = raw_decipher  # keep original with punctuation/spacing
    cleaned_plain_target = clean(plaintext_target)

    # Create a random cipher for the target (you can reuse make_cipher_map if desired)
    true_cipher_map = make_cipher_map()
    ciphered_target_full = cipher(plaintext_target, true_cipher_map)           # with punctuation preserved
    ciphered_clean_target = clean(ciphered_target_full)                       # letters-only version used for scoring

    # Build bigram frequencies from training text
    learn_bigrams = get_bigrams(text_to_learn)
    counter = collections.Counter(learn_bigrams)
    learn_unique_character = sorted(list(set(text_to_learn)))
    # ensure alphabet subset includes all alphabet for mapping; if some letters not in learn set, still include them
    # We'll create learn_idx only for characters present in training data
    learn_idx = {c: i for i, c in enumerate(learn_unique_character)}

    # build probability matrix and convert to log
    M = receive_array_counts(counter, learn_unique_character, smoothing=1.0)
    logM = np.log(M + EPS)

    # initialize random mapping: cipher_char -> hypothesized_plain_char
    f = random_key()
    # current score
    current_score = log_score(f, logM, ciphered_clean_target, learn_idx)
    best_map = f.copy()
    best_score = current_score

    print(f"Start score: {current_score:.2f}, unique learn chars: {len(learn_unique_character)}")
    for it in trange(1, iterations + 1):
        f, current_score = find_best_score(f, logM, ciphered_clean_target, learn_idx, current_score)
        if current_score > best_score:
            best_score = current_score
            best_map = f.copy()
        if it % report_every == 0 or it == 1:
            print(f"iter {it}/{iterations} | curr {current_score:.2f} | best {best_score:.2f}")

    # apply best_map to the full ciphered text (preserving punctuation/case)
    # But best_map maps lowercase source->lowercase target; build reverse mapping for deciphering:
    # best_map: cipher_letter -> plaintext_letter
    deciphered_full = []
    for ch in ciphered_target_full:
        low = ch.lower()
        if low in best_map:
            mapped = best_map[low]
            deciphered_full.append(mapped.upper() if ch.isupper() else mapped)
        else:
            deciphered_full.append(ch)
    deciphered_full_text = ''.join(deciphered_full)

    print("\n=== Example outputs ===")
    print("Ciphered sample (first 200 chars):\n", ciphered_target_full[:200])
    print("\nDeciphered sample (first 200 chars):\n", deciphered_full_text[:200])
    print("\nTrue mapping (a->?) sample:")
    for k in sorted(list(true_cipher_map.keys()))[:10]:
        print(f"{k} -> {true_cipher_map[k]}")
    print("\nRecovered mapping (a->?) sample:")
    for k in sorted(list(best_map.keys()))[:10]:
        print(f"{k} -> {best_map[k]}")

    return {
        "best_map": best_map,
        "best_score": best_score,
        "deciphered_text": deciphered_full_text,
        "ciphered_text": ciphered_target_full,
        "true_cipher_map": true_cipher_map
    }

if __name__ == '__main__':
    res = task_2("./Data/TheWarOfTheWorlds.txt", "./Data/TheTimeMachine.txt", iterations=35000, report_every=1000)
    # Save deciphered output to file
    with open("deciphered_output.txt", "w", encoding="utf-8") as out:
        out.write(res["deciphered_text"])
    print("\nSaved best deciphered text to deciphered_output.txt")
