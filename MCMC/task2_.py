import numpy as np
import re
from collections import defaultdict, Counter
import chardet


def load_text(path: str) -> str:
    with open(path, 'rb') as f:
        raw = f.read()
    enc = chardet.detect(raw)['encoding'] or 'utf-8'
    return raw.decode(enc, errors='ignore')

text = load_text("./Data/TheWarOfTheWorlds.txt")

# # 1. Розбиваємо на слова та приводимо до нижнього регістру
# words = re.findall(r'[a-zA-Zа-яА-ЯёЁ]+', text.lower())

# # 2. Формуємо біграми та рахуємо частоти
# bigram_counts = defaultdict(Counter)

# for word in words:
#     for i in range(len(word)-1):
#         first, second = word[i], word[i+1]
#         bigram_counts[first][second] += 1

# # 3. Обчислюємо ймовірності для кожної біграми
# bigram_probs = {}
# for first_letter, counter in bigram_counts.items():
#     total = sum(counter.values())
#     bigram_probs[first_letter] = {second: count/total for second, count in counter.items()}

# # 4. Вивід результату
# print("Частоти біграм:")
# for k, v in bigram_counts.items():
#     print(f"{k}: {dict(v)}")

# print("\nЙмовірності наступної букви:")
# for k, v in bigram_probs.items():
#     prob_dict = {k_: round(v_, 4) for k_, v_ in v.items()}
#     print(f"{k}: {prob_dict}")




# 1. Беремо тільки ASCII + українські літери
words = re.findall(r'[a-zA-Zа-яА-ЯёЁ]+', text.lower())

# 2. Створюємо множину всіх унікальних букв
letters = sorted(set(''.join(words)))
letter_to_idx = {c: i for i, c in enumerate(letters)}

# 3. Ініціалізуємо матрицю M (розмір: n_letters x n_letters)
n = len(letters)
M = np.zeros((n, n), dtype=float)

# 4. Рахуємо частоти біграм
for word in words:
    for i in range(len(word)-1):
        first, second = word[i], word[i+1]
        M[letter_to_idx[first], letter_to_idx[second]] += 1

M += 1

# 5. Нормалізуємо по рядках, щоб отримати ймовірності
row_sums = M.sum(axis=1, keepdims=True)
M_prob = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums!=0)

# 6. Вивід
print("Літери:", letters)
print("Матриця ймовірностей M_prob:")
print(np.round(M_prob, 2))
