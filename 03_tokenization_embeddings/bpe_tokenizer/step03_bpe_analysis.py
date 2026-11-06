"""Skeleton untuk membaca dan membandingkan hasil training BPE."""

import sys
from pathlib import Path

# Masukkan folder tempat file ini berada ke search path
sys.path.insert(0, str(Path(__file__).parent))

from step02_bpe_tokenizer import CORPUS_TEXT, VOCAB_SIZE, BPECharacterTokenizer


def print_pair_and_result_token(vocab, token):
    for pair, new_id in token:
        print(vocab[pair[0]], vocab[pair[1]], "=>", vocab[new_id])


def print_todo_4(tokenizer, word):
    encoded_ids = tokenizer.encode(word)
    print("kalimat : ", word, " total token: ", len(encoded_ids))
    print("vocabnya : ", [tokenizer.vocab[id] for id in encoded_ids])


def main():
    print("Skeleton analisis hasil training BPE")
    # print("10 pertama: ", tokenizer.vocab[:10])
    # print("10 terakhir: ", tokenizer.vocab[-10:])
    # Hint: script ini harus tetap bisa dijalankan dari root repo.
    # TODO 1: gunakan class, korpus, dan ukuran vocab dari script 02.
    tokenizer = BPECharacterTokenizer()
    tokenizer.train(text=CORPUS_TEXT, vocab_size=VOCAB_SIZE)
    # TODO 2: print total vocab size.
    print("=== TODO 2 ===")
    print("cek vocab: ", len(tokenizer.vocab))
    # TODO 3: print 10 merge pertama dan 10 merge terakhir berurutan.
    print("=== TODO 3 ===")
    first_10_token = tokenizer.merge_order[:10]
    print("=== 10 Merge Pertama ===")
    print_pair_and_result_token(tokenizer.vocab, first_10_token)
    last_10_token = tokenizer.merge_order[-10:]
    print("=== 10 Merge terakhir ===")
    print_pair_and_result_token(tokenizer.vocab, last_10_token)
    # TODO 4: bandingkan satu kata sering dengan kata langka/typo.
    print("=== TODO 4 ===")
    frequent_word = "keren"
    print_todo_4(tokenizer, frequent_word)
    typo_word = "bagnet"
    print_todo_4(tokenizer, typo_word)
    # TODO 5: cetak insight tentang kompromi ukuran vocab dan jumlah token.
    print(
        "kalau vocab size yang di train sedikit otomatis token makin banyak dan"
        "komputasinya berat sedangkan vocabsize nya banyak komputasi sedikit dan cepat tapi tradeoffnya di RAM butuh kerja berat"
    )


if __name__ == "__main__":
    main()
