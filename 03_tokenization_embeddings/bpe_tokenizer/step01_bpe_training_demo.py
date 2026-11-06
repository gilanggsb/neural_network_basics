"""Skeleton demo untuk tiga merge BPE karakter-level."""

text = "aaabdaaabac"
characters = sorted(set(text))
# TODO 1: bikin mapping karakter -> ID.
char_to_id = {char: id for id, char in enumerate(characters)}
print("Chart to id = ", char_to_id)

# TODO 1: bikin mapping ID -> potongan teks.
vocab = {id: char for char, id in char_to_id.items()}
print("vocab = ", vocab)
# TODO 2: ubah setiap karakter di text menjadi ID.
ids = [char_to_id[char] for char in text]
print("ids = ", ids)


def get_stats(sequence):
    # TODO 3: hitung frekuensi semua pasangan adjacent.
    # Hint: posisi overlapping ikut dihitung.
    stats = {}
    for i in range(len(sequence) - 1):
        pair = (sequence[i], sequence[i + 1])
        stats[pair] = stats.get(pair, 0) + 1
    return stats


def choose_pair(stats):
    # TODO 4: pilih pasangan dengan frekuensi tertinggi.
    # Hint: tentukan juga aturan konsisten kalau frekuensinya seri.
    max_pair = max(stats, key=stats.get)
    return max_pair


def merge_left_to_right(sequence, pair, new_id):
    # TODO 5: ganti pasangan dari kiri ke kanan.
    # Hint: setelah mengganti dua ID, lanjutkan scan setelah keduanya.
    new_sequence = []
    i = 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
            new_sequence.append(new_id)
            i += 2
            continue
        new_sequence.append(sequence[i])
        i += 1
    return new_sequence


def main():
    print("Skeleton BPE untuk teks:", text)
    print("TODO 1-2: siapkan vocab karakter dan sekuens IDs awal.")
    print("TODO 3-6: hitung stats, pilih pemenang, merge, lalu print hasilnya.")
    print("Target latihan: ulangi proses merge sebanyak tiga ronde.")
    current_ids = ids
    # TODO 6: di setiap ronde, print semua stats, pasangan pemenang,
    for round in range(1, 4):
        stats = get_stats(current_ids)
        pair = choose_pair(stats)
        print("round = ", round, "\nstats = ", stats, "\npair = ", pair)
        new_id = len(vocab)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        current_ids = merge_left_to_right(current_ids, pair, new_id)
        print(f"New ID = {new_id}, teks = {vocab[new_id]}, current ids = {current_ids}")

    # token baru, dan sekuens IDs setelah merge.
    print(f"vocab {vocab} sequence ids = {current_ids}")
    # TODO 7: tambahkan kalimat penutup tentang vocab.json.
    print(
        "Inilah cara vocab.json Lahir: bukan dari pabrik, tapi dari penggabungan pasangan terpopuler berulang kali."
    )


if __name__ == "__main__":
    main()
