"""Skeleton tokenizer BPE karakter-level; isi TODO-nya sedikit demi sedikit."""

CORPUS = [
    "produk ini keren banget",
    "filmnya keren banget sih",
    "keren banget bikin senyum",
    "kopinya enak banget hari ini",
    "menu baru enak dan murah",
    "pelayanan cepat dan ramah",
    "fitur aplikasi ini sangat membantu",
    "belajar bahasa indonesia itu seru",
    "aku suka cara kamu menjelaskan",
    "gw suka belajar hal baru",
    "gw suka ngoding bareng teman",
    "kalau error cek log dulu",
    "ga mutu kalau sering macet",
    "gaskeun kita berangkat sekarang",
    "wkwk typo bangett tapi masih paham",
    "server lokal berjalan dengan baik",
    "jaringan rumah kadang lambat",
    "laptop lama masih bisa dipakai",
    "cita - cita",
]
CORPUS_TEXT = " ".join(CORPUS)
VOCAB_SIZE = 86


class BPECharacterTokenizer:
    def __init__(self):
        # Contoh setelah train("abc", vocab_size=5):
        # self.char_to_id = {"a": 0, "b": 1, "c": 2}
        # self.vocab = {0: "a", 1: "b", 2: "c", 3: "ab", 4: "abc"}
        # self.merges = {(0, 1): 3, (3, 2): 4}
        # self.merge_order = [((0, 1), 3), ((3, 2), 4)]
        # Contohnya komentar saja; train() yang mengisi data sungguhan.
        self.vocab = {}  # TODO: ID -> potongan teks.
        self.char_to_id = {}  # TODO: karakter awal -> ID.
        self.merges = {}  # TODO: pasangan ID -> ID token baru.
        self.merge_order = []  # TODO: simpan urutan merge untuk analisis.

    def get_stats(self, sequence):
        stats = {}
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            stats[pair] = stats.get(pair, 0) + 1
        return stats

    def choose_pair(self, stats):
        return max(stats, key=stats.get)

    def merge_left_to_right(self, sequence, pair, new_id):
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

    def train(self, text, vocab_size):
        # TODO 1: mulai dari karakter unik, lalu siapkan sekuens ID.
        characters = sorted(set(text))
        self.char_to_id = {char: id for id, char in enumerate(characters)}
        self.vocab = {id: char for char, id in self.char_to_id.items()}
        ids = [self.char_to_id[char] for char in text]
        # TODO 2: hitung pasangan, pilih pemenang, merge, simpan vocab/merges.
        while len(self.vocab) < vocab_size:
            stats = self.get_stats(ids)
            if not stats:
                break

            pair = self.choose_pair(stats)
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            ids = self.merge_left_to_right(ids, pair, new_id)
            self.merges[pair] = new_id
            self.merge_order.append((pair, new_id))
        # Berhenti saat ukuran vocab tercapai atau tidak ada pasangan tersisa.

    def encode(self, text):
        # TODO 3: mulai dari ID karakter, lalu terapkan merge yang dipelajari.
        # Hint: merge yang lahir lebih awal punya prioritas lebih dulu.
        ids = [self.char_to_id[char] for char in text]
        while True:
            available = []
            for pair in zip(ids, ids[1:]):
                if pair in self.merges:
                    available.append((self.merges[pair], pair))

            if not available:
                break

            new_id, pair = min(available)
            ids = self.merge_left_to_right(ids, pair, new_id)

        return ids

    def decode(self, ids):
        # TODO 4: gabungkan teks untuk setiap ID menjadi satu string.
        result = ""
        for id in ids:
            result += self.vocab[id]

        return result


def main():
    print("Skeleton tokenizer BPE karakter-level")
    print(f"Jumlah kalimat korpus: {len(CORPUS)}")
    print("TODO 5: latih tokenizer, lalu tampilkan subword kalimat uji.")
    tokenizer = BPECharacterTokenizer()
    tokenizer.train(CORPUS_TEXT, VOCAB_SIZE)
    print("TODO 6: coba kata baru yang huruf-hurufnya sudah dikenal.")
    test_sentence = "gw suka ngegas bangett"
    test_sentence_encoded = tokenizer.encode(test_sentence)

    print("------ BREAK ------")
    print("cekk test sentence enc", test_sentence_encoded)
    test_sentence_decoded = tokenizer.decode(test_sentence_encoded)
    print("cekk test sentence dec", test_sentence_decoded)
    for token_id in test_sentence_encoded:
        print(tokenizer.decode([token_id]))

    print("------ BREAK ------")
    oov_word = "misuh"
    oov_word_enc = tokenizer.encode(oov_word)
    oov_word_dec = tokenizer.decode(oov_word_enc)
    print("cek oov word enc = ", oov_word_enc, " \noov word dec = ", oov_word_dec)
    for token_id in oov_word_enc:
        print(tokenizer.decode([token_id]))

    print("------ BREAK ------")
    print("TODO 7: cek round-trip untuk korpus dan beberapa kalimat uji.")
    list_text = [
        *CORPUS,
        "saya ini anak singkong",
        "bapak saya suka nyawit",
        "cita - cita saya adalah malakin rakyat",
        "ingin jadi penguasa negara",
    ]
    for text in list_text:
        decoded = tokenizer.decode(tokenizer.encode(text))
        print(
            f"isTrue {text == decoded} result = {decoded} ",
        )


if __name__ == "__main__":
    main()
