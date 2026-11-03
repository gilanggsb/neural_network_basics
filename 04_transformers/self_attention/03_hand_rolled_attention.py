import numpy as np

vektor_kata = {
    "kucing": np.array([2, 0]),
    "mengejar": np.array([1, 0]),  # TODO: pilih vektor sederhana untuk kata ini.
    "tikus": np.array([-2, 2]),
    "karena": np.array([0, 0]),  # TODO: pilih vektor sederhana untuk kata ini.
    "ia": np.array([0, 0]),
    "takut": np.array([0, 10]),
    "lapar": np.array([0, -10]),
}

# TODO 1: tentukan W_query, W_key, dan W_value secara manual.
# W_query bisa dimulai dari matriks identitas. Susun W_key agar target
# ronde kedua bisa dibedakan dari vektor konteks yang terbentuk.
W_query = np.eye(2)
W_key = np.array([[0, -0.5], [0, 0]])
W_value = np.eye(2)


def softmax(skor):
    # TODO 2: stabilkan skor dengan mengurangi nilai maksimum, lalu normalisasi.
    skor_geser = skor - np.max(skor)
    return np.exp(skor_geser) / np.sum(np.exp(skor_geser))


if __name__ == "__main__":
    print("=== STEP 3: self-attention satu head, tanpa skala ===")
    print("Dimensi 1 = agresor/pengejar; dimensi 2 = korban/takut.")

    hasil_r2 = {}
    for kata_akhir in ("takut", "lapar"):
        words = ["kucing", "mengejar", "tikus", "karena", "ia", kata_akhir]

        # TODO 3: susun X dari vektor kata, lalu hitung Q, K, V memakai np.matmul.
        # Cetak shape dan matriksnya supaya proyeksi manualnya terlihat.
        X = np.array([vektor_kata[w] for w in words])
        print("Shape X = ", X.shape)
        print("CEK X = ", X)
        Q = np.matmul(X, W_query)
        K = np.matmul(X, W_key)
        V = np.matmul(X, W_value)
        print("CEK Q = ", Q)
        print("CEK K = ", K)
        print("CEK V = ", V)

        # TODO 4 — ronde 1:
        # ambil Q untuk "ia", hitung skor terhadap semua K, softmax-kan,
        # lalu jumlahkan bobot * V menjadi vektor baru untuk "ia".
        pos_ia = words.index("ia")
        print("pos ia ", pos_ia)
        Q_ia = Q[pos_ia]
        print("Q_ia: ", Q_ia)
        skor_r1 = np.matmul(Q_ia, K.T)
        print("skor mentah: ", skor_r1)
        bobot_r1 = softmax(skor_r1)
        print("bobot r1: ", bobot_r1)
        baru_ia = np.sum(bobot_r1[:, None] * V, axis=0)
        print("baru_ia: ", baru_ia)

        # TODO 5 — ronde 2:
        # proyeksikan vektor baru menjadi Q baru, hitung ulang skor dan bobot.
        # Simpan bobot ronde 2 agar bisa dibandingkan untuk dua kalimat.
        Q_ia_new = np.matmul(baru_ia, W_query)
        print("Q_ia_new: ", Q_ia_new)
        skor_r2 = np.matmul(Q_ia_new, K.T)
        print("skor r2: ", skor_r2)
        bobot_r2 = softmax(skor_r2)
        print("bobot r2: ", bobot_r2)
        hasil_r2[kata_akhir] = bobot_r2

    # TODO 6: cetak berdampingan. Target: takut -> TIKUS; lapar -> KUCING.
    # Jelaskan kenapa Q berubah setelah vektor ia menyerap V di ronde 1.
    target_takut = np.argmax(hasil_r2["takut"])
    target_lapar = np.argmax(hasil_r2["lapar"])
    print(
        f"... ia takut -> TIKUS "
        f"({hasil_r2['takut'][target_takut]:.3f}) | "
        f"... ia lapar -> KUCING "
        f"({hasil_r2['lapar'][target_lapar]:.3f})"
    )
    print(
        "Q ronde 2 beda karena vektor ia di ronde 1 sudah menyerap V tetangganya. "
        "Isi kepala dulu, baru pertanyaan berubah."
    )
