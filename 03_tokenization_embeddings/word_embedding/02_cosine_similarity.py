import numpy as np

makan = np.array([0.9, 0.1])
apel_awal = np.array([0.5, 0.5])
perusahaan = np.array([0.1, 0.9])


def cosine_similarity(a, b):
    # TODO 1: hitung dot product dibagi hasil kali dua normanya.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# TODO 2: cetak cosine apel_awal terhadap makan dan perusahaan.
# Apa yang bisa disimpulkan kalau kedua nilainya sama?
print("apel_awal = ", apel_awal)
print("perusahaan = ", perusahaan)
print("makan = ", makan)
food_similarity = cosine_similarity(apel_awal, makan)
company_similarity = cosine_similarity(apel_awal, perusahaan)
print("food_similarity = ", food_similarity)
print("company_similarity = ", company_similarity)

# TODO 3: buat apel_baru = 0.8*makan + 0.2*apel_awal.
# Cetak apel_awal, apel_baru, lalu cosine keduanya terhadap dua kandidat.
apel_baru = (0.8 * makan) + (0.2 * apel_awal)
print("apel_baru = ", apel_baru)
print("apel_awal = ", apel_awal)
new_food_similarity = cosine_similarity(apel_baru, makan)
new_company_similarity = cosine_similarity(apel_baru, perusahaan)
print("Aha: apel sekarang lebih dekat ke makanan -> efek bunglon.")
print("new_food_similarity = ", new_food_similarity)
print("new_company_similarity = ", new_company_similarity)

if __name__ == "__main__":
    print("=== STEP 2: kemiripan statis lalu vektor berubah ===")
