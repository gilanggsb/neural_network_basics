import torch
from torch import nn

kalimat = "kucing mengejar tikus karena ia takut"
vocab = {"<pad>": 0}

# TODO 1: tambahkan tiap kata unik dari kalimat ke vocab.
# Petunjuk: kata pertama sesudah <pad> harus punya ID 1.
for i, kata in enumerate(kalimat.split()):
    if kata in vocab:
        continue

    vocab[kata] = len(vocab)

print(vocab)


# TODO 2: set seed ke 42, lalu buat nn.Embedding dengan dimensi 4.
torch.manual_seed(42)
embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=4)

# TODO 3: cetak mapping dan setiap baris embedding.weight dengan label kata.
print("=== Mapping ===")
print("shape:", tuple(embedding.weight.shape))
for k, v in vocab.items():
    print(f"key={k} vektor={embedding.weight[v].detach().numpy().round(4)}")
# TODO 4: ambil ID untuk kata "ia" dari vocab.
token_id_ia = vocab["ia"]
# TODO 5: buat tensor berisi satu token ID tersebut.
input_ids = torch.tensor([token_id_ia])

# TODO 6: ambil vektor dengan dua cara:
# a) embedding.weight[token_id_ia]
print("\n=== Ambil vektor ===")
vecA = embedding.weight[token_id_ia]
# b) embedding(input_ids)
lookup_result = embedding(input_ids)
# Cetak shape input dan output, bandingkan nilainya dengan torch.equal.
print("Shape Input: ", input_ids.shape)
print("Shape Output: ", lookup_result.shape)
print("Bandingin Vektor a dan b", torch.equal(vecA, lookup_result.squeeze(0)))

if __name__ == "__main__":
    print("=== STEP 1: token ID dan tabel embedding ===")
    print("Token ID = ", token_id_ia)
    print("Result Embedding", lookup_result)
