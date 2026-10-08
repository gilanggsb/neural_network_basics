# 06 — Fine-Tuning & AI Agents

> **Tujuan:** Mengubah perilaku LLM (bukan sekadar menambah data) dan men-deploy LLM ke dunia nyata agar bisa bertindak secara mandiri.

## Status: 🔒 Belum Dimulai

**Prerequisite:** Selesaikan folder `05_rag_and_llm` terlebih dahulu.

---

## Roadmap Materi (Level 5 & 6 — Modern NLP/LLM Engineer)

### Level 5 — Efficient Fine-Tuning (PEFT)

#### RAG vs Fine-Tuning: Kapan Pakai Apa?

| Kebutuhan | Solusi |
|---|---|
| Tambah pengetahuan baru | **RAG** |
| Ubah cara/gaya bicara LLM | **Fine-Tuning** |
| Format output spesifik | **Fine-Tuning** |
| Data yang sering berubah | **RAG** |
| Kurangi ukuran prompt | **Fine-Tuning** |

#### LoRA (Low-Rank Adaptation)
Melatih hanya sebagian kecil parameter model — bukan semua miliaran parameter.

```
Full Fine-Tuning: Update 7 Miliar parameter (butuh GPU A100 x8)
         vs
LoRA:             Update ~4 Juta parameter saja (bisa di GPU lokal!)
```

- Menginsert matrix kecil (adapter) di setiap layer Transformer
- Efisien: Menghemat 99%+ memori GPU
- **QLoRA** — LoRA + Quantization (4-bit) → bisa fine-tune Llama 7B dengan GPU 24GB

---

### Level 6 — AI Agents, Tool Use & LLMOps

#### Agents & Function Calling
LLM yang tidak hanya menjawab, tapi bisa **memicu aksi nyata**.

```
User: "Cek cuaca Jakarta hari ini dan kirim email ke Tim"
         ↓
LLM memutuskan tool mana yang dipanggil:
  → get_weather(location="Jakarta")
  → send_email(to="tim@...", body=...)
```

#### Model Context Protocol (MCP)
Standar terbuka untuk konektivitas antara LLM dan tools/resources eksternal.

#### LLMOps
- **vLLM** — High-performance inference server (throughput tinggi dengan PagedAttention)
- **Guardrails** — Filter input/output agar AI tidak menjawab hal berbahaya
- **Latency & Cost monitoring** — Metrik penting untuk production deployment

---

## Proyek yang Akan Dibuat

| Proyek | Konsep | Status |
|---|---|---|
| `lora_finetuning/` | Fine-tune SLM dengan LoRA | 🔲 Belum |
| `qlora_finetuning/` | QLoRA dengan 4-bit quantization | 🔲 Belum |
| `simple_agent/` | Agent dengan function calling dasar | 🔲 Belum |
| `mcp_integration/` | Integrasi MCP server | 🔲 Belum |
| `guardrails_demo/` | Input/output filtering | 🔲 Belum |

---
*Level 6 — Production AI: Fine-Tuning, Agents, dan LLMOps*
