"""DyVo re-implementation + extensions (CS6101 course project).

Dynamic Vocabularies for Learned Sparse Retrieval with Entities
(Nguyen et al., EMNLP 2024). Reference code: https://github.com/thongnt99/DyVo
"""
WORD_VOCAB_SIZE = 30522  # BERT/DistilBERT uncased word-piece vocab; entity ids start here


def bf16_ok():
    """bf16 autocast only when the CPU has native bf16 (AMX / AVX512-BF16); otherwise it is
    emulated and ~20x slower than fp32. Override with DYVO_BF16=0/1."""
    import os
    if "DYVO_BF16" in os.environ:
        return os.environ["DYVO_BF16"] == "1"
    try:
        flags = open("/proc/cpuinfo").read()
    except OSError:
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags
