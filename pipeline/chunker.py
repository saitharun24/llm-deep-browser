import re

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 150, min_chunk_size: int = 100) -> list[str]:
    if not text or not text.strip():
        return []
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]

        if len(chunk_words) < min_chunk_size:
            # Too small to be useful — merge into previous chunk if possible
            if chunks:
                chunks[-1] = chunks[-1] + " " + " ".join(chunk_words)
            continue

        chunk = " ".join(chunk_words)

        # Extend to nearest sentence boundary to avoid mid-sentence cuts
        if i + chunk_size < len(words):
            last_period = max(
                chunk.rfind("."),
                chunk.rfind("!"),
                chunk.rfind("?"),
            )
            if last_period > len(chunk) * 0.5:  # only trim if boundary is in second half
                chunk = chunk[:last_period + 1]

        chunks.append(chunk)

    return chunks