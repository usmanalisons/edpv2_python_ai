# app/utils/text_splitter.py

def split_text(text, chunk_size=1000, overlap=200, min_chunk_size=100):
    chunks = []
    text_length = len(text)
    start = 0
    chunk_number = 1

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end]

        chunks.append({
            "chunk_number": chunk_number,
            "text": chunk_text,
            "chunk_len": len(chunk_text)
        })
        chunk_number += 1

        start += chunk_size - overlap

    merged_chunks = []
    for chunk in chunks:
        if merged_chunks and chunk["chunk_len"] < min_chunk_size:
            merged_chunks[-1]["text"] += " " + chunk["text"]
            merged_chunks[-1]["chunk_len"] = len(merged_chunks[-1]["text"])
        else:
            merged_chunks.append(chunk)

    for i in range(len(merged_chunks) - 1):
        overlap_text = merged_chunks[i]["text"][-overlap:]
        merged_chunks[i + 1]["text"] = overlap_text + merged_chunks[i + 1]["text"]
        merged_chunks[i + 1]["chunk_len"] = len(merged_chunks[i + 1]["text"])

    return merged_chunks