"""最小 RAG：切块 + 内存向量检索 + 带引用的回答。

不依赖向量库，先把原理跑通；跑通后再换成 Chroma/FAISS 等。
"""
import hashlib
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)
CHAT_MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DOC_DIR = Path("docs")
CHUNK, OVERLAP, TOP_K = 500, 50, 3


def split(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK])
        start += CHUNK - OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def build_index() -> tuple[list[dict], list[list[float]]]:
    chunks: list[dict] = []
    for path in sorted(DOC_DIR.glob("**/*")):
        if path.suffix.lower() not in {".md", ".txt", ".py", ".json"}:
            continue
        for i, piece in enumerate(split(path.read_text(encoding="utf-8", errors="ignore"))):
            chunks.append({"id": f"{path.name}#{i}", "text": piece,
                           "hash": hashlib.md5(piece.encode()).hexdigest()[:8]})
    print(f"[index] {len(chunks)} chunks from {DOC_DIR}")
    return chunks, embed([c["text"] for c in chunks])


def answer(question: str, chunks: list[dict], vectors: list[list[float]]) -> str:
    qv = embed([question])[0]
    scored = sorted(zip(chunks, vectors), key=lambda cv: cos(qv, cv[1]), reverse=True)[:TOP_K]
    ctx = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c, _ in scored)
    print("\n[retrieved]", ", ".join(f"{c['id']}({cos(qv, v):.2f})" for c, v in scored))

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content":
             "只依据提供的资料回答。每条结论后用 [片段ID] 标注来源。"
             "资料里没有的内容，直接回答「资料中没有提到」，不要推测。"},
            {"role": "user", "content": f"<资料>\n{ctx}\n</资料>\n\n问题：{question}"},
        ],
    )
    return resp.choices[0].message.content


def main() -> None:
    chunks, vectors = build_index()
    print("输入问题，空行退出。")
    while True:
        q = input("\n问 > ").strip()
        if not q:
            break
        print("\n答 >", answer(q, chunks, vectors))


if __name__ == "__main__":
    main()
