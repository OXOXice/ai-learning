"""最小可用：一次调用，看懂请求/响应的结构。"""
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")


def ask(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
    )
    u = resp.usage
    print(f"[tokens] in={u.prompt_tokens} out={u.completion_tokens}", file=sys.stderr)
    return resp.choices[0].message.content


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "用一句话解释什么是 token"
    print(ask(q))
