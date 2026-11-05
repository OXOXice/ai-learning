"""多轮 + 流式 + 命令。v1 骨架，v2 自己加：历史压缩、花费统计、system prompt 热改。"""
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

SYSTEM = "你是一个简洁的中文助手。回答抓重点，不确定就说不确定。"
history: list[dict] = [{"role": "system", "content": SYSTEM}]
temperature = 0.7

HELP = """命令：
  /reset        清空对话历史
  /temp 0.3     设置温度
  /help         显示帮助
  /exit         退出
"""


def stream_reply() -> str:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=history,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},  # 部分服务支持；不支持就删掉这行
    )
    parts: list[str] = []
    for chunk in stream:
        if getattr(chunk, "usage", None):
            u = chunk.usage
            print(f"\n[tokens] in={u.prompt_tokens} out={u.completion_tokens}")
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
            print(delta, end="", flush=True)
    print()
    return "".join(parts)


def main() -> None:
    global temperature
    print(HELP)
    while True:
        try:
            user = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history[:] = [{"role": "system", "content": SYSTEM}]
            print("[已清空]")
            continue
        if user == "/help":
            print(HELP)
            continue
        if user.startswith("/temp"):
            try:
                temperature = float(user.split()[1])
                print(f"[温度 = {temperature}]")
            except (IndexError, ValueError):
                print("[用法：/temp 0.3]")
            continue

        history.append({"role": "user", "content": user})
        reply = stream_reply()
        history.append({"role": "assistant", "content": reply})

        # 简单保护：历史过长就砍掉最早的几轮（v2 改成摘要压缩）
        if len(history) > 21:
            history[1:3] = []
            print("[历史过长，已丢弃最早一轮]")


if __name__ == "__main__":
    main()
