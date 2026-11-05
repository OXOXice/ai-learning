"""每日简报：抓 RSS → 去重 → 摘要 → 打印/推送。

先跑通 dry-run，再接定时任务。
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or None)
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
STATE = Path("state/seen.json")
FEEDS = [
    "https://hnrss.org/frontpage",
    # 换成你真正会读的源：博客、期刊、社区
]

PROMPT = """你是我的信息助理。基于下面这些条目，输出：
1. 三条今天最值得知道的（每条一句话，不超过 40 字，带来源）
2. 一条值得深读的（说明为什么）
3. 一条可以直接忽略的（说明为什么是噪音）
规则：只依据给定条目，不确定的不写，总长度不超过 200 字。"""


def load_seen() -> set[str]:
    return set(json.loads(STATE.read_text())) if STATE.exists() else set()


def save_seen(seen: set[str]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(sorted(seen)[-2000:], ensure_ascii=False))


def fetch() -> list[dict]:
    try:
        import feedparser
    except ImportError:
        sys.exit("需要 feedparser：pip install feedparser")
    cutoff = datetime.now() - timedelta(days=1)
    items = []
    for url in FEEDS:
        try:
            for e in feedparser.parse(url).entries[:10]:
                t = e.get("published_parsed") or e.get("updated_parsed")
                if t and datetime(*t[:6]) < cutoff:
                    continue
                items.append({"title": e.get("title", ""), "link": e.get("link", ""),
                              "src": url.split("/")[2]})
        except Exception as ex:  # 单个源失败不影响整体
            print(f"[warn] {url} 抓取失败：{ex}")
    return items


def summarize(items: list[dict]) -> str:
    lines = "\n".join(f"- [{i['src']}] {i['title']} ({i['link']})" for i in items[:15])
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.3,
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": lines}])
    return resp.choices[0].message.content


def main() -> None:
    seen = load_seen()
    items = [i for i in fetch() if i["link"] not in seen]
    if not items:
        print(f"[{datetime.now():%Y-%m-%d}] 今天没有新条目（或抓取失败）。")
        return
    brief = summarize(items)
    print(f"\n=== {datetime.now():%Y-%m-%d} 简报 ===\n{brief}\n")
    save_seen(seen | {i["link"] for i in items})
    # TODO 第 8 周：接手机通知 / 邮件 / 写入 /sdcard/Download


if __name__ == "__main__":
    main()
