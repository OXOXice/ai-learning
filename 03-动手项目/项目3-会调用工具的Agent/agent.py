"""最小 Agent 循环：模型提议 → 代码执行 → 结果回传 → 直到不再调工具。"""
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from tools import REGISTRY, SCHEMA

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or None)
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
MAX_STEPS = 8

SYSTEM = """你是一个会使用工具的助手。规则：
1. 任何算术都必须调用 calculator，不要心算。
2. 涉及"今天/现在"必须先调用 get_current_time。
3. 需要文件内容时调用 read_file；失败就看错误信息，换一个路径再试。
4. 拿到足够信息就直接回答，不要多余调用。
5. 最终回答要给出依据（用了哪些工具、得到什么结果）。"""


def run(task: str) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]
    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=SCHEMA, temperature=0)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            print(f"\n[step {step}] 完成")
            return msg.content or ""

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                result = {"ok": False, "error": f"参数不是合法 JSON：{e}"}
            else:
                print(f"\n[step {step}] 调用 {name}({args})")
                fn = REGISTRY.get(name)
                result = fn(**args) if fn else {"ok": False, "error": f"没有工具 {name}"}
            print(f"[step {step}] 结果 {str(result)[:200]}")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})

    return f"达到最大步数 {MAX_STEPS} 仍未完成，请缩小任务范围或检查工具。"


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "现在几点？sandbox 里有什么文件？"
    print(run(q))
