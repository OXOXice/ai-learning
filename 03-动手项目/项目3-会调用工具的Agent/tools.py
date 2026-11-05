"""三个工具。每个都返回结构化的结果，失败不抛异常。"""
import ast
import operator
import os
from datetime import datetime
from pathlib import Path

SANDBOX = Path("sandbox").resolve()
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}


def _eval(node):
    if isinstance(node, ast.Num):  # py<3.8 兼容，可删
        return node.n
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    raise ValueError("只支持 + - * / % ** 和数字")


def calculator(expression: str) -> dict:
    try:
        return {"ok": True, "result": _eval(ast.parse(expression, mode="eval").body)}
    except Exception as e:
        return {"ok": False, "error": f"计算失败：{e}。请只传数学表达式，例如 128*7"}


def get_current_time(_: str = "") -> dict:
    return {"ok": True, "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "weekday": "一二三四五六日"[datetime.now().weekday()]}


def read_file(path: str) -> dict:
    target = (SANDBOX / path).resolve()
    if not str(target).startswith(str(SANDBOX)):
        return {"ok": False, "error": f"只能读 sandbox/ 目录下的文件，拒绝访问 {path}"}
    if not target.is_file():
        return {"ok": False, "error": f"文件不存在：{path}。sandbox/ 下有：{[p.name for p in SANDBOX.glob('*')]}"}
    text = target.read_text(encoding="utf-8", errors="ignore")
    return {"ok": True, "content": text[:2000], "truncated": len(text) > 2000}


REGISTRY = {"calculator": calculator, "get_current_time": get_current_time, "read_file": read_file}

SCHEMA = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "计算数学表达式，支持 + - * / % **。用于任何算术，不要心算。",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string", "description": "如 128*7"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "获取当前本地时间和星期。任何涉及今天/现在/几号的问题都必须先调用。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "读取 sandbox/ 目录下的文本文件，最多返回前 2000 字。只能读 sandbox 内。",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "相对 sandbox/ 的路径"}},
                       "required": ["path"]}}},
]
