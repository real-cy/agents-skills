#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLM (智谱AI / BigModel) 通用调用客户端
======================================
- 支持非流式 / 流式输出
- 自动展示推理模型的思考过程 (reasoning_content)
- API Key 读取优先级: 命令行 --key > 环境变量 ZHIPU_API_KEY > 同目录 .env

用法:
  # 命令行直接提问
  python glm_client.py "你好，你是谁？"
  python glm_client.py "写一段 Python 代码" -m glm-5.3-flash --max-tokens 4096
  python glm_client.py "不用流式" --no-stream

  # 作为模块引用
  from glm_client import chat, stream_chat
  print(chat("你好"))
"""

import argparse
import json
import os
import sys

import requests

BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 常用模型速查: 名称 -> (是否推理模型, 建议默认 max_tokens)
MODELS = {
    "glm-5.3-flash": (True, 2048),
    "glm-4.7-flash": (True, 2048),
    "glm-4.7": (True, 2048),
    "glm-4.5": (True, 2048),
    "glm-4.5-flash": (True, 2048),
}


def _ensure_utf8_stdout():
    """Windows 控制台默认 GBK，强制 UTF-8 避免中文乱码/报错"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_api_key(cli_key=None):
    """按优先级获取 API Key: 命令行 > 环境变量 > 同目录 .env"""
    if cli_key:
        return cli_key
    env = os.environ.get("ZHIPU_API_KEY")
    if env:
        return env
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "ZHIPU_API_KEY":
                        return v.strip().strip('"').strip("'")
    return None


def chat(messages, model="glm-5.3-flash", max_tokens=None, temperature=None, api_key=None):
    """非流式调用，返回完整响应 dict。

    messages: [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
    返回响应 JSON，正文在 resp["choices"][0]["message"]["content"]，
    思考过程在 ["reasoning_content"]。
    """
    key = load_api_key(api_key)
    if not key:
        raise RuntimeError("未找到 API Key，请用 --key 传入、设置环境变量 ZHIPU_API_KEY，或在脚本同目录 .env 中配置")

    if max_tokens is None:
        max_tokens = MODELS.get(model, (False, 1024))[1]

    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        payload["temperature"] = temperature

    resp = requests.post(
        BASE_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API 错误 {resp.status_code}: {resp.text}")
    return resp.json()


def stream_chat(messages, model="glm-5.3-flash", max_tokens=None, temperature=None,
                api_key=None, show_reasoning=False, on_delta=None):
    """流式调用。SSE 逐块返回。

    on_delta: 可选回调，收到 (kind, text) 其中 kind 为 "reasoning" 或 "content"。
    返回 (完整正文, 完整思考过程, 用量 dict)。
    """
    key = load_api_key(api_key)
    if not key:
        raise RuntimeError("未找到 API Key，请用 --key 传入、设置环境变量 ZHIPU_API_KEY，或在脚本同目录 .env 中配置")

    if max_tokens is None:
        max_tokens = MODELS.get(model, (False, 1024))[1]

    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True}
    if temperature is not None:
        payload["temperature"] = temperature

    reasoning, content, usage = "", "", None
    with requests.post(
        BASE_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
        stream=True,
    ) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"API 错误 {resp.status_code}: {resp.text}")
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            r_text = delta.get("reasoning_content", "")
            c_text = delta.get("content", "")
            if r_text:
                reasoning += r_text
                if on_delta:
                    on_delta("reasoning", r_text)
                elif show_reasoning:
                    sys.stdout.write(r_text)
                    sys.stdout.flush()
            if c_text:
                content += c_text
                if on_delta:
                    on_delta("content", c_text)
                else:
                    sys.stdout.write(c_text)
                    sys.stdout.flush()
            if chunk.get("usage"):
                usage = chunk["usage"]
    return content, reasoning, usage


def _cli():
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="GLM (智谱AI) 通用调用客户端")
    parser.add_argument("prompt", nargs="*", help="要提问的内容（可多段，自动拼接）")
    parser.add_argument("-m", "--model", default="glm-5.3-flash", help="模型名，默认 glm-5.3-flash")
    parser.add_argument("--key", help="API Key（也可用环境变量 ZHIPU_API_KEY 或同目录 .env）")
    parser.add_argument("--max-tokens", type=int, default=None, help="最大 token 数（推理模型建议 >=2048）")
    parser.add_argument("--temperature", type=float, default=None, help="采样温度")
    parser.add_argument("--system", default=None, help="可选的 system 提示词")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    parser.add_argument("--show-reasoning", action="store_true", help="流式时也展示思考过程")
    args = parser.parse_args()

    prompt = " ".join(args.prompt) if args.prompt else None
    if not prompt:
        prompt = input("请输入问题: ").strip()
    if not prompt:
        parser.error("需要提供问题内容")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    try:
        if args.no_stream:
            resp = chat(messages, model=args.model, max_tokens=args.max_tokens,
                        temperature=args.temperature, api_key=args.key)
            msg = resp["choices"][0]["message"]
            if msg.get("reasoning_content"):
                print("── 思考过程 ──")
                print(msg["reasoning_content"])
                print("──────────────")
            print(msg.get("content", ""))
            print(f"\n[模型: {resp.get('model')}  用量: {json.dumps(resp.get('usage', {}), ensure_ascii=False)}]")
        else:
            print("── 思考过程 ──" if args.show_reasoning else "── 回复 ──")
            content, reasoning, usage = stream_chat(
                messages, model=args.model, max_tokens=args.max_tokens,
                temperature=args.temperature, api_key=args.key,
                show_reasoning=args.show_reasoning,
            )
            print(f"\n[模型: {args.model}  用量: {json.dumps(usage, ensure_ascii=False) if usage else 'N/A'}]")
    except Exception as e:
        print(f"调用失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
