# GLM (智谱AI) 调用工具

独立通用的 GLM 大模型调用客户端，可在任意项目里引用。

## 文件

| 文件 | 说明 |
|---|---|
| `glm_client.py` | 主脚本：命令行工具 + 可 import 的模块 |
| `.env` | API Key 配置（本地私密文件，勿外传） |

## 快速开始

```bash
# 命令行直接提问（默认模型 glm-5.3-flash）
python glm_client.py "你好，你是谁？"

# 指定模型、加大 token 上限
python glm_client.py "写一段 Python 冒泡排序" -m glm-5.3-flash --max-tokens 4096

# 非流式 + 显示思考过程
python glm_client.py "1+1=?" --no-stream

# 带 system 提示词
python glm_client.py "总结这段文字" --system "你是严谨的编辑"
```

## 在代码里引用

```python
import sys
sys.path.insert(0, r"C:\Users\Administrator\.agents\tools\glm_api")
from glm_client import chat, stream_chat

# 非流式
resp = chat([{"role": "user", "content": "你好"}])
print(resp["choices"][0]["message"]["content"])

# 流式（实时打印）
content, reasoning, usage = stream_chat(
    [{"role": "user", "content": "讲个笑话"}],
    show_reasoning=True,
)
```

## API Key 三种配置方式（按优先级）

1. 命令行 `--key xxx`
2. 环境变量 `ZHIPU_API_KEY`
3. 同目录 `.env` 文件（当前已配置）

## 重要注意事项

- **`glm-5.3-flash` 是推理模型**：回答前会生成 `reasoning_content` 思考过程，
  `max_tokens` 太小（如 200）时正文可能为空，建议 ≥ 2048。
- 接口为 OpenAI 兼容格式：`https://open.bigmodel.cn/api/paas/v4/chat/completions`。
- 其他模型名可直接替换 `-m`，如 `glm-4.7-flash`（免费）。

## 接口速查

- 官方文档: <https://docs.bigmodel.cn/>
- GLM-5.3-Flash 介绍: <https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash>
