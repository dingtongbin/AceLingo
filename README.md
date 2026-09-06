# AceLingo

本地翻译工具，基于 RapidOCR + CTranslate2，支持文本翻译和图片翻译。

## 功能

- 文本翻译（英译中）
- 图片翻译（OCR 识别 + 翻译）
- 超长文本自动分段翻译，逐段输出
- 翻译历史记录持久化
- 系统托盘常驻

## 安装

需要 Python 3.12+，推荐使用 [uv](https://github.com/astral-sh/uv)：

```bash
# 安装 uv
pip install uv

# 克隆项目
git clone https://github.com/dingtongbin/AceLingo.git
cd AceLingo

# 安装依赖
uv sync

# 下载模型（首次需要）
uv run python scripts/converts.py
```

## 运行

```bash
uv run python acelingo.py
```

## 模型

使用 [Helsinki-NLP/opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) 翻译模型（INT8 量化）。

## 许可证

GPL-3.0
