# convert_model.py
import ctranslate2
import ssl
import torch  # noqa: F401 - ctranslate2 converter requires torch

ssl._create_default_https_context = ssl._create_unverified_context

print("开始下载并转换翻译模型到 INT8 格式，这可能需要几分钟...")

model_name = "Helsinki-NLP/opus-mt-en-zh"

converter = ctranslate2.converters.TransformersConverter(model_name)
converter.convert("en-zh-int8", quantization="int8")

print("模型转换完成！保存在 en-zh-int8 文件夹中。")

# 如果国内网络不通则在 powershell 用这个
# $env:HF_ENDPOINT = "https://hf-mirror.com"
# $env:SSL_CERT_FILE = ""
# $env:REQUESTS_CA_BUNDLE = ""
# $env:CURL_CA_BUNDLE = ""

# uv run python .\scripts\converts.py
