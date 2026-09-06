import numpy as np

print("测试 OCR 引擎...")
try:
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    print("✅ OCR 模型加载成功！")
    
    # 用一张白色测试图
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    result, elapse = ocr(img)
    print(f"✅ OCR 调用成功！耗时: {elapse}")
except Exception as e:
    print(f"❌ OCR 失败: {e}")
