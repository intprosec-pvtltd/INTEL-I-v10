# Awiros Indian ANPR OCR deployment

The application does not download executable code or model weights at runtime.
Provision these reviewed assets before starting INTEL-I:

```text
models/awiros_anpr_ocr/model.safetensors
models/awiros_anpr_ocr/en_dict.txt
vendor/PaddleOCR/ppocr/__init__.py
```

Download `model.safetensors` and `en_dict.txt` from the official
`Awiros/anpr-ocr` release. Clone the official PaddleOCR repository into
`vendor/PaddleOCR`, review it, and pin the exact commit in deployment records.
Use a PaddlePaddle GPU wheel compatible with the deployment CUDA version; the
portable requirements file contains the CPU package for local validation.

Copy the ANPR settings from `.env.production.example` into the private runtime
environment. Never place credentials or camera URLs in the example file.

To roll back deliberately, set `ANPR_OCR_ENGINE=ppocrv5_onnx`. Awiros startup
failure does not silently select the old engine because that could hide an
incorrect production deployment.
