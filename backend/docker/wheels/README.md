Place the operator-reviewed paddlepaddle_gpu wheel here before building the Triton image.
Match the NVIDIA image's Python ABI, CUDA, and cuDNN versions. Pass its filename
as PADDLE_GPU_WHEEL. This archive contains no model files or third-party wheels.
The Docker build intentionally fails without this input; it does not silently use CPU OCR.
