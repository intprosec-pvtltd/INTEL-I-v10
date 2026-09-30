"""Stage a native TensorRT model repository from a reviewed dynamic ONNX graph.

This is an explicit model-build/benchmark tool, not a second production path.
The compatibility pipeline uses Ultralytics .engine metadata; native plans
must pass preprocessing/postprocessing parity before a client switches to them.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("onnx", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--shape", required=True, help="non-batch input dimensions, e.g. 3,640,640")
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--queue-us", type=int, default=5000)
    p.add_argument("--repository", type=Path, default=Path("triton_native_models"))
    a = p.parse_args()
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.name):
        p.error("invalid model name")
    if not a.onnx.is_file() or not shutil.which("trtexec"):
        p.error("existing ONNX file and TensorRT trtexec are required")
    shape = [int(x) for x in a.shape.split(",")]
    if any(x <= 0 for x in shape) or not 1 <= a.max_batch <= 32 or not 0 <= a.queue_us <= 100000:
        p.error("invalid shape, batch or queue delay")
    import onnx
    graph = onnx.load(str(a.onnx))
    onnx.checker.check_model(graph)
    initializers = {x.name for x in graph.graph.initializer}
    inputs = [x for x in graph.graph.input if x.name not in initializers]
    if len(inputs) != 1:
        p.error("multi-input models require an explicit reviewed build profile")
    source_input = inputs[0]
    dims = source_input.type.tensor_type.shape.dim
    if len(dims) != len(shape) + 1 or dims[0].HasField("dim_value"):
        p.error("export a dynamic batch ONNX graph first; batch axis must be axis zero")
    if source_input.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        p.error("tool expects FP32 I/O with FP16 internal TensorRT precision")
    target = a.repository / a.name
    (target / "1").mkdir(parents=True, exist_ok=True)
    plan = target / "1/model.plan"
    def profile(batch):
        return source_input.name + ":" + "x".join(map(str, [batch] + shape))
    subprocess.run(["trtexec", f"--onnx={a.onnx.resolve()}", f"--saveEngine={plan.resolve()}",
                    "--fp16", "--memPoolSize=workspace:2048",
                    f"--minShapes={profile(1)}", f"--optShapes={profile(min(8,a.max_batch))}",
                    f"--maxShapes={profile(a.max_batch)}"], check=True)
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(plan.read_bytes())
        if engine is None:
            raise RuntimeError("engine deserialization failed")
        io = {"input": [], "output": []}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_dtype(name) != trt.float32:
                raise ValueError(f"unexpected I/O dtype for {name}; review config explicitly")
            key = "input" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "output"
            tensor_shape = list(engine.get_tensor_shape(name))[1:]
            io[key].append(f'{{ name: {json.dumps(name)} data_type: TYPE_FP32 dims: {tensor_shape} }}')
    preferred = [x for x in (4,8,16) if x <= a.max_batch] or [1]
    config = f'name: {json.dumps(a.name)}\nplatform: "tensorrt_plan"\nmax_batch_size: {a.max_batch}\n'
    for key in ("input","output"):
        config += key + ' [\n' + '\n'.join(io[key]) + '\n]\n'
    config += f'instance_group [{{ count: 1 kind: KIND_GPU gpus: [0] }}]\ndynamic_batching {{ preferred_batch_size: {preferred} max_queue_delay_microseconds: {a.queue_us} default_queue_policy {{ max_queue_size: 100 timeout_action: REJECT default_timeout_microseconds: 500000 }} }}\n'
    (target / "config.pbtxt").write_text(config)
    (target / "build.json").write_text(json.dumps({"onnx_sha256":hashlib.sha256(a.onnx.read_bytes()).hexdigest(),"plan_sha256":hashlib.sha256(plan.read_bytes()).hexdigest(),"accuracy_validated":False},indent=2))
    print(f"Built {target}; parity validation required before production use")


if __name__ == "__main__":
    main()
