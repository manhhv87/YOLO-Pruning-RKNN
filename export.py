"""
Fix ONNX opset18 → opset11 cho TensorRT 8.2 (Jetson Nano).

Sửa các operator mà opset18 chuyển axes/dims từ attribute → input tensor:
  - ReduceMean
  - Unsqueeze
  - Squeeze

Chạy: python fix_onnx_for_jetson.py
"""

import onnx
from onnx import helper, numpy_helper
import numpy as np
from pathlib import Path

INPUT_DIR = "runs/detect"
OUTPUT_DIR = "/tmp/onnx_fixed"


def get_initializer_value(graph, name):
    """Tìm giá trị tensor từ initializer hoặc Constant node."""
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init).flatten().tolist()
    for node in graph.node:
        if node.op_type == "Constant" and len(node.output) > 0 and node.output[0] == name:
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t).flatten().tolist()
    return None


def fix_onnx(input_path, output_path):
    model = onnx.load(input_path)
    graph = model.graph
    fixed = 0

    for node in graph.node:

        # Fix ReduceMean: axes từ input → attribute
        if node.op_type == "ReduceMean":
            has_axes_attr = any(a.name == "axes" for a in node.attribute)
            if not has_axes_attr and len(node.input) > 1:
                axes = get_initializer_value(graph, node.input[1])
                if axes is not None:
                    while len(node.input) > 1:
                        node.input.pop()
                    node.attribute.append(helper.make_attribute("axes", [int(a) for a in axes]))
                    if not any(a.name == "keepdims" for a in node.attribute):
                        node.attribute.append(helper.make_attribute("keepdims", 1))
                    fixed += 1

        # Fix Unsqueeze: axes từ input → attribute
        if node.op_type == "Unsqueeze":
            has_axes_attr = any(a.name == "axes" for a in node.attribute)
            if not has_axes_attr and len(node.input) > 1:
                axes = get_initializer_value(graph, node.input[1])
                if axes is not None:
                    while len(node.input) > 1:
                        node.input.pop()
                    node.attribute.append(helper.make_attribute("axes", [int(a) for a in axes]))
                    fixed += 1

        # Fix Squeeze: axes từ input → attribute
        if node.op_type == "Squeeze":
            has_axes_attr = any(a.name == "axes" for a in node.attribute)
            if not has_axes_attr and len(node.input) > 1:
                axes = get_initializer_value(graph, node.input[1])
                if axes is not None:
                    while len(node.input) > 1:
                        node.input.pop()
                    node.attribute.append(helper.make_attribute("axes", [int(a) for a in axes]))
                    fixed += 1

    # Xóa initializer + Constant nodes không dùng nữa
    used_inputs = set()
    for node in graph.node:
        for inp in node.input:
            used_inputs.add(inp)
    for inp in graph.input:
        used_inputs.add(inp.name)

    to_remove_init = [i for i in graph.initializer if i.name not in used_inputs]
    for init in to_remove_init:
        graph.initializer.remove(init)

    to_remove_nodes = [n for n in graph.node if n.op_type == "Constant"
                       and len(n.output) > 0 and n.output[0] not in used_inputs]
    for n in to_remove_nodes:
        graph.node.remove(n)

    # Hạ opset xuống 11
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset.version = 11

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return fixed


# Chạy
onnx_files = sorted(Path(INPUT_DIR).rglob("best.onnx"))
print(f"Found {len(onnx_files)} ONNX files\n")

total_ok = 0
for onnx_path in onnx_files:
    model_name = onnx_path.parent.parent.name
    out_path = Path(OUTPUT_DIR) / f"{model_name}.onnx"
    print(f"Fixing: {model_name}...", end=" ")
    try:
        n = fix_onnx(str(onnx_path), str(out_path))
        print(f"OK ({n} nodes fixed)")
        total_ok += 1
    except Exception as e:
        print(f"FAIL: {e}")

print(f"\nDone: {total_ok} files → {OUTPUT_DIR}/")
print(f"\nCopy sang Jetson:")
print(f"  sshpass -p '1' scp {OUTPUT_DIR}/*.onnx manhhv@192.168.1.3:/home/manhhv/models/")