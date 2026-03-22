#!/usr/bin/env python3
"""Convert Nemotron ONNX models from FP32 to FP16.

Reads from models/fp32/, writes to models/fp16/.
Converts all weights, constants, and Cast nodes to FP16.
Wraps I/O with Cast nodes so external interface stays FP32.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "models" / "fp32"
DST_DIR = PROJECT_DIR / "models" / "fp16"

ONNX_MODELS = [
    "encoder-model.onnx",
    "encoder-model-streaming.onnx",
    "decoder_joint-model.onnx",
    "decoder_joint-model-streaming.onnx",
]

COPY_FILES = [
    "filterbank.bin",
    "filterbank.meta",
    "tokens.txt",
    "preprocessor.config",
]


def convert_initializers_to_fp16(model: onnx.ModelProto) -> None:
    """Convert all FP32 initializers to FP16 in-place."""
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init).astype(np.float16)
            new_tensor = numpy_helper.from_array(arr, init.name)
            init.CopyFrom(new_tensor)


def convert_constant_nodes_to_fp16(model: onnx.ModelProto) -> None:
    """Convert all Constant nodes with FP32 values to FP16."""
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t).astype(np.float16)
                    new_tensor = numpy_helper.from_array(arr, attr.t.name or "")
                    attr.t.CopyFrom(new_tensor)


def convert_cast_nodes_to_fp16(model: onnx.ModelProto) -> None:
    """Update Cast nodes that output FLOAT to output FLOAT16 instead."""
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16


def update_type_info(model: onnx.ModelProto) -> None:
    """Update value_info type annotations from FLOAT to FLOAT16."""
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16
    for out in model.graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT:
            out.type.tensor_type.elem_type = TensorProto.FLOAT16


def wrap_io_with_casts(model: onnx.ModelProto) -> onnx.ModelProto:
    """Add Cast(FP32->FP16) at float16 inputs and Cast(FP16->FP32) at float16 outputs.

    Keeps external API as FP32 so Zig code doesn't need changes.
    Non-float I/O (int32, int64) is left untouched.
    """
    graph = model.graph
    new_nodes = list(graph.node)
    new_inputs = []
    new_outputs = []

    for inp in graph.input:
        elem_type = inp.type.tensor_type.elem_type
        if elem_type == TensorProto.FLOAT16:
            # Keep original name for external FP32 input, rename internal FP16 edge
            fp16_internal = inp.name + "_fp16_internal"
            fp32_input = helper.make_tensor_value_info(
                inp.name, TensorProto.FLOAT,
                [d.dim_value if d.dim_value else d.dim_param
                 for d in inp.type.tensor_type.shape.dim],
            )
            new_inputs.append(fp32_input)
            # Rename all uses of the old input name in graph nodes
            for node in new_nodes:
                for i, ni in enumerate(node.input):
                    if ni == inp.name:
                        node.input[i] = fp16_internal
            cast_node = helper.make_node(
                "Cast", [inp.name], [fp16_internal],
                to=TensorProto.FLOAT16,
                name=f"cast_input_{inp.name}",
            )
            new_nodes.insert(0, cast_node)
        else:
            new_inputs.append(inp)

    for out in graph.output:
        elem_type = out.type.tensor_type.elem_type
        if elem_type == TensorProto.FLOAT16:
            fp16_name = out.name + "_fp16"
            for node in new_nodes:
                for i, o in enumerate(node.output):
                    if o == out.name:
                        node.output[i] = fp16_name
            cast_node = helper.make_node(
                "Cast", [fp16_name], [out.name],
                to=TensorProto.FLOAT,
                name=f"cast_output_{out.name}",
            )
            new_nodes.append(cast_node)
            fp32_output = helper.make_tensor_value_info(
                out.name, TensorProto.FLOAT,
                [d.dim_value if d.dim_value else d.dim_param
                 for d in out.type.tensor_type.shape.dim],
            )
            new_outputs.append(fp32_output)
        else:
            new_outputs.append(out)

    new_graph = helper.make_graph(
        new_nodes, graph.name, new_inputs, new_outputs,
        initializer=list(graph.initializer),
    )
    new_model = helper.make_model(new_graph)
    new_model.ir_version = model.ir_version
    del new_model.opset_import[:]
    new_model.opset_import.MergeFrom(model.opset_import)
    return new_model


def convert_model(src_path: Path, dst_path: Path) -> None:
    print(f"Loading {src_path.name}...")
    model = onnx.load(str(src_path), load_external_data=True)

    print(f"  Converting initializers to FP16...")
    convert_initializers_to_fp16(model)

    print(f"  Converting Constant nodes to FP16...")
    convert_constant_nodes_to_fp16(model)

    print(f"  Converting Cast node targets to FP16...")
    convert_cast_nodes_to_fp16(model)

    print(f"  Updating type annotations...")
    update_type_info(model)

    print(f"  Adding I/O Cast wrappers (FP32 external interface)...")
    model = wrap_io_with_casts(model)

    print(f"  Saving to {dst_path.name}...")
    onnx.save(
        model,
        str(dst_path),
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=1024,
    )
    print(f"  Done.")


def main() -> None:
    if not SRC_DIR.exists():
        print(f"ERROR: Source directory not found: {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    DST_DIR.mkdir(parents=True, exist_ok=True)

    for name in ONNX_MODELS:
        src = SRC_DIR / name
        dst = DST_DIR / name
        if not src.exists():
            print(f"SKIP: {name} not found in source")
            continue
        convert_model(src, dst)

    for name in COPY_FILES:
        src = SRC_DIR / name
        dst = DST_DIR / name
        if src.exists():
            shutil.copy2(str(src), str(dst))
            print(f"Copied {name}")

    src_size = sum(f.stat().st_size for f in SRC_DIR.rglob("*") if f.is_file())
    dst_size = sum(f.stat().st_size for f in DST_DIR.rglob("*") if f.is_file())
    print(f"\nFP32 total: {src_size / 1024**3:.2f} GB")
    print(f"FP16 total: {dst_size / 1024**3:.2f} GB")
    print(f"Reduction:  {(1 - dst_size / src_size) * 100:.1f}%")


if __name__ == "__main__":
    main()
