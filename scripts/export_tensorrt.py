#!/usr/bin/env python3
"""
TensorRT export and optimization for mini-GPT.

Exports ONNX model to TensorRT engine for maximum inference performance.
Supports FP16, INT8 calibration, and dynamic shapes.

Usage:
  python scripts/export_tensorrt.py --onnx checkpoints/gpt_rope.onnx --fp16 --output checkpoints/gpt_rope.trt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:
    import tensorrt as trt
except ImportError:
    print("TensorRT/PyCUDA not installed. Install with:")
    print("  pip install tensorrt pycuda")
    print("  Or use the NGC TensorRT container")
    trt = None


def _import_pycuda():
    """Lazy import pycuda."""
    try:
        import pycuda.driver as cuda
        return cuda
    except ImportError:
        raise RuntimeError("PyCUDA not installed. Install with: pip install pycuda")


def build_engine(onnx_path, engine_path, fp16=True, int8=False, 
                 max_batch_size=32, max_seq_len=2048,
                 workspace_size=4 << 30, calibrator=None):
    """Build TensorRT engine from ONNX model."""
    
    if trt is None:
        raise RuntimeError("TensorRT not available")
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    print(f"Parsing ONNX model from {onnx_path}...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX")
    
    print(f"Network inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  {inp.name}: {inp.shape} ({inp.dtype})")
    
    print(f"Network outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  {out.name}: {out.shape} ({out.dtype})")
    
    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    
    # FP16
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")
    
    # INT8
    if int8 and calibrator is not None:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
        print("INT8 mode enabled")
    
    # Dynamic shapes optimization profile
    profile = builder.create_optimization_profile()
    
    # Assume first input is input_ids [batch, seq_len]
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        if len(inp.shape) == 2:  # [batch, seq_len]
            profile.set_shape(
                inp.name,
                min=(1, 1),
                opt=(max_batch_size // 2, max_seq_len // 2),
                max=(max_batch_size, max_seq_len)
            )
    
    config.add_optimization_profile(profile)
    
    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        raise RuntimeError("Failed to build engine")
    
    # Save engine
    print(f"Saving engine to {engine_path}...")
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print("Engine built successfully!")
    return engine


class TensorRTCalibrator(trt.IInt8Calibrator):
    """INT8 calibrator using training data."""
    
    def __init__(self, data_loader, cache_file="calibration.cache"):
        trt.IInt8Calibrator.__init__(self)
        self.data_loader = data_loader
        self.cache_file = cache_file
        self.data_iter = iter(data_loader)
        self.batch_size = data_loader.batch_size
        self.current_batch = None
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        try:
            self.current_batch = next(self.data_iter)
        except StopIteration:
            return None
        
        # Assuming data_loader returns (input_ids, labels)
        input_ids = self.current_batch[0].numpy()
        
        # Allocate device memory
        cuda = _import_pycuda()
        self.device_input = cuda.mem_alloc(input_ids.nbytes)
        cuda.memcpy_htod(self.device_input, input_ids)
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def run_inference(engine_path, input_ids, max_batch_size=32):
    """Run inference with TensorRT engine."""
    
    if trt is None:
        raise RuntimeError("TensorRT not available")
    
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    
    # Set dynamic input shape
    batch_size, seq_len = input_ids.shape
    context.set_binding_shape(0, (batch_size, seq_len))
    
    # Allocate buffers
    cuda = _import_pycuda()
    import numpy as np
    
    # Input
    d_input = cuda.mem_alloc(input_ids.nbytes)
    cuda.memcpy_htod(d_input, input_ids.astype(np.int64))
    
    # Output - get shape from context
    output_shape = context.get_binding_shape(1)
    output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize
    d_output = cuda.mem_alloc(output_size)
    h_output = np.empty(output_shape, dtype=np.float32)
    
    bindings = [int(d_input), int(d_output)]
    
    # Execute
    context.execute_v2(bindings)
    
    # Copy output back
    cuda.memcpy_dtoh(h_output, d_output)
    
    return h_output


def benchmark_engine(engine_path, batch_sizes=[1, 4, 16], seq_lens=[128, 512, 1024], 
                     warmup=10, runs=100):
    """Benchmark TensorRT engine."""
    
    import time
    import numpy as np
    
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    cuda = _import_pycuda()
    
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            print(f"\nBenchmarking batch={batch_size}, seq_len={seq_len}...")
            
            input_ids = np.random.randint(0, 50000, (batch_size, seq_len), dtype=np.int64)
            
            # Set shape
            context.set_binding_shape(0, (batch_size, seq_len))
            
            # Allocate
            d_input = cuda.mem_alloc(input_ids.nbytes)
            output_shape = context.get_binding_shape(1)
            output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize
            d_output = cuda.mem_alloc(output_size)
            
            bindings = [int(d_input), int(d_output)]
            
            # Warmup
            for _ in range(warmup):
                cuda.memcpy_htod(d_input, input_ids)
                context.execute_v2(bindings)
            
            cuda.Context.synchronize()
            
            # Benchmark
            start = time.time()
            for _ in range(runs):
                cuda.memcpy_htod(d_input, input_ids)
                context.execute_v2(bindings)
            cuda.Context.synchronize()
            end = time.time()
            
            elapsed = (end - start) / runs * 1000  # ms
            tokens = batch_size * seq_len
            tok_per_sec = tokens / (elapsed / 1000)
            
            results.append({
                "batch_size": batch_size,
                "seq_len": seq_len,
                "latency_ms": elapsed,
                "tok_per_sec": tok_per_sec,
            })
            
            print(f"  Latency: {elapsed:.2f} ms, Throughput: {tok_per_sec:.0f} tok/s")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="ONNX model path")
    parser.add_argument("--output", type=str, required=True, help="Output engine path")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16")
    parser.add_argument("--int8", action="store_true", help="Use INT8 (requires calibration)")
    parser.add_argument("--max-batch", type=int, default=32, help="Max batch size")
    parser.add_argument("--max-seq", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--workspace", type=int, default=4, help="Workspace size in GB")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after build")
    parser.add_argument("--calib-data", type=str, help="Calibration data for INT8")
    args = parser.parse_args()
    
    if trt is None:
        print("TensorRT not available. Please install tensorrt and pycuda.")
        print("Or use NVIDIA's TensorRT container: nvcr.io/nvidia/tensorrt:xx.xx-py3")
        return
    
    # Build engine
    build_engine(
        args.onnx, args.output,
        fp16=args.fp16,
        int8=args.int8,
        max_batch_size=args.max_batch,
        max_seq_len=args.max_seq,
        workspace_size=args.workspace << 30,
    )
    
    # Benchmark
    if args.benchmark:
        results = benchmark_engine(args.output)
        
        print("\n=== Benchmark Summary ===")
        print(f"{'Batch':>6} {'SeqLen':>8} {'Latency(ms)':>12} {'Tok/s':>12}")
        print("-" * 40)
        for r in results:
            print(f"{r['batch_size']:>6} {r['seq_len']:>8} {r['latency_ms']:>12.2f} {r['tok_per_sec']:>12.0f}")


if __name__ == "__main__":
    main()