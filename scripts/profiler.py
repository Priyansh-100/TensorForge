#!/usr/bin/env python3
"""
PyTorch Profiler integration for mini-GPT performance analysis.

Usage:
  python scripts/profile.py --mode train --steps 10 --output trace.json
  python scripts/profile.py --mode generate --n-tokens 50 --output trace.json
  python scripts/profile.py --mode memory --output memory_snapshot.pkl
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader

from transformer.gpt import GPT, CharTokenizer
from transformer.model import create_look_ahead_mask, NoamSchedule


def profile_training(model, train_loader, mask, optimizer, scheduler, criterion, device, 
                     steps=10, warmup=2, output="trace.json", use_amp=False):
    """Profile training steps."""
    
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    mps_scaler = torch.amp.GradScaler('mps') if use_amp and device.type == 'mps' else None
    
    # Profiler schedule
    prof_schedule = schedule(wait=1, warmup=warmup, active=steps, repeat=1)
    
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    
    print(f"Profiling training for {steps} steps (warmup={warmup})...")
    
    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(os.path.dirname(output) or "."),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        
        model.train()
        step = 0
        
        for epoch in range(100):  # Max epochs, will break early
            for x, y in train_loader:
                if step >= warmup + steps:
                    break
                
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                
                if use_amp and (scaler or mps_scaler):
                    ctx = torch.amp.autocast(device.type)
                    with ctx:
                        with record_function("forward"):
                            logits = model(x, mask)
                        with record_function("loss"):
                            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    current_scaler = scaler if device.type == 'cuda' else mps_scaler
                    current_scaler.scale(loss).backward()
                    with record_function("optimizer_step"):
                        current_scaler.step(optimizer)
                        current_scaler.update()
                else:
                    with record_function("forward"):
                        logits = model(x, mask)
                    with record_function("loss"):
                        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    with record_function("backward"):
                        loss.backward()
                    with record_function("optimizer_step"):
                        optimizer.step()
                
                scheduler.step()
                prof.step()
                step += 1
                
                if step >= warmup + steps:
                    break
            
            if step >= warmup + steps:
                break
    
    # Export trace
    prof.export_chrome_trace(output)
    print(f"Trace saved to {output}")
    
    # Print summary
    print("\nTop 20 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    if device.type == 'cuda':
        print("\nTop 20 operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    print("\nMemory summary:")
    print(prof.memory_profile())
    
    return prof


def profile_generation(model, tokenizer, prompt, n_tokens=50, temperature=0.8, top_p=0.9,
                       output="trace_gen.json", device="cpu", use_amp=False):
    """Profile generation/inference."""
    
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    
    print(f"Profiling generation of {n_tokens} tokens...")
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        
        model.eval()
        with record_function("generation"):
            idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            
            with torch.no_grad():
                if use_amp and device.type in ('cuda', 'mps'):
                    ctx = torch.amp.autocast(device.type)
                    with ctx:
                        _ = model.generate_cached(idx, n_tokens, temperature, top_p)
                else:
                    _ = model.generate_cached(idx, n_tokens, temperature, top_p)
    
    # Export trace
    prof.export_chrome_trace(output)
    print(f"Trace saved to {output}")
    
    # Print summary
    print("\nTop 20 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    if device.type == 'cuda':
        print("\nTop 20 operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    return prof


def profile_memory(model, train_loader, mask, optimizer, criterion, device,
                   steps=5, output="memory_snapshot.pkl"):
    """Profile memory usage."""
    
    print(f"Profiling memory for {steps} training steps...")
    
    torch.cuda.memory._record_memory_history(max_entries=100000)
    
    model.train()
    step = 0
    
    for epoch in range(100):
        for x, y in train_loader:
            if step >= steps:
                break
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x, mask)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            step += 1
        
        if step >= steps:
            break
    
    # Save memory snapshot
    torch.cuda.memory._dump_snapshot(output)
    torch.cuda.memory._record_memory_history(enabled=False)
    
    print(f"Memory snapshot saved to {output}")
    
    # Print memory stats
    print(f"\nAllocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    print(f"Peak: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")


def analyze_trace(trace_file):
    """Analyze a Chrome trace file."""
    with open(trace_file, "r") as f:
        trace = json.load(f)
    
    events = trace.get("traceEvents", [])
    
    # Aggregate by name
    op_times = {}
    op_counts = {}
    op_flops = {}
    
    for event in events:
        if event.get("ph") == "X":  # Complete event
            name = event.get("name", "unknown")
            dur = event.get("dur", 0) / 1000  # Convert to ms
            flops = event.get("args", {}).get("flops", 0)
            
            if name not in op_times:
                op_times[name] = 0
                op_counts[name] = 0
                op_flops[name] = 0
            
            op_times[name] += dur
            op_counts[name] += 1
            op_flops[name] += flops
    
    # Sort by total time
    sorted_ops = sorted(op_times.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nOperation analysis from {trace_file}:")
    print(f"{'Operation':<50} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'FLOPs':>15}")
    print("-" * 95)
    
    for name, total_time in sorted_ops[:30]:
        count = op_counts[name]
        avg_time = total_time / count if count > 0 else 0
        flops = op_flops.get(name, 0)
        print(f"{name:<50} {count:>8} {total_time:>12.2f} {avg_time:>10.2f} {flops:>15}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "generate", "memory", "analyze"], 
                        default="train", help="Profiling mode")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to profile")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup steps")
    parser.add_argument("--n-tokens", type=int, default=50, help="Tokens to generate")
    parser.add_argument("--prompt", type=str, default="To be, or not to be", help="Generation prompt")
    parser.add_argument("--output", type=str, default="trace.json", help="Output file")
    parser.add_argument("--amp", action="store_true", help="Use AMP")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-file", type=str, help="Trace file to analyze (for analyze mode)")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == "analyze":
        if not args.trace_file:
            print("Please provide --trace-file for analyze mode")
            return
        analyze_trace(args.trace_file)
        return
    
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, _ = data[:-n_val], data[-n_val:]
    
    # Model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        max_len=args.block_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        rope=args.rope,
    ).to(device)
    
    print(f"Model parameters: {model.count_params():,}")
    
    if args.mode == "train":
        from transformer.gpt import CharDataset
        train_loader = DataLoader(
            CharDataset(train_data, args.block_size, 20000), 
            batch_size=args.batch_size, 
            shuffle=True
        )
        mask = create_look_ahead_mask(args.block_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = NoamSchedule(optimizer, d_model=args.d_model, warmup_steps=500)
        criterion = nn.CrossEntropyLoss()
        
        profile_training(model, train_loader, mask, optimizer, scheduler, criterion, device,
                         steps=args.steps, warmup=args.warmup, output=args.output, use_amp=args.amp)
    
    elif args.mode == "generate":
        profile_generation(model, tokenizer, args.prompt, args.n_tokens, 
                          output=args.output, device=device, use_amp=args.amp)
    
    elif args.mode == "memory":
        if device.type != 'cuda':
            print("Memory profiling requires CUDA")
            return
        from transformer.gpt import CharDataset
        train_loader = DataLoader(
            CharDataset(train_data, args.block_size, 20000), 
            batch_size=args.batch_size, 
            shuffle=True
        )
        mask = create_look_ahead_mask(args.block_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95), weight_decay=0.1)
        criterion = nn.CrossEntropyLoss()
        
        profile_memory(model, train_loader, mask, optimizer, criterion, device,
                       steps=args.steps, output=args.output)


if __name__ == "__main__":
    main()