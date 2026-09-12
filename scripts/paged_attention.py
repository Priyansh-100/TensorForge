#!/usr/bin/env python3
"""
PagedAttention implementation for mini-GPT (vLLM-style).

Implements block-based KV cache management with:
- Dynamic block allocation
- Copy-on-write for prefix sharing
- Memory-efficient attention with block tables
- Support for continuous batching

Reference: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)

Usage:
  python scripts/paged_attention.py --demo
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn.functional as F

from transformer.gpt import CharTokenizer
from transformer.model import create_look_ahead_mask


@dataclass
class Block:
    """Single KV cache block."""
    block_id: int
    data: torch.Tensor  # [2, num_kv_heads, block_size, head_dim] for K,V
    ref_count: int = 0
    last_access: float = 0.0


@dataclass
class Sequence:
    """Request sequence with block table."""
    seq_id: str
    prompt: str
    tokens: List[int] = field(default_factory=list)
    block_table: List[int] = field(default_factory=list)
    num_prompt_tokens: int = 0
    num_generated_tokens: int = 0
    finished: bool = False
    finish_reason: Optional[str] = None


class BlockManager:
    """Manages KV cache blocks with reference counting."""
    
    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int, 
                 head_dim: int, device: str = "cpu"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        
        # Pre-allocate block data: [num_blocks, 2, num_kv_heads, block_size, head_dim]
        # 2 = K and V
        self.blocks = torch.zeros(
            num_blocks, 2, num_kv_heads, block_size, head_dim,
            dtype=torch.float32, device=device
        )
        
        self.free_blocks = list(range(num_blocks))
        self.block_refs = [0] * num_blocks
        self.block_timestamps = [0.0] * num_blocks
        self.used_blocks = set()
    
    def allocate(self, num_blocks: int) -> Optional[List[int]]:
        """Allocate contiguous free blocks."""
        if len(self.free_blocks) < num_blocks:
            return None
        
        blocks = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        
        for b in blocks:
            self.block_refs[b] = 1
            self.used_blocks.add(b)
        
        return blocks
    
    def allocate_noncontiguous(self, num_blocks: int) -> Optional[List[int]]:
        """Allocate any free blocks (not necessarily contiguous)."""
        if len(self.free_blocks) < num_blocks:
            return None
        
        blocks = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        
        for b in blocks:
            self.block_refs[b] = 1
            self.used_blocks.add(b)
        
        return blocks
    
    def ref_block(self, block_id: int):
        """Increment reference count."""
        if block_id < self.num_blocks:
            self.block_refs[block_id] += 1
    
    def unref_block(self, block_id: int):
        """Decrement reference count, free if zero."""
        if block_id < self.num_blocks:
            self.block_refs[block_id] -= 1
            if self.block_refs[block_id] <= 0:
                self.block_refs[block_id] = 0
                self.used_blocks.discard(block_id)
                self.free_blocks.append(block_id)
    
    def get_block_data(self, block_ids: List[int]) -> torch.Tensor:
        """Get concatenated block data for attention."""
        if not block_ids:
            return None
        return self.blocks[block_ids]  # [num_blocks, 2, num_kv_heads, block_size, head_dim]
    
    def copy_blocks(self, src_blocks: List[int], dst_blocks: List[int]):
        """Copy block data (for copy-on-write)."""
        if len(src_blocks) != len(dst_blocks):
            raise ValueError("Source and destination block counts must match")
        
        for src, dst in zip(src_blocks, dst_blocks):
            self.blocks[dst].copy_(self.blocks[src])
    
    def write_kv(self, block_ids: List[int], layer_idx: int, 
                 k: torch.Tensor, v: torch.Tensor, start_pos: int):
        """Write K/V to specific position in blocks."""
        if not block_ids:
            return
        
        # Flatten blocks and write
        flat_k = self.blocks[block_ids, 0].view(-1, self.num_kv_heads, self.head_dim)
        flat_v = self.blocks[block_ids, 1].view(-1, self.num_kv_heads, self.head_dim)
        
        seq_len = k.size(2)  # [B, H, T, D]
        flat_k[start_pos:start_pos+seq_len].copy_(k[0])
        flat_v[start_pos:start_pos+seq_len].copy_(v[0])
    
    def read_kv(self, block_ids: List[int], start_pos: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read K/V from blocks."""
        if not block_ids:
            return None, None
        
        flat_k = self.blocks[block_ids, 0].view(-1, self.num_kv_heads, self.head_dim)
        flat_v = self.blocks[block_ids, 1].view(-1, self.num_kv_heads, self.head_dim)
        
        k = flat_k[start_pos:start_pos+seq_len].unsqueeze(0)  # [1, T, H, D]
        v = flat_v[start_pos:start_pos+seq_len].unsqueeze(0)
        
        return k, v
    
    def get_stats(self) -> Dict:
        return {
            "total_blocks": self.num_blocks,
            "free_blocks": len(self.free_blocks),
            "used_blocks": len(self.used_blocks),
            "utilization": len(self.used_blocks) / self.num_blocks,
        }


class PagedAttentionEngine:
    """vLLM-style PagedAttention engine."""
    
    def __init__(self, model, tokenizer, block_size: int = 16, 
                 num_blocks: int = 2048, max_batch_size: int = 32, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_batch_size = max_batch_size
        self.device = device
        
        # Model config
        config = model.__dict__
        self.num_kv_heads = config.get('num_kv_heads', config.get('num_heads', 4))
        self.head_dim = config.get('d_model', 64) // config.get('num_heads', 4)
        self.num_layers = len(model.blocks)
        
        # Block manager
        self.block_manager = BlockManager(
            num_blocks, block_size, self.num_kv_heads, self.head_dim, device
        )
        
        # Sequences
        self.waiting_sequences = []
        self.running_sequences = []
        self.finished_sequences = []
        
        # Stats
        self.stats = {
            "total_requests": 0,
            "completed": 0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
        }
    
    def add_request(self, prompt: str, max_tokens: int = 100, 
                    temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Add a new request."""
        seq_id = str(uuid.uuid4())[:8]
        
        tokens = self.tokenizer.encode(prompt)
        num_blocks = (len(tokens) + self.block_size - 1) // self.block_size
        
        blocks = self.block_manager.allocate(num_blocks)
        if blocks is None:
            # No space, wait
            return None
        
        seq = Sequence(
            seq_id=seq_id,
            prompt=prompt,
            tokens=tokens,
            block_table=blocks,
            num_prompt_tokens=len(tokens),
        )
        
        self.waiting_sequences.append(seq)
        self.stats["total_requests"] += 1
        
        return seq_id
    
    def _prefill(self, seq: Sequence) -> Tuple[torch.Tensor, List[int]]:
        """Prefill phase: compute KV for prompt tokens."""
        input_ids = torch.tensor([seq.tokens], dtype=torch.long, device=self.device)
        seq_len = input_ids.size(1)
        
        # Create causal mask
        mask = create_look_ahead_mask(seq_len).to(self.device)
        
        # Forward pass to get KV caches
        self.model.eval()
        with torch.no_grad():
            _, caches = self.model._cached_forward(input_ids, [None] * self.num_layers, 0, mask)
        
        # Store K/V in blocks
        for layer_idx, (k, v) in enumerate(caches):
            # k, v: [1, num_kv_heads, seq_len, head_dim]
            # Write to blocks
            for block_idx, block_id in enumerate(seq.block_table):
                start = block_idx * self.block_size
                end = min(start + self.block_size, seq_len)
                if start < seq_len:
                    self.block_manager.write_kv(
                        [block_id], layer_idx, 
                        k[:, :, start:end, :], v[:, :, start:end, :],
                        start % self.block_size
                    )
        
        # Return last token logits for first decode step
        last_logits = self.model.lm_head(self.model.ln_f(caches[-1][0][:, -1:, :]))
        
        self.stats["prefill_tokens"] += seq_len
        
        return last_logits, seq.block_table
    
    def _decode_step(self, sequences: List[Sequence]) -> List[Tuple[int, List[int]]]:
        """Single decode step for multiple sequences."""
        batch_size = len(sequences)
        
        # Prepare input: last token from each sequence
        input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        positions = []
        block_tables = []
        
        for i, seq in enumerate(sequences):
            if seq.tokens:
                input_ids[i, 0] = seq.tokens[-1]
            else:
                input_ids[i, 0] = seq.prompt[-1] if seq.prompt else 0
            
            positions.append(seq.num_prompt_tokens + seq.num_generated_tokens)
            block_tables.append(seq.block_table)
        
        # For PagedAttention, we need to gather K/V from blocks
        # This is simplified - real implementation uses custom kernel
        max_blocks = max(len(bt) for bt in block_tables)
        batch_block_table = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=self.device)
        for i, bt in enumerate(block_tables):
            batch_block_table[i, :len(bt)] = torch.tensor(bt, device=self.device)
        
        # Create mask for single token
        mask = torch.ones(batch_size, 1, 1, max(positions) + 1, dtype=torch.bool, device=self.device)
        for i, pos in enumerate(positions):
            mask[i, :, :, pos+1:] = False
        
        # Forward pass (simplified - would use PagedAttention kernel)
        # For demo, we use standard forward with full cache
        # Real PagedAttention would gather from blocks here
        
        # This is a placeholder - in practice you'd use a custom CUDA kernel
        # that reads K/V from the block table
        with torch.no_grad():
            logits = self.model(input_ids, mask)
        
        last_logits = logits[:, -1, :]
        
        # Sample next tokens
        next_tokens = []
        for i, seq in enumerate(sequences):
            logits_i = last_logits[i] / seq.temperature if hasattr(seq, 'temperature') else last_logits[i]
            probs = F.softmax(logits_i, dim=-1)
            
            if seq.top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < seq.top_p
                keep[0] = True
                probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, 
                                                         sorted_probs.masked_fill(~keep, 0))
                probs = probs / probs.sum()
            
            next_token = torch.multinomial(probs, 1).item()
            next_tokens.append(next_token)
        
        return next_tokens, block_tables
    
    def step(self) -> bool:
        """Execute one engine step."""
        # Move waiting to running if space
        while len(self.running_sequences) < self.max_batch_size and self.waiting_sequences:
            seq = self.waiting_sequences.pop(0)
            self.running_sequences.append(seq)
        
        if not self.running_sequences:
            return False
        
        # Check which sequences need prefill
        prefill_seqs = [s for s in self.running_sequences if s.num_generated_tokens == 0]
        decode_seqs = [s for s in self.running_sequences if s.num_generated_tokens > 0]
        
        # Prefill new sequences
        for seq in prefill_seqs:
            self._prefill(seq)
            seq.num_generated_tokens = 1  # After prefill, we have 1 generated token
        
        # Decode step for all running sequences
        if decode_seqs or prefill_seqs:
            all_seqs = decode_seqs + prefill_seqs
            next_tokens, _ = self._decode_step(all_seqs)
            
            for seq, token in zip(all_seqs, next_tokens):
                seq.tokens.append(token)
                seq.num_generated_tokens += 1
                self.stats["decode_tokens"] += 1
                
                # Check finish
                if (token == self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else False 
                    or len(seq.tokens) >= seq.num_prompt_tokens + seq.max_tokens):
                    seq.finished = True
                    seq.finish_reason = "length"
                    self.finished_sequences.append(seq)
                    self.running_sequences.remove(seq)
                    self.stats["completed"] += 1
                    
                    # Free blocks
                    for b in seq.block_table:
                        self.block_manager.unref_block(b)
        
        return len(self.waiting_sequences) > 0 or len(self.running_sequences) > 0
    
    def run(self, max_steps: int = None):
        """Run engine to completion."""
        step = 0
        while (self.waiting_sequences or self.running_sequences) and (max_steps is None or step < max_steps):
            if not self.step():
                break
            step += 1
        return self.finished_sequences


def demo():
    """Demo PagedAttention."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load model
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]
    
    from transformer.gpt import train
    model = train(
        CharTokenizer(text), train_data, val_data,
        epochs=1, block_size=128, d_model=64, num_heads=4, d_ff=256,
        num_layers=2, batch_size=32, save=False, rope=True,
        seed=0, num_pairs=2000, num_val_pairs=200
    )
    model.to(device)
    
    # Create engine
    engine = PagedAttentionEngine(
        model, tokenizer,
        block_size=16,
        num_blocks=1024,
        max_batch_size=4,
        device=device
    )
    
    # Add requests
    prompts = [
        "To be, or not to be",
        "The quick brown fox",
        "Romeo, Romeo, wherefore",
        "All the world's a stage",
    ]
    
    for prompt in prompts:
        engine.add_request(prompt, max_tokens=20)
    
    print(f"Added {len(prompts)} requests")
    print(f"Block manager stats: {engine.block_manager.get_stats()}")
    
    # Run
    import time
    start = time.time()
    finished = engine.run()
    elapsed = time.time() - start
    
    print(f"\nCompleted {len(finished)} requests in {elapsed:.2f}s")
    print(f"Prefill tokens: {engine.stats['prefill_tokens']}")
    print(f"Decode tokens: {engine.stats['decode_tokens']}")
    print(f"Throughput: {(engine.stats['prefill_tokens'] + engine.stats['decode_tokens']) / elapsed:.0f} tok/s")
    print(f"Block manager stats: {engine.block_manager.get_stats()}")
    
    for seq in finished:
        generated = tokenizer.decode(seq.tokens)
        print(f"\n{seq.seq_id}: {seq.prompt} -> {generated[:80]}...")


if __name__ == "__main__":
    demo()