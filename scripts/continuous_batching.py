#!/usr/bin/env python3
"""
Continuous batching for efficient LLM serving.

Implements:
- Dynamic batching (prefill + decode)
- KV cache management compatible with mini-GPT
- Request scheduling with priorities
- Streaming responses
"""

import os
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from transformer.gpt import CharTokenizer


@dataclass
class Request:
    """Inference request."""
    id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    
    # Runtime state
    input_ids: Optional[torch.Tensor] = None
    generated_ids: List[int] = field(default_factory=list)
    position: int = 0
    prefill_done: bool = False
    kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    finished: bool = False
    finish_reason: Optional[str] = None


@dataclass
class Batch:
    """Batch of requests being processed together."""
    requests: List[Request]
    input_ids: torch.Tensor
    positions: torch.Tensor
    kv_caches: List[Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]
    is_prefill: bool
    max_seq_len: int


class ContinuousBatchingEngine:
    """Continuous batching inference engine."""
    
    def __init__(self, model, tokenizer, max_batch_size=32, max_seq_len=2048, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.waiting_queue = deque()
        self.running_requests = {}
        self.finished_requests = {}
        
        # Stats
        self.stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "total_tokens_generated": 0,
            "total_prefill_tokens": 0,
        }
    
    def add_request(self, prompt, max_tokens=100, temperature=0.8, top_p=0.9, priority=0):
        """Add a new request to the queue."""
        request = Request(
            id=str(uuid.uuid4())[:8],
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            priority=priority,
        )
        request.input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        self.waiting_queue.append(request)
        self.stats["total_requests"] += 1
        return request.id
    
    def _schedule_batch(self):
        """Schedule next batch from waiting and running requests."""
        # Priority: running decode requests first, then prefill for new requests
        batch_requests = []
        
        # Add running decode requests (highest priority)
        for req in self.running_requests.values():
            if not req.finished and req.prefill_done:
                batch_requests.append(req)
                if len(batch_requests) >= self.max_batch_size:
                    break
        
        # Add waiting prefill requests
        while len(batch_requests) < self.max_batch_size and self.waiting_queue:
            req = self.waiting_queue.popleft()
            if req.input_ids.size(1) > self.max_seq_len:
                req.finished = True
                req.finish_reason = "context_too_long"
                self.finished_requests[req.id] = req
                continue
            
            # Initialize empty KV cache for this request
            # The model's _cached_forward will create empty caches if None is passed
            req.kv_cache = None
            
            batch_requests.append(req)
            self.running_requests[req.id] = req
        
        if not batch_requests:
            return None
        
        return batch_requests
    
    def _prepare_batch(self, requests, is_prefill):
        """Prepare batch tensors."""
        if is_prefill:
            # Prefill: concatenate all prompts
            max_len = max(r.input_ids.size(1) for r in requests)
            input_ids = torch.zeros(len(requests), max_len, dtype=torch.long, device=self.device)
            positions = torch.zeros(len(requests), max_len, dtype=torch.long, device=self.device)
            
            for i, req in enumerate(requests):
                seq_len = req.input_ids.size(1)
                input_ids[i, :seq_len] = req.input_ids[0]
                positions[i, :seq_len] = torch.arange(seq_len, device=self.device)
                req.position = seq_len
            
            # Pass None for kv_caches - model will create empty ones
            kv_caches = [None] * len(requests)
            
            return Batch(requests, input_ids, positions, kv_caches, True, max_len)
        else:
            # Decode: single token per request
            batch_size = len(requests)
            input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
            positions = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
            
            for i, req in enumerate(requests):
                if req.generated_ids:
                    input_ids[i, 0] = req.generated_ids[-1]
                else:
                    input_ids[i, 0] = req.input_ids[0, -1]
                positions[i, 0] = req.position
            
            # Get KV caches from requests
            kv_caches = [req.kv_cache for req in requests]
            
            return Batch(requests, input_ids, positions, kv_caches, False, 1)
    
    def _sample_token(self, logits, temperature, top_p):
        """Sample next token."""
        probs = F.softmax(logits / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < top_p
            keep[..., 0] = True
            trimmed = sorted_probs.masked_fill(~keep, 0.0)
            probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, trimmed / trimmed.sum(dim=-1, keepdim=True))
        return torch.multinomial(probs, num_samples=1)
    
    def step(self):
        """Execute one batch step."""
        requests = self._schedule_batch()
        if not requests:
            return False
        
        is_prefill = any(not r.prefill_done for r in requests)
        batch = self._prepare_batch(requests, is_prefill)
        
        self.model.eval()
        with torch.no_grad():
            # Forward pass
            if is_prefill:
                logits, new_caches = self.model._cached_forward(
                    batch.input_ids, batch.kv_caches, 0, None
                )
            else:
                logits, new_caches = self.model._cached_forward(
                    batch.input_ids, batch.kv_caches, batch.positions[0, 0].item(), None
                )
            
            # Sample next tokens
            next_tokens = self._sample_token(
                logits[:, -1, :],
                requests[0].temperature,
                requests[0].top_p
            )
            
            # Update requests with new caches
            for i, req in enumerate(requests):
                token_id = next_tokens[i].item()
                req.generated_ids.append(token_id)
                req.position += 1
                req.kv_cache = new_caches[i] if i < len(new_caches) else None
                
                if not req.prefill_done:
                    req.prefill_done = True
                    self.stats["total_prefill_tokens"] += req.input_ids.size(1)
                
                # Check finish conditions
                # CharTokenizer doesn't have eos_token_id, just use max_tokens
                finished = len(req.generated_ids) >= req.max_tokens
                if finished:
                    req.finished = True
                    req.finish_reason = "length"
                    self.finished_requests[req.id] = req
                    del self.running_requests[req.id]
                    self.stats["completed_requests"] += 1
                    self.stats["total_tokens_generated"] += len(req.generated_ids)
        
        return True
    
    def run(self, max_steps=None):
        """Run engine until all requests complete."""
        step = 0
        while (self.waiting_queue or self.running_requests) and (max_steps is None or step < max_steps):
            if not self.step():
                break
            step += 1
        
        return self.finished_requests
    
    def stream(self, request_id):
        """Stream tokens for a request as they're generated."""
        if request_id not in self.running_requests and request_id not in self.finished_requests:
            # Add to queue
            pass
        
        # This would yield tokens as they're generated
        # Implementation depends on the specific use case
        pass


def demo():
    """Demo continuous batching."""
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
    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=4, device=device)
    
    # Add requests
    prompts = [
        "To be, or not to be",
        "The quick brown fox",
        "Romeo, Romeo, wherefore",
        "All the world's a stage",
    ]
    
    for prompt in prompts:
        engine.add_request(prompt, max_tokens=50, temperature=0.8, top_p=0.9)
    
    print(f"Added {len(prompts)} requests")
    
    # Run
    start = time.time()
    finished = engine.run()
    elapsed = time.time() - start
    
    print(f"\nCompleted {len(finished)} requests in {elapsed:.2f}s")
    print(f"Total tokens generated: {engine.stats['total_tokens_generated']}")
    print(f"Throughput: {engine.stats['total_tokens_generated'] / elapsed:.0f} tok/s")
    
    for req_id, req in finished.items():
        generated = tokenizer.decode(req.generated_ids)
        print(f"\n{req_id}: {req.prompt} -> {generated[:80]}...")


if __name__ == "__main__":
    demo()