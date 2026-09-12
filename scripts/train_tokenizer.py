#!/usr/bin/env python3
"""
BPE Tokenizer training pipeline for mini-GPT.

Supports:
- Training on large text corpora
- Byte-level BPE (GPT-2 style) or character-level
- Special token handling (pad, bos, eos, unk)
- Vocabulary size configuration
- Serialization/deserialization
- Tokenizer comparison and evaluation

Usage:
  python scripts/train_tokenizer.py --corpus data/corpus.txt --vocab-size 50000 --output tokenizer.json
"""

import argparse
import os
import sys
import json
import re
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from transformer.bpe import BPETokenizer


class TokenizerTrainer:
    """Advanced BPE tokenizer training with preprocessing."""
    
    def __init__(self, vocab_size: int = 50000, min_frequency: int = 2, 
                 special_tokens: Optional[Dict[str, str]] = None):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or {
            "[PAD]": "<pad>",
            "[BOS]": "<bos>",
            "[EOS]": "<eos>",
            "[UNK]": "<unk>",
        }
        self.tokenizer = None
    
    def preprocess_text(self, text: str, lowercase: bool = False, 
                       remove_urls: bool = True, remove_emails: bool = True) -> str:
        """Preprocess text corpus."""
        if lowercase:
            text = text.lower()
        
        if remove_urls:
            text = re.sub(r'http[s]?://\S+', '', text)
        
        if remove_emails:
            text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def train_from_files(self, file_paths: List[str], 
                        lowercase: bool = False,
                        chunk_size: int = 1000000) -> BPETokenizer:
        """Train tokenizer from multiple files."""
        print(f"Training BPE tokenizer from {len(file_paths)} files...")
        
        # Read and preprocess all files
        full_text = ""
        for path in file_paths:
            print(f"  Reading {path}...")
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                if chunk_size:
                    # Process in chunks for large files
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        full_text += self.preprocess_text(chunk, lowercase)
                else:
                    full_text += self.preprocess_text(f.read(), lowercase)
        
        print(f"  Total corpus size: {len(full_text):,} characters")
        
        # Train tokenizer (BPETokenizer trains during __init__)
        self.tokenizer = BPETokenizer(full_text, vocab_size=self.vocab_size)
        
        print(f"  Trained tokenizer: {self.tokenizer.vocab_size} tokens")
        print(f"  Compression ratio: {len(full_text) / len(self.tokenizer.encode(full_text)):.2f}x")
        
        return self.tokenizer
    
    def train_from_iterator(self, text_iterator, total_chars: int = None) -> BPETokenizer:
        """Train from text iterator (for streaming large corpora)."""
        print("Training from iterator...")
        
        # Collect text (in practice, would use incremental training)
        full_text = ""
        count = 0
        for text in text_iterator:
            full_text += text
            count += len(text)
            if total_chars and count >= total_chars:
                break
        
        self.tokenizer = BPETokenizer(full_text, vocab_size=self.vocab_size)
        
        return self.tokenizer
    
    def evaluate_tokenizer(self, test_text: str) -> Dict:
        """Evaluate tokenizer quality."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not trained yet")
        
        tokens = self.tokenizer.encode(test_text)
        decoded = self.tokenizer.decode(tokens)
        
        # Round-trip accuracy
        round_trip = test_text == decoded
        
        # Statistics
        char_count = len(test_text)
        token_count = len(tokens)
        compression = char_count / token_count if token_count > 0 else 0
        
        # Vocabulary coverage
        unique_tokens = len(set(tokens))
        vocab_coverage = unique_tokens / self.tokenizer.vocab_size
        
        # Token length distribution
        token_lengths = [len(self.tokenizer.decode([t])) for t in tokens]
        avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        
        return {
            "round_trip": round_trip,
            "char_count": char_count,
            "token_count": token_count,
            "compression_ratio": compression,
            "unique_tokens": unique_tokens,
            "vocab_coverage": vocab_coverage,
            "avg_token_length": avg_token_length,
        }
    
    def save(self, path: str):
        """Save tokenizer to file."""
        if not self.tokenizer:
            raise ValueError("No tokenizer to save")
        
        # Convert tuple keys to strings for JSON serialization
        merges_serializable = {f"{k[0]} {k[1]}": v for k, v in self.tokenizer.merges.items()}
        
        # Convert bytes in vocab to strings
        vocab_serializable = {k: v.decode('utf-8', errors='replace') if isinstance(v, bytes) else v 
                              for k, v in self.tokenizer.vocab.items()}
        
        data = {
            "vocab_size": self.vocab_size,
            "merges": merges_serializable,
            "vocab": vocab_serializable,
            "special_tokens": self.special_tokens,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Tokenizer saved to {path}")
    
    def load(self, path: str) -> BPETokenizer:
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert string keys back to tuples
        merges = {tuple(k.split()): v for k, v in data["merges"].items()}
        
        self.tokenizer = BPETokenizer("", vocab_size=data["vocab_size"])
        self.tokenizer.merges = merges
        self.tokenizer.vocab = data["vocab"]
        self.special_tokens = data.get("special_tokens", self.special_tokens)
        
        print(f"Tokenizer loaded from {path}: {self.tokenizer.vocab_size} tokens")
        return self.tokenizer


def compare_tokenizers(tokenizers: Dict[str, BPETokenizer], test_texts: List[str]) -> Dict:
    """Compare multiple tokenizers on test texts."""
    results = {}
    
    for name, tokenizer in tokenizers.items():
        print(f"\nEvaluating {name}...")
        total_chars = 0
        total_tokens = 0
        total_unique = set()
        round_trip_ok = 0
        
        for text in test_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            
            total_chars += len(text)
            total_tokens += len(tokens)
            total_unique.update(tokens)
            
            if text == decoded:
                round_trip_ok += 1
        
        results[name] = {
            "avg_compression": total_chars / total_tokens if total_tokens > 0 else 0,
            "round_trip_accuracy": round_trip_ok / len(test_texts),
            "vocab_usage": len(total_unique) / tokenizer.vocab_size,
            "total_tokens": total_tokens,
        }
        
        print(f"  Compression: {results[name]['avg_compression']:.2f}x")
        print(f"  Round-trip: {results[name]['round_trip_accuracy']:.2%}")
        print(f"  Vocab usage: {results[name]['vocab_usage']:.2%}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, nargs="+", required=True, help="Corpus file(s)")
    parser.add_argument("--vocab-size", type=int, default=50000, help="Vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum merge frequency")
    parser.add_argument("--output", type=str, required=True, help="Output tokenizer file")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--test-files", type=str, nargs="+", help="Test files for evaluation")
    parser.add_argument("--compare", type=str, nargs="+", help="Compare with existing tokenizers")
    args = parser.parse_args()
    
    # Train tokenizer
    trainer = TokenizerTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    
    tokenizer = trainer.train_from_files(args.corpus, lowercase=args.lowercase)
    
    # Save
    trainer.save(args.output)
    
    # Evaluation
    if args.eval or args.test_files:
        test_texts = []
        if args.test_files:
            for path in args.test_files:
                with open(path, 'r', encoding='utf-8') as f:
                    test_texts.append(f.read()[:10000])  # First 10k chars
        else:
            # Use last portion of training data
            with open(args.corpus[0], 'r', encoding='utf-8') as f:
                full = f.read()
                test_texts.append(full[-10000:])
        
        print("\nEvaluation:")
        eval_results = trainer.evaluate_tokenizer(test_texts[0])
        for k, v in eval_results.items():
            print(f"  {k}: {v}")
    
    # Comparison
    if args.compare:
        tokenizers = {"trained": tokenizer}
        for path in args.compare:
            name = os.path.basename(path).replace(".json", "")
            t = BPETokenizer(vocab_size=50000)
            t.load(path)
            tokenizers[name] = t
        
        print("\nComparison:")
        compare_tokenizers(tokenizers, test_texts)


if __name__ == "__main__":
    main()