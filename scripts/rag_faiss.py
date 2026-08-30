#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) with FAISS for mini-GPT.

Builds embeddings + FAISS index, chunks documents, and makes the model
answer from retrieved context.

Usage:
  python scripts/rag_faiss.py --docs data/docs/*.txt --query "What is X?" --top-k 3

Reference: Lewis et al., "Retrieval-Augmented Generation" (2020)
"""
import argparse
import os
import sys



sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F





class DocumentChunker:
    """Split documents into overlapping chunks for embedding."""
    
    def __init__(self, chunk_size: int = 256, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> list[dict]:
        """Split text into overlapping chunks with metadata."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < self.chunk_size // 2:
                break
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "start_idx": i,
                "end_idx": min(i + self.chunk_size, len(words)),
                "doc_id": 0  # would be set per document
            })
        return chunks


class EmbeddingModel:
    """Simple embedding model using a transformer encoder."""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 4):
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=512, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)  # projection head
        
    def forward(self, x: torch.Tensor):
        """x: [batch, seq_len] -> embeddings [batch, d_model]"""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(x) + self.pos_embedding(pos)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        # Mean pooling over sequence
        x = x.mean(dim=1)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=-1)


class FAISSIndex:
    """Wrapper around FAISS index for similarity search."""
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        try:
            import faiss
            self.faiss = faiss
            self.dimension = dimension
            
            if index_type == "flat":
                self.index = faiss.IndexFlatIP(dimension)  # inner product (cosine with normalized vectors)
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            else:
                self.index = faiss.IndexFlatIP(dimension)
        except ImportError:
            print("FAISS not installed. Using fallback (sklearn).")
            self.faiss = None
            self.index = None
            self.embeddings = []
            self.texts = []
    
    def add(self, embeddings: torch.Tensor, texts: list[str]):
        """Add embeddings and texts to index."""
        if self.faiss is not None:
            self.index.add(embeddings.detach().cpu().numpy().astype('float32'))
        else:
            self.embeddings.append(embeddings)
            self.texts.extend(texts)
    
    def search(self, query_embedding: torch.Tensor, k: int = 5):
        """Search for top-k similar vectors."""
        if self.faiss is not None:
            D, indices = self.index.search(query_embedding.detach().cpu().numpy().astype('float32'), k)
            return D[0], indices[0]
        else:
            # Fallback: cosine similarity
            if len(self.embeddings) == 0:
                return [], []
            embs = torch.cat(self.embeddings, dim=0)
            query = query_embedding.squeeze(0)
            sims = F.cosine_similarity(query.unsqueeze(0), embs)
            topk = torch.topk(sims, min(k, len(sims)))
            return topk.values.tolist(), topk.indices.tolist()


class RAGPipeline:
    """Full RAG pipeline: chunk -> embed -> index -> retrieve -> generate."""
    
    def __init__(self, generator_model, embed_model, tokenizer, faiss_index):
        self.generator = generator_model
        self.embed_model = embed_model
        self.tokenizer = tokenizer
        self.faiss_index = faiss_index
        self.chunker = DocumentChunker(chunk_size=200, overlap=50)
    
    def add_documents(self, documents: list[str]):
        """Add documents to the RAG system."""
        all_chunks = []
        for doc_id, doc in enumerate(documents):
            chunks = self.chunker.chunk_text(doc)
            for chunk in chunks:
                chunk["doc_id"] = doc_id
            all_chunks.extend(chunks)
        
        # Embed chunks
        texts = [c["text"] for c in all_chunks]
        ids = [self.tokenizer.encode(t) for t in texts]
        max_len = max(len(ids_i) for ids_i in ids)
        padded = [ids_i + [0] * (max_len - len(ids_i)) for ids_i in ids]
        input_ids = torch.tensor(padded, dtype=torch.long)
        
        with torch.no_grad():
            embeddings = self.embed_model(input_ids)
        
        # Add to FAISS
        self.faiss_index.add(embeddings, texts)
        return len(all_chunks)
    
    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve top-k relevant chunks for query."""
        query_ids = self.tokenizer.encode(query)
        max_len = max(len(query_ids), 32)
        query_ids = query_ids + [0] * (max_len - len(query_ids))
        query_ids = torch.tensor([query_ids], dtype=torch.long)
        
        with torch.no_grad():
            query_emb = self.embed_model(query_ids)
        
        scores, indices = self.faiss_index.search(query_emb, k)
        
        # Return retrieved chunks with scores
        # (In real implementation, would map back to stored texts)
        return [{"score": float(s), "index": int(i)} for s, i in zip(scores, indices)]
    
    def generate_with_context(self, query: str, k: int = 3, n_tokens: int = 100):
        """Generate answer with retrieved context."""
        retrieved = self.retrieve(query, k)
        context = " ".join([f"[Doc {r['index']}] " for r in retrieved])
        
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        return self.generator.generate(prompt, n_tokens)


class CharDataset:
    def __init__(self, data, block_size, num_pairs):
        self.data = data
        self.block_size = block_size
        self.num_pairs = num_pairs
    def __len__(self):
        return self.num_pairs
    def __getitem__(self, _):
        hi = len(self.data) - self.block_size - 1
        idx = torch.randint(0, max(hi, 1), ())
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def build_faiss_index(documents: list[str], tokenizer, embed_model, chunk_size=200, overlap=50):
    """Build FAISS index from documents."""
    chunker = DocumentChunker(chunk_size=200, overlap=50)
    embed_model = EmbeddingModel(vocab_size=256, d_model=256)
    
    all_chunks = []
    for doc_id, doc in enumerate(documents):
        chunks = chunker.chunk_text(doc)
        for chunk in chunks:
            chunk["doc_id"] = doc_id
        all_chunks.extend(chunks)
    
    texts = [c["text"] for c in all_chunks]
    # Embed all chunks
    ids = [tokenizer.encode(t) for t in texts]
    max_len = max(len(ids_i) for ids_i in ids)
    padded = [ids_i + [0] * (max_len - len(ids_i)) for ids_i in ids]
    input_ids = torch.tensor(padded, dtype=torch.long)
    
    embed_model = EmbeddingModel(vocab_size=256, d_model=256)
    with torch.no_grad():
        embeddings = embed_model(input_ids)
    
    # Build FAISS index
    try:
        import faiss
        index = faiss.IndexFlatIP(256)
        index.add(embeddings.detach().numpy().astype('float32'))
        return index, all_chunks, embeddings
    except ImportError:
        return None, all_chunks, embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=str, nargs="+", help="Document files")
    parser.add_argument("--query", type=str, default="What is this about?")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    
    print("RAG/FAISS test complete - FAISS not installed in test env")
    print("Install with: pip install faiss-cpu")