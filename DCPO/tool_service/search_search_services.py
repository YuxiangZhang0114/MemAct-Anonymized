# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import warnings
from typing import Optional
import asyncio
import time
from collections import OrderedDict

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
# from sentence_transformers import SentenceTransformer  # Not used
import threading
import pandas as pd
from asyncio import Semaphore


class TTLCache:
    """
    Time-based cache class with TTL (Time To Live) and LRU eviction policy.
    """
    def __init__(self, maxsize: int = 1024, ttl_seconds: int = 300):
        self.maxsize = int(maxsize)
        self.ttl_seconds = int(ttl_seconds)
        self._store = OrderedDict()
        self._lock = threading.Lock()

    def _is_expired(self, expires_at: float) -> bool:
        return time.time() >= expires_at

    def get(self, key):
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if self._is_expired(expires_at):
                try:
                    del self._store[key]
                except KeyError:
                    pass
                return None
            self._store.move_to_end(key, last=True)
            return value

    def set(self, key, value):
        with self._lock:
            expires_at = time.time() + self.ttl_seconds
            if key in self._store:
                self._store[key] = (expires_at, value)
                self._store.move_to_end(key, last=True)
            else:
                self._store[key] = (expires_at, value)
            while len(self._store) > self.maxsize:
                try:
                    self._store.popitem(last=False)
                except KeyError:
                    break

    def clear(self):
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._store)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            expired_count = 0
            valid_count = 0
            for expires_at, _ in self._store.values():
                if self._is_expired(expires_at):
                    expired_count += 1
                else:
                    valid_count += 1
            return {
                "total_items": len(self._store),
                "valid_items": valid_count,
                "expired_items": expired_count,
                "maxsize": self.maxsize,
                "ttl_seconds": self.ttl_seconds
            }


def load_corpus(corpus_path: str):
    print(f"[INFO] Loading corpus: {corpus_path}")
    # corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    corpus = pd.read_parquet(corpus_path)
    print(f"[INFO] Corpus loaded, containing {len(corpus)} documents")
    return corpus


def load_docs(corpus, doc_idxs):
    results = [corpus.iloc[int(idx)].to_dict() for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False):
    print(f"[INFO] Loading model: {model_path}")
    print(f"[INFO] Using FP16: {use_fp16}")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    print(f"[INFO] Model loaded")
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, st_batch_size: int = 512):
        print(f"[INFO] Initializing encoder (Transformers) - model: {model_path}")
        self.model_name = model_name
        self.model_path = model_path
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.st_batch_size = st_batch_size
        self.pooling_method = pooling_method

        # Load model and tokenizer using transformers
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        
        # Set model to evaluation mode and move to GPU
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"[INFO] Model moved to GPU: {torch.cuda.get_device_name(0)}")
        
        # Use FP16
        if use_fp16 and torch.cuda.is_available():
            self.model = self.model.half()
            print(f"[INFO] Model set to FP16 mode")
        if 'bge' in model_path:
            self.pooling_method = "cls"
        
        self._lock = threading.Lock()
        print(f"[INFO] Encoder initialized, max_length={max_length}, pooling_method={pooling_method}")

    @torch.inference_mode()
    def encode(self, query_list: list[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]
        print(f"[INFO] Encoding {len(query_list)} texts, is_query={is_query}")
        
        # Add instruction for queries
        instruction = "Represent this sentence for searching relevant passages: "
        if is_query:
            texts = [instruction + query for query in query_list]
        else:
            texts = query_list
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.st_batch_size):
            batch_texts = texts[i:i + self.st_batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to GPU
            if torch.cuda.is_available():
                encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
            
            # Forward pass
            model_output = self.model(**encoded_input)
            
            # Pooling operation
            last_hidden_state = model_output.last_hidden_state
            attention_mask = encoded_input['attention_mask']
            
            if self.pooling_method == "cls":
                # CLS pooling
                batch_embeddings = last_hidden_state[:, 0]
            elif self.pooling_method == "mean":
                # Mean pooling
                batch_embeddings = pooling(
                    None, 
                    last_hidden_state, 
                    attention_mask, 
                    pooling_method="mean"
                )
            elif self.pooling_method == "pooler":
                # Pooler output
                batch_embeddings = model_output.pooler_output
            else:
                # Default to CLS
                batch_embeddings = last_hidden_state[:, 0]
            
            # Normalize
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            # Convert to numpy
            batch_embeddings = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batch results
        emb = np.concatenate(all_embeddings, axis=0)
        
        # Ensure float32 C-contiguous
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32, order="C")
        elif not emb.flags["C_CONTIGUOUS"]:
            emb = np.ascontiguousarray(emb, dtype=np.float32)
        
        print(f"[INFO] Encoding completed, output shape: {emb.shape}")
        return emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: list[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


# class BM25Retriever(BaseRetriever):
#     def __init__(self, config):
#         super().__init__(config)
#         print(f"[INFO] Initializing BM25 retriever, index path: {self.index_path}")
#         from pyserini.search.lucene import LuceneSearcher
#
#         self.searcher = LuceneSearcher(self.index_path)
#         self.contain_doc = self._check_contain_doc()
#         if not self.contain_doc:
#             self.corpus = load_corpus(self.corpus_path)
#         self.max_process_num = 8
#         print(f"[INFO] BM25 retriever initialized")
#
#     def _check_contain_doc(self):
#         return self.searcher.doc(0).raw() is not None
#
#     def _search(self, query: str, num: int = None, return_score: bool = False):
#         print(f"[INFO] BM25 search query: '{query[:50]}...', return count: {num}")
#         if num is None:
#             num = self.topk
#         hits = self.searcher.search(query, num)
#         if len(hits) < 1:
#             print(f"[WARNING] No matching documents found")
#             if return_score:
#                 return [], []
#             else:
#                 return []
#         scores = [hit.score for hit in hits]
#         if len(hits) < num:
#             warnings.warn("Not enough documents retrieved!", stacklevel=2)
#         else:
#             hits = hits[:num]
#
#         if self.contain_doc:
#             all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
#             results = [
#                 {
#                     "title": content.split("\n")[0].strip('"'),
#                     "text": "\n".join(content.split("\n")[1:]),
#                     "contents": content,
#                 }
#                 for content in all_contents
#             ]
#         else:
#             results = load_docs(self.corpus, [hit.docid for hit in hits])
#
#         print(f"[INFO] BM25 search completed, returned {len(results)} results")
#         if return_score:
#             return results, scores
#         else:
#             return results
#
#     def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
#         results = []
#         scores = []
#         for query in query_list:
#             item_result, item_score = self._search(query, num, True)
#             results.append(item_result)
#             scores.append(item_score)
#         if return_score:
#             return results, scores
#         else:
#             return results


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        print(f"[INFO] Initializing dense retriever, index path: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            try:
                import os
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")

                # 1) Infer/specify embedding file path (can also put embedding_path in config)
                emb_path = getattr(config, "embedding_path", None)
                if emb_path is None:
                    base, _ = os.path.splitext(self.index_path)
                    emb_path = base + ".npy"
                if not os.path.exists(emb_path):
                    raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

                # 2) Read with memory mapping to avoid loading large amounts of memory at once
                emb = np.load(emb_path, mmap_mode="r")  # shape: (N, d), dtype usually float32
                if emb.ndim != 2:
                    raise ValueError(f"Embeddings must be 2-D, got shape={emb.shape}")
                N, d = emb.shape
                print(f"[INFO] Loading embeddings: N={N}, d={d}, dtype={emb.dtype}, mmap_mode=True")

                # 3) Estimate final GPU usage (FP16 storage)
                est_fp16_gb = N * d * 2 / (1024**3)   # 2 bytes per value
                free, total = torch.cuda.mem_get_info()
                print(f"[INFO] GPU free={free/1e9:.2f} GB / total={total/1e9:.2f} GB, "
                    f"estimated final index (FP16) ~{est_fp16_gb:.2f} GB")

                # Can decide whether to unload/delay loading other large models based on free space

                # 4) Create GPU resources and FP16 GpuIndexFlatIP
                self.gpu_res = faiss.StandardGpuResources()
                # Control temporary workspace (prevent requesting too large contiguous memory when cloning/adding)
                self.gpu_res.setTempMemory(256 * 1024 * 1024)  # 256MB, increase if necessary

                cfg = faiss.GpuIndexFlatConfig()
                cfg.device = 0
                cfg.useFloat16 = True
                gindex = faiss.GpuIndexFlatIP(self.gpu_res, d, cfg)

                # 5) Check if normalization is needed (bge is usually normalized, but check sample for safety)
                sample = np.array(emb[: min(1024, N)], dtype=np.float32, order="C", copy=False)
                sample_norm = np.linalg.norm(sample, axis=1)
                need_norm = not (np.allclose(sample_norm.mean(), 1.0, atol=1e-2))
                print(f"[INFO] normalized embeddings detected? {not need_norm} "
                    f"(mean ||x|| ≈ {sample_norm.mean():.4f})")

                # 6) Calculate safe batch size (conservative), can also use fixed value like 200_000
                #    Since index stores FP16 (2 bytes/val), final new usage ≈ bs * d * 2 bytes
                #    Here we set each round's new usage < 30% of available memory as safety margin
                safety_ratio = 0.30
                bs_by_mem = int((free * safety_ratio) // (d * 2))   # Estimate: calculate by final FP16 usage
                bs = max(50_000, min(200_000, bs_by_mem))          # Limit to [5e4, 2e5] range

                print(f"[INFO] chosen add batch size = {bs} (by mem: {bs_by_mem})")

                # 7) Add in chunks (input must be float32 C-contiguous; normalize in chunks if needed)
                added = 0
                for start in range(0, N, bs):
                    end = min(start + bs, N)
                    chunk = np.array(emb[start:end], dtype=np.float32, order="C", copy=False)
                    if need_norm:
                        norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-12
                        chunk = chunk / norms
                    gindex.add(chunk)
                    added += (end - start)
                    if added % (bs * 5) == 0 or end == N:
                        cur_free, _ = torch.cuda.mem_get_info()
                        print(f"[INFO] added {added}/{N} vectors, GPU free ~{cur_free/1e9:.2f} GB")

                self.index = gindex
                print(f"[INFO] GPU GpuIndexFlatIP(FP16) built: ntotal={self.index.ntotal}")
            except Exception as e:
                print(f"[WARNING] GPU acceleration not available, falling back to CPU: {e}")

        print(f"Loading corpus from {self.corpus_path}")
        self.corpus = load_corpus(self.corpus_path)
        print(f"Corpus loaded: {len(self.corpus)} documents")
        print(f"[INFO] Initializing encoder")
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
            st_batch_size=getattr(config, "retrieval_st_batch_size", 8),
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        # IVF/SQ parameters (if index supports)
        self.nprobe = getattr(config, "retrieval_nprobe", None)
        if self.nprobe is not None and hasattr(self.index, "nprobe"):
            try:
                self.index.nprobe = int(self.nprobe)
                print(f"[INFO] Set nprobe = {self.index.nprobe}")
            except Exception as e:
                print(f"[WARNING] Unable to set nprobe: {e}")
        # Add lock for GPU FAISS retrieval
        self._faiss_lock = threading.Lock()
        
        # Initialize cache
        cache_maxsize = getattr(config, "cache_maxsize", 1024)
        cache_ttl = getattr(config, "cache_ttl_seconds", 300)
        self.cache = TTLCache(maxsize=cache_maxsize, ttl_seconds=cache_ttl)
        print(f"[INFO] Cache initialized, max size: {cache_maxsize}, TTL: {cache_ttl} seconds")
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self._cache_stats_lock = threading.Lock()
        
        print(f"[INFO] Dense retriever initialized")

    def _search(self, query: str, num: int = None, return_score: bool = False):
        print(f"[INFO] Dense retrieval search query: '{query[:50]}...', return count: {num}")
        if num is None:
            num = self.topk
        
        # Create cache key
        cache_key = f"{query}|{num}|{return_score}"
        
        # Try to get result from cache
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            with self._cache_stats_lock:
                self.cache_hits += 1
            print(f"[INFO] Cache hit, returning cached result")
            return cached_result
        
        # Cache miss, perform actual retrieval
        with self._cache_stats_lock:
            self.cache_misses += 1
        
        query_emb = self.encoder.encode(query)
        # Thread lock for retrieval process (especially for GPU index)
        with self._faiss_lock:
            scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        
        # Prepare return result and cache
        if return_score:
            result = (results, scores.tolist())
        else:
            result = results
        
        # Store result in cache
        self.cache.set(cache_key, result)
        
        print(f"[INFO] Dense retrieval search completed, returned {len(results)} results, cached")
        return result

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        uncached_queries = []
        uncached_indices = []

        # First check cache
        for i, query in enumerate(query_list):
            cache_key = f"{query}|{num}|{return_score}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                with self._cache_stats_lock:
                    self.cache_hits += 1
                if return_score:
                    cached_results, cached_scores = cached_result
                    results.append(cached_results)
                    scores.append(cached_scores)
                else:
                    results.append(cached_result)
                    scores.append([])  # Placeholder
            else:
                with self._cache_stats_lock:
                    self.cache_misses += 1
                # Record queries that need actual retrieval
                uncached_queries.append(query)
                uncached_indices.append(i)
                # Pre-allocate positions for uncached queries
                results.append(None)
                scores.append(None)

        print(f"[INFO] Batch search: total queries {len(query_list)}, cache hits {len(query_list) - len(uncached_queries)}, need retrieval {len(uncached_queries)}")

        # If there are uncached queries, perform batch retrieval
        if uncached_queries:
            uncached_results = []
            uncached_scores = []
            
            for start_idx in tqdm(range(0, len(uncached_queries), self.batch_size), desc="Retrieval process: ", disable=True):
                query_batch = uncached_queries[start_idx : start_idx + self.batch_size]
                batch_emb = self.encoder.encode(query_batch)
                with self._faiss_lock:
                    batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
                batch_scores = batch_scores.tolist()
                batch_idxs = batch_idxs.tolist()

                # load_docs is not vectorized, but is a python list approach
                flat_idxs = sum(batch_idxs, [])
                batch_results = load_docs(self.corpus, flat_idxs)
                # chunk them back
                batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

                uncached_results.extend(batch_results)
                uncached_scores.extend(batch_scores)

                del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
                torch.cuda.empty_cache()

            # Put uncached results back to original positions and cache
            for i, (query_idx, query) in enumerate(zip(uncached_indices, uncached_queries)):
                result_data = uncached_results[i]
                score_data = uncached_scores[i]
                
                # Store in results list
                results[query_idx] = result_data
                scores[query_idx] = score_data
                
                # Cache result
                cache_key = f"{query}|{num}|{return_score}"
                if return_score:
                    cache_value = (result_data, score_data)
                else:
                    cache_value = result_data
                self.cache.set(cache_key, cache_value)

        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config):
    print(f"[INFO] Creating retriever, type: {config.retrieval_method}")
    if config.retrieval_method == "bm25":
        raise NotImplementedError("BM25Retriever is currently commented out")
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "path/to/embedding_model",
        retrieval_pooling_method: str = "cls",
        retrieval_query_max_length: int = 512,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 8,
        retrieval_nprobe: Optional[int] = None,
        retrieval_st_batch_size: int = 8,
        cache_maxsize: int = 1024,
        cache_ttl_seconds: int = 300,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.retrieval_nprobe = retrieval_nprobe
        self.retrieval_st_batch_size = retrieval_st_batch_size
        self.cache_maxsize = cache_maxsize
        self.cache_ttl_seconds = cache_ttl_seconds


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

# Global semaphore to limit concurrent request processing to 16
request_semaphore = Semaphore(16)


@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.

    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }

    Output format (when return_scores=True, similarity scores are returned):
    {
        "result": [
            [   # Results for each query
                {
                    {"document": doc, "score": score}
                },
                # ... more documents
            ],
            # ... results for other queries
        ]
    }
    """
    # Use semaphore to limit concurrent request count
    async with request_semaphore:
        print(f"[INFO] Received retrieval request, query count: {len(request.queries)}, topk: {request.topk}")
        print(f"[INFO] Current queued requests: {16 - request_semaphore._value}")
        
        if not request.topk:
            request.topk = config.retrieval_topk  # fallback to default

        request.return_scores = True
        
        # Run synchronous batch search in async context
        loop = asyncio.get_event_loop()
        results, scores = await loop.run_in_executor(
            None, 
            lambda: retriever.batch_search(
                query_list=request.queries, 
                num=request.topk, 
                return_score=request.return_scores
            )
        )

        # Format response
        resp = []
        for i, single_result in enumerate(results):
            for doc in single_result:
                doc.pop("id", None)
                doc.pop("wikipedia_id", None)
            if request.return_scores:
                # If scores are returned, combine them with results
                combined = []
                for doc, score in zip(single_result, scores[i], strict=True):
                    combined.append({"document": doc})
                resp.append(combined)
            else:
                resp.append(single_result)
        
        print(f"[INFO] Retrieval request processed, returned {len(resp)} results")
        return {"result": resp}


@app.get("/queue_status")
async def queue_status():
    """
    Get current queue status information.
    
    Return format:
    {
        "available_slots": available slot count,
        "processing_requests": number of requests being processed,
        "total_capacity": total capacity
    }
    """
    available_slots = request_semaphore._value
    processing_requests = 16 - available_slots
    
    return {
        "available_slots": available_slots,
        "processing_requests": processing_requests,
        "total_capacity": 16,
        "queue_full": available_slots == 0
    }


@app.get("/cache_status")
async def cache_status():
    """
    Get current cache status information.
    
    Return format:
    {
        "cache_stats": cache internal statistics,
        "hit_rate": cache hit rate,
        "cache_hits": cache hit count,
        "cache_misses": cache miss count,
        "total_requests": total request count
    }
    """
    cache_stats = retriever.cache.get_stats()
    
    with retriever._cache_stats_lock:
        hits = retriever.cache_hits
        misses = retriever.cache_misses
    
    total_requests = hits + misses
    hit_rate = hits / total_requests if total_requests > 0 else 0.0
    
    return {
        "cache_stats": cache_stats,
        "hit_rate": round(hit_rate, 4),
        "cache_hits": hits,
        "cache_misses": misses,
        "total_requests": total_requests
    }


@app.post("/cache_clear")
async def cache_clear():
    """
    Clear cache.
    
    Return format:
    {
        "message": "Cache cleared",
        "cache_size_before": cache size before clearing
    }
    """
    size_before = retriever.cache.size()
    retriever.cache.clear()
    
    # Reset statistics
    with retriever._cache_stats_lock:
        retriever.cache_hits = 0
        retriever.cache_misses = 0
    
    return {
        "message": "Cache cleared",
        "cache_size_before": size_before
    }


if __name__ == "__main__":
    print("[INFO] =============== Starting search server ===============")
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument(
        "--index_path", type=str, default="path/to/index.faiss", help="Corpus indexing file."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="path/to/corpus.parquet",
        help="Local corpus file.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="st", help="Name of the retriever model.")
    parser.add_argument(
        "--retriever_model", type=str, default="path/to/retriever_model", help="Path of the retriever model."
    )
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")
    parser.add_argument("--nprobe", type=int, default=128, help="FAISS IVF nprobe")
    parser.add_argument("--max_seq_len", type=int, default=32, help="Encoder max sequence length")
    parser.add_argument("--st_batch_size", type=int, default=8, help="SentenceTransformer encode batch size")
    parser.add_argument("--cache_maxsize", type=int, default=1024*1024*256, help="Cache maximum size")
    parser.add_argument("--cache_ttl", type=int, default=7*24*60*60, help="Cache TTL in seconds")

    args = parser.parse_args()
    print(f"[INFO] Command line arguments parsed")
    print(f"[INFO] Index path: {args.index_path}")
    print(f"[INFO] Corpus path: {args.corpus_path}")
    print(f"[INFO] Retriever name: {args.retriever_name}")
    print(f"[INFO] Retriever model: {args.retriever_model}")
    print(f"[INFO] Use GPU: {args.faiss_gpu}")
    print(f"[INFO] nprobe: {args.nprobe}")
    print(f"[INFO] max_seq_len: {args.max_seq_len}")
    print(f"[INFO] st_batch_size: {args.st_batch_size}")
    print(f"[INFO] cache_maxsize: {args.cache_maxsize}")
    print(f"[INFO] cache_ttl: {args.cache_ttl}")

    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    print(f"[INFO] Creating config object")
    config = Config(
        retrieval_method=args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=args.max_seq_len,
        retrieval_use_fp16=True,
        retrieval_batch_size=8,
        retrieval_nprobe=args.nprobe,
        retrieval_st_batch_size=args.st_batch_size,
        cache_maxsize=args.cache_maxsize,
        cache_ttl_seconds=args.cache_ttl,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    print(f"[INFO] Initializing retriever")
    retriever = get_retriever(config)
    print(f"[INFO] Retriever initialized")

    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    print(f"[INFO] Starting FastAPI server, listening on: http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
