# LlamaHome Performance Metrics

## Current Performance

### Document Processing

- Text files: ~2000 docs/min
- Office formats: ~500 docs/min
- PDF processing: ~200 docs/min

### Memory Usage

- Base system: ~100MB
- Document processing: 200MB-1GB (depending on batch size)
- Model inference: Varies by model size
  - 7B: ~8GB GPU VRAM
  - 13B: ~16GB GPU VRAM
  - 70B: ~80GB GPU VRAM
  - CPU Memory Requirements:
    - 7B: ~14GB RAM
    - 13B: ~28GB RAM
    - 70B: ~140GB RAM

### Response Times

- Document preprocessing: 50-200ms/doc
- API response: <100ms
- Model inference: 100-500ms (varies by length)

### ROUGE Scores

Latest benchmark results on test dataset:

- ROUGE-1:
  - F1: 45.2%
  - Precision: 48.7%
  - Recall: 42.8%
- ROUGE-2:
  - F1: 21.3%
  - Precision: 23.5%
  - Recall: 19.6%
- ROUGE-L:
  - F1: 42.1%
  - Precision: 44.8%
  - Recall: 39.9%

## Optimization Goals

### Phase 1 (Current)

- Basic caching implementation
- Document batch processing
- Memory-mapped file handling

### Phase 2 (Planned)

- Enhanced caching system
- Response streaming
- GPU memory optimization

### Phase 3 (Future)

- Distributed processing
- Load balancing
- Real-time monitoring

## Benchmarks

Run benchmarks with:

```bash
python run.py benchmark --iterations 100
```

### Latest Benchmarks

- Document processing: 1000 docs in 30s
- Model inference: 100 requests in 45s
- Memory efficiency: 85% GPU utilization

### Long Context Performance (with H2O)

- Maximum sequence length: 32K tokens
- KV cache optimization: Heavy-Hitter Oracle (H2O)
- Memory efficiency: ~50% reduction in KV cache size
- Performance retention: >95% with 512 window length

Run long context inference:

```bash
python run.py model --enable-h2o \
    --window-length 512 \
    --heavy-hitters 128 \
    --prompt "Your very long prompt here..."
```

### Needle-in-Haystack Performance

Testing information retrieval at various context depths:

| Context Length | Depth | Standard | With H2O |
|---------------|-------|-----------|----------|
| 2000          | 10%   | 98%      | 97%      |
| 2000          | 50%   | 95%      | 94%      |
| 2000          | 90%   | 92%      | 91%      |
| 8000          | 10%   | 90%      | 89%      |
| 8000          | 50%   | 85%      | 84%      |
| 8000          | 90%   | 80%      | 79%      |
| 32000         | 10%   | 75%      | 85%      |
| 32000         | 50%   | 65%      | 82%      |
| 32000         | 90%   | 55%      | 80%      |

Run needle-in-haystack tests:

```bash
make needle-test
```
