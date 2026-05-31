# LiteLM

Browser-native LLM orchestration in JupyterLite - run language models entirely in your browser using WebAssembly.

## Features

- **Browser-native**: Run LLMs entirely in the browser, no server required
- **Zero setup**: No installation, no API keys, works offline
- **Privacy-first**: All computation happens locally in your browser
- **Multiple models**: Support for GPT-2, DistilGPT-2, and more via Transformers.js
- **Embeddings**: Generate text embeddings for semantic search
- **Similarity search**: Built-in cosine similarity for RAG applications
- **Model caching**: Download once, use offline forever
- **Pythonic API**: Clean, async Python interface

<!--
## Quick Start

### Try the Demo (No Installation)

Visit the live demo at: [https://decitre.github.io/litelm/pyodide/](https://decitre.github.io/litelm/pyodide/)

Open the `demo.ipynb` notebook and run the cells to see LiteLM in action.

**Note**: LiteLM is designed to run in JupyterLite/Pyodide environments. For local development and testing, see the Development section below.
-->

## Usage

```python
from litelm import LLM

# Create LLM instance (downloads model on first run)
llm = await LLM.create(generator_model="gpt2")

# Generate text
text = await llm.generate("Python is a great language because")
print(text)

# Generate embeddings
embedding = await llm.embed("Hello world")
print(embedding[:5])  # First 5 dimensions

# Similarity search
docs = [
    "JupyterLite runs entirely in the browser.",
    "Python is widely used for machine learning.",
    "TypeScript is great for frontend applications.",
]

results = await llm.similarity_search(
    "Where does JupyterLite run?",
    docs
)

for doc, score in results:
    print(f"{score:.3f} -> {doc}")
```

### Configuration Options

```python
llm = await LLM.create(
    generator_model="gpt2",           # or "distilgpt2", etc.
    embedding_model="all-MiniLM-L6-v2",
    max_new_tokens=50,
    temperature=0.7,
    top_k=50,
    do_sample=True,
    use_local_models=False,           # Auto-detect local models
    local_models_path="/drive/models"
)
```

### Export Models for Offline Use

After using models in the browser, you can export them for offline use:

```python
# Export to zip file (default)
await llm.export_model_files("gpt2")

# Export to directory
await llm.export_model_files("gpt2", as_zip=False)
```

The exported files will be saved to `/drive/models/` and can be downloaded from JupyterLite.

## Development

### Prerequisites
[pixi](https://pixi.sh/) and [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) package managers

### Setup

```bash
# Clone the repository
git clone https://github.com/decitre/litelm.git
cd litelm
pixi install
```

### Development Tasks

```bash
# Run tests
pixi run test

# Quick tests (skip notebook tests)
pixi run quick-test

# Run linter
pixi run lint

# Check linting without fixing
pixi run lint-check

# Run tests with coverage
pixi run coverage

# Build wheel
pixi run wheel
```

### Build JupyterLite Demo

```bash
pixi run lite-build
```
Follow the instructions

### Run Tests in Different Python Versions

```bash
pixi run --environment py311 test
pixi run --environment py312 test
pixi run --environment py313 test
```

## How It Works

LiteLM bridges Python (via Pyodide) and JavaScript (via Transformers.js):

1. **JavaScript Layer**: Uses [@huggingface/transformers](https://www.npmjs.com/package/@xenova/transformers) to run ONNX models in the browser
2. **Python Bridge**: Exposes JavaScript functionality through a Pythonic async API
3. **Model Loading**:
   - First run: Downloads models from HuggingFace CDN
   - Cached: Uses browser's Cache API or IndexedDB
   - Local: Reads from Pyodide filesystem if available
4. **Execution**: Models run entirely in-browser using WebAssembly (WASM)

## Supported Models

### Text Generation
- `gpt2` - GPT-2 (124M parameters)
- `distilgpt2` - Smaller, faster GPT-2 variant

### Embeddings
- `all-MiniLM-L6-v2` - Sentence embeddings (384 dimensions)

For other models, check [Xenova's model list](https://huggingface.co/Xenova).

## Architecture

```
┌─────────────────────────────────────┐
│   Python (Pyodide/JupyterLite)     │
│                                     │
│   from litelm import LLM           │
│   llm = await LLM.create()         │
│   text = await llm.generate(...)   │
└─────────────┬───────────────────────┘
              │ Bridge
              │ (pyodide.ffi)
┌─────────────▼───────────────────────┐
│   JavaScript (Browser)              │
│                                     │
│   Transformers.js                   │
│   ├─ Model loading                  │
│   ├─ ONNX Runtime (WASM)           │
│   └─ WebGPU (optional)             │
└─────────────────────────────────────┘
```

## Examples

See the `notebooks/demo.ipynb` for a comprehensive tutorial covering:
- Basic text generation
- Embeddings and similarity search
- Building a simple RAG pipeline
- Model configuration and customization

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Run tests: `pixi run test`
4. Run linter: `pixi run lint`
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

**Dependencies:**
- [Transformers.js](https://github.com/xenova/transformers.js) - Apache 2.0

**Note**: Individual model licenses may vary. Check the model card on HuggingFace before use.

## Acknowledgments

- [Transformers.js](https://github.com/xenova/transformers.js) by [@xenova](https://github.com/xenova)
- [JupyterLite](https://jupyterlite.readthedocs.io/) team
- [Pyodide](https://pyodide.org/) project
- [Hugging Face](https://huggingface.co/) for model hosting
- [LLMs running in the browser](https://thekevinscott.com/llms-in-the-browser/)

## Troubleshooting

### Models downloading every time?

Check browser console for Cache API availability. Some privacy settings may disable caching.

### Out of memory errors?

Try smaller models like `distilgpt2` or reduce `max_new_tokens`.

### CORS errors?

Ensure you're running from `http://localhost` or a proper HTTPS domain, not `file://`.

## Citation

If you use LiteLM in your research, please cite:

```bibtex
@software{litelm2026,
  author = {Decitre, Emmanuel},
  title = {LiteLM: Browser-native LLM orchestration in JupyterLite},
  year = {2026},
  url = {https://github.com/decitre/litelm}
}
```
