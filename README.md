# lmLite

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

<!-- To re-introduce when the Actions page is up
## Quick Start

### Try the Demo (No Installation)

**Recommended**: Visit the live demo with xeus-python kernel at:
[https://decitre.github.io/lmlite/](https://decitre.github.io/lmlite/)

**Alternative**: Pyodide kernel demo at:
[https://decitre.github.io/lmlite/pyodide/](https://decitre.github.io/lmlite/pyodide/)

Open the `demo.ipynb` notebook and run the cells to see LMLite in action.

### Install from PyPI

```bash
pip install lmlite
```

**Note**: LMLite is designed to run in JupyterLite environments (xeus-python or pyodide kernels). For local development and testing, see the Development section below.

-->

## Usage

```python
from lmlite import LLM

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

- [pixi](https://pixi.sh/) - Package manager (required)
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) - Conda package manager (required for JupyterLite builds)

**Supported Platforms**: macOS (Intel/Apple Silicon) and Linux. Windows is not currently supported for development.

### Setup

```bash
# Clone the repository
git clone https://github.com/decitre/lmlite.git
cd lmlite
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

Follow the instructions provided by the command.

### Run Tests in Different Python Versions

```bash
pixi run --environment py311 test
pixi run --environment py312 test
pixi run --environment py313 test
```

## How It Works

LMLite bridges Python (via xeus-python or pyodide) and JavaScript (via Transformers.js):

1. **JavaScript Layer**: Uses [@huggingface/transformers](https://www.npmjs.com/package/@xenova/transformers) to run ONNX models in the browser
2. **Python Bridge**: Exposes JavaScript functionality through a Pythonic async API
3. **Kernel Support**:
   - **xeus-python** (recommended): Full CPython in WebAssembly via emscripten
   - **pyodide**: Alternative WebAssembly Python runtime
4. **Model Loading**:
   - First run: Downloads models from HuggingFace CDN
   - Cached: Uses browser's Cache API or IndexedDB
   - Local: Reads from virtual filesystem if available
5. **Execution**: Models run entirely in-browser using WebAssembly (WASM)

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
│   Python (JupyterLite)              │
│   ├─ xeus-python (recommended)      │
│   └─ pyodide (alternative)          │
│                                     │
│   from lmlite import LLM            │
│   llm = await LLM.create()          │
│   text = await llm.generate(...)    │
└─────────────┬───────────────────────┘
              │ Bridge
              │ (pyodide.ffi / pjs)
┌─────────────▼───────────────────────┐
│   JavaScript (Browser)              │
│                                     │
│   Transformers.js                   │
│   ├─ Model loading                  │
│   ├─ ONNX Runtime (WASM)            │
│   └─ WebGPU (optional)              │
└─────────────────────────────────────┘
```


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

If you use LMLite in your research, please cite:

```bibtex
@software{lmlite2026,
  author = {Decitre, Emmanuel},
  title = {LMLite: Browser-native LLM orchestration in JupyterLite},
  year = {2026},
  url = {https://github.com/decitre/lmlite}
}
```
