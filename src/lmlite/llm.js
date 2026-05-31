// lmLite JS runtime

class BrowserLLM {
  constructor(config) {
    this.generatorModel = config.generator_model;
    this.embeddingModel = config.embedding_model;
    this.useLocalModels = config.use_local_models ?? false;
    this.localModelsPath = config.local_models_path ?? '/drive/models';
    this.emfsModelUri = config.emfs_model_uri ?? null;

    this.max_new_tokens = config.max_new_tokens ?? 50;
    this.temperature = config.temperature ?? 0.7;
    this.top_k = config.top_k ?? 50;
    this.do_sample = config.do_sample ?? true;

    this.generator = null;
    this.embedder = null;
  }

  async init() {
    const { pipeline, env } = await import(
      "https://cdn.jsdelivr.net/npm/@xenova/transformers"
    );

    // Configure for browser environment
    if (this.useLocalModels) {
      env.allowLocalModels = false;
      env.useBrowserCache = false;
      env.allowRemoteModels = true; // Keep true but intercept fetch

      // Set up custom backend to read from Pyodide FS
      const pyodide = globalThis.pyodide;
      if (pyodide) {
        env.backends.onnx.wasm.proxy = false; // Disable web worker for local files

        // Custom fetch that reads from Pyodide FS
        const originalFetch = globalThis.fetch;
        globalThis.fetch = async (url, options) => {
          const urlStr = typeof url === 'string' ? url : url.toString();

          // Check if this is a model file request for HuggingFace
          if (urlStr.includes('huggingface.co') &&
              (urlStr.includes(`Xenova/${this.generatorModel}`) ||
               (this.embeddingModel && urlStr.includes(`Xenova/${this.embeddingModel}`)))) {

            // Extract filename from URL
            const parts = urlStr.split('/');
            const filename = parts[parts.length - 1];
            const modelName = parts[parts.indexOf('Xenova') + 1];
            const filepath = `${this.localModelsPath}/Xenova/${modelName}/${filename}`;

            console.log(`Intercepting HF request, reading from Pyodide FS: ${filepath}`);

            try {
              const data = pyodide.FS.readFile(filepath);
              const blob = new Blob([data]);

              return new Response(blob, {
                status: 200,
                statusText: 'OK',
                headers: new Headers({
                  'Content-Type': 'application/octet-stream',
                  'Content-Length': data.length.toString()
                })
              });
            } catch (err) {
              console.error(`Failed to read ${filepath}:`, err);
              return new Response(null, { status: 404, statusText: 'Not Found' });
            }
          }

          // Fall back to original fetch for non-model requests
          return originalFetch(url, options);
        };

        console.log(`Using local models from Pyodide FS: ${this.localModelsPath}`);
      }
    } else {
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      console.log('Using remote models with browser cache (IndexedDB)');
    }

    try {
      let modelPath;
      if (this.useLocalModels) {
        // Use HuggingFace path format - our custom fetch will intercept
        modelPath = `Xenova/${this.generatorModel}`;
      } else {
        // Download from HuggingFace
        modelPath = `Xenova/${this.generatorModel}`;
      }

      console.log(`Loading text-generation model: ${modelPath}`);

      // Add progress callback
      this.generator = await pipeline(
        "text-generation",
        modelPath,
        {
          // Only use quantized models when downloading from HuggingFace
          // Local models are usually not quantized
          quantized: !this.useLocalModels,
          // Explicitly use WASM backend
          device: 'wasm',
          // Progress callback
          progress_callback: (progress) => {
            if (progress.status === 'downloading') {
              console.log(`Downloading ${progress.file}: ${progress.progress?.toFixed(2)}%`);
            } else if (progress.status === 'done') {
              console.log(`Downloaded ${progress.file}`);
            } else if (progress.status === 'ready') {
              console.log(`Model ready: ${progress.file}`);
            } else if (progress.status === 'progress') {
              console.log(`Loading ${progress.file}: ${progress.progress?.toFixed(2)}%`);
            }
          }
        }
      );

      console.log(`Model loaded successfully: ${this.generatorModel}`);

      // Warm up the model with a dummy inference to ensure it's fully ready
      console.log(`Warming up model...`);
      await this.generator("test", { max_new_tokens: 1 });
      console.log(`Model warm-up complete`);

    } catch (error) {
      throw new Error(
        `Failed to load text-generation model '${this.generatorModel}': ${error.message}. ` +
        `This may be due to model incompatibility with browser WASM/WebGPU. ` +
        `Try a different model like 'gpt2' or check model availability.`
      );
    }

    if (this.embeddingModel) {
      try {
        // Always use HuggingFace format - custom fetch intercepts if local
        const embeddingPath = `Xenova/${this.embeddingModel}`;

        console.log(`Loading embedding model: ${embeddingPath}`);
        this.embedder = await pipeline(
          "feature-extraction",
          embeddingPath
        );
        console.log(`Embedding model loaded successfully: ${this.embeddingModel}`);
      } catch (error) {
        throw new Error(
          `Failed to load embedding model '${this.embeddingModel}': ${error.message}`
        );
      }
    }
  }

  async exportModelFiles(modelName) {
    try {
      console.log(`Starting export for model: ${modelName}`);

      // Check both Cache API and IndexedDB
      const cacheNames = await caches.keys();
      console.log('Available Cache API caches:', cacheNames);

      // List all IndexedDB databases to find the right one
      const databases = await indexedDB.databases();
      console.log('Available IndexedDB databases:', databases.map(db => db.name));

      // First try Cache API (Transformers.js v3+ uses this)
      const modelCacheName = cacheNames.find(name =>
        name.includes('transformers') ||
        name.includes('huggingface') ||
        name.includes('onnx')
      );

      if (modelCacheName) {
        console.log(`Found model cache: ${modelCacheName}`);
        return await this._exportFromCacheAPI(modelName, modelCacheName);
      }

      // Try different possible database names for IndexedDB (older versions)
      const possibleDbNames = [
        'transformers-cache',
        'transformers_cache',
        'huggingface-assets',
        'transformers.js',
        ...databases.map(db => db.name)
      ];

      let db = null;
      let dbName = null;

      for (const name of possibleDbNames) {
        try {
          db = await new Promise((resolve, reject) => {
            const request = indexedDB.open(name);
            request.onsuccess = () => {
              if (request.result.objectStoreNames.length > 0) {
                resolve(request.result);
              } else {
                request.result.close();
                resolve(null);
              }
            };
            request.onerror = () => resolve(null);
          });
          if (db) {
            dbName = name;
            console.log(`Found database: ${dbName}`);
            break;
          }
        } catch (e) {
          console.log(`Database ${name} not found or empty`);
        }
      }

      if (!db) {
        console.error('No IndexedDB cache found. Available databases:', databases);
        return `Error: No model cache database found. Models may still be downloading or using a different cache mechanism.`;
      }

      console.log(`Object stores in ${dbName}:`, Array.from(db.objectStoreNames));

      const storeName = db.objectStoreNames[0];
      const transaction = db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);

      const allKeys = await new Promise((resolve, reject) => {
        const request = store.getAllKeys();
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });

      console.log(`Total cached entries: ${allKeys.length}`);
      console.log(`Sample keys:`, allKeys.slice(0, 5));

      // Filter keys for this model
      const modelKeys = allKeys.filter(key =>
        String(key).includes(`Xenova/${modelName}`) || String(key).includes(modelName)
      );

      console.log(`Found ${modelKeys.length} cached files for ${modelName}`);

      if (modelKeys.length === 0) {
        return `Error: No cached files found for model '${modelName}'. Model may not be cached yet.`;
      }

      // Create directory for model files
      const pyodide = globalThis.pyodide;
      if (!pyodide) {
        return 'Error: Pyodide not available, cannot export to filesystem';
      }

      const modelDir = `/drive/models/Xenova/${modelName}`;
      pyodide.FS.mkdirTree(modelDir);
      console.log(`Created directory: ${modelDir}`);

      let exportedCount = 0;

      // Export each file
      for (const key of modelKeys) {
        const data = await new Promise((resolve, reject) => {
          const request = store.get(key);
          request.onsuccess = () => resolve(request.result);
          request.onerror = () => reject(request.error);
        });

        console.log(`Processing key: ${key}, data type:`, typeof data, data);

        if (data) {
          let arrayBuffer;

          // Handle different data formats
          if (data instanceof Blob) {
            arrayBuffer = await data.arrayBuffer();
          } else if (data.blob instanceof Blob) {
            arrayBuffer = await data.blob.arrayBuffer();
          } else if (data instanceof ArrayBuffer) {
            arrayBuffer = data;
          } else if (data.data instanceof ArrayBuffer) {
            arrayBuffer = data.data;
          } else {
            console.warn(`Unknown data format for key ${key}:`, data);
            continue;
          }

          const uint8Array = new Uint8Array(arrayBuffer);

          // Extract filename from key
          const filename = String(key).split('/').pop();
          const filepath = `${modelDir}/${filename}`;

          pyodide.FS.writeFile(filepath, uint8Array);
          console.log(`Exported: ${filepath} (${uint8Array.length} bytes)`);
          exportedCount++;
        }
      }

      db.close();

      const message = `Successfully exported ${exportedCount} files to ${modelDir}`;
      console.log(message);
      return message;

    } catch (error) {
      const errorMsg = `Export failed: ${error.message}`;
      console.error(errorMsg, error);
      return errorMsg;
    }
  }

  async _exportFromCacheAPI(modelName, cacheName) {
    try {
      const cache = await caches.open(cacheName);
      const requests = await cache.keys();

      console.log(`Total cached requests: ${requests.length}`);

      // Filter for this model
      const modelRequests = requests.filter(req =>
        req.url.includes(`Xenova/${modelName}`) || req.url.includes(modelName)
      );

      console.log(`Found ${modelRequests.length} cached files for ${modelName}`);

      if (modelRequests.length === 0) {
        return `Error: No cached files found for model '${modelName}' in Cache API.`;
      }

      // Access Pyodide filesystem (in web worker context)
      // Try different ways to access pyodide
      let pyodide = null;
      try {
        pyodide = self.pyodide;
      } catch (e) {}

      if (!pyodide) {
        try {
          pyodide = globalThis.pyodide;
        } catch (e) {}
      }

      if (!pyodide) {
        console.error('Pyodide not found in self or globalThis');
        // Return data to Python side instead
        return await this._exportViaPython(modelName, cacheName, modelRequests);
      }

      console.log('Pyodide found, exporting via FS:', pyodide);

      const modelDir = `/drive/models/Xenova/${modelName}`;

      // Create directory using Pyodide FS
      try {
        pyodide.FS.mkdirTree(modelDir);
        console.log(`Created directory: ${modelDir}`);
      } catch (e) {
        console.error('Failed to create directory:', e);
        return `Error: Failed to create directory ${modelDir}: ${e.message}`;
      }

      let exportedCount = 0;

      for (const request of modelRequests) {
        const response = await cache.match(request);
        if (response) {
          const arrayBuffer = await response.arrayBuffer();
          const uint8Array = new Uint8Array(arrayBuffer);

          // Extract filename from URL
          const url = new URL(request.url);
          const pathParts = url.pathname.split('/');
          const filename = pathParts[pathParts.length - 1];

          const filepath = `${modelDir}/${filename}`;
          pyodide.FS.writeFile(filepath, uint8Array);
          console.log(`Exported: ${filepath} (${uint8Array.length} bytes)`);
          exportedCount++;
        }
      }

      const message = `Successfully exported ${exportedCount} files to ${modelDir}`;
      console.log(message);
      return message;

    } catch (error) {
      const errorMsg = `Cache API export failed: ${error.message}`;
      console.error(errorMsg, error);
      return errorMsg;
    }
  }

  async _exportViaPython(modelName, cacheName, modelRequests) {
    // Export by returning data to Python side
    console.log('Exporting via Python side');

    const cache = await caches.open(cacheName);
    const files = [];

    for (const request of modelRequests) {
      const response = await cache.match(request);
      if (response) {
        const arrayBuffer = await response.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);

        // Extract filename from URL
        const url = new URL(request.url);
        const pathParts = url.pathname.split('/');
        const filename = pathParts[pathParts.length - 1];

        files.push({
          filename: filename,
          data: uint8Array
        });
      }
    }

    return {
      success: true,
      model: modelName,
      files: files
    };
  }

  async generate(prompt) {
    if (!this.generator) {
      throw new Error("Generator model not initialized. Call init() first.");
    }

    try {
      const output = await this.generator(prompt, {
        max_new_tokens: this.max_new_tokens,
        temperature: this.temperature,
        top_k: this.top_k,
        do_sample: this.do_sample
      });

      return output[0].generated_text;
    } catch (error) {
      throw new Error(
        `Text generation failed: ${error.message}. ` +
        `This could be due to: ` +
        `(1) Model still downloading - wait and retry, ` +
        `(2) Memory constraints in browser, ` +
        `(3) ONNX Runtime error - try a different model like 'gpt2', ` +
        `(4) Model incompatibility with browser WASM environment.`
      );
    }
  }

  async embed(text) {
    if (!this.embedder) {
      throw new Error("No embedding model configured");
    }

    const result = await this.embedder(text);
    return Array.from(result.data);
  }
}

// --- Factory ---

function createLLM(config) {
  return new BrowserLLM(config);
}

// --- Similarity helpers ---

function cosineSimilarity(a, b) {
  let dot = 0.0, normA = 0.0, normB = 0.0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function cosineSimilarityBatch(queryVec, docVecs) {
  return docVecs.map(doc => cosineSimilarity(queryVec, doc));
}

// --- expose globally (critical for Python) ---

globalThis.createLLM = createLLM;
globalThis.cosineSimilarity = cosineSimilarity;
globalThis.cosineSimilarityBatch = cosineSimilarityBatch;

undefined;

