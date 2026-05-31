# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

from .bridge.runtime import ensure_runtime


class LLM:
    DEFAULTS = {
        "generator_model": "gpt2",
        "embedding_model": "all-MiniLM-L6-v2",
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_k": 50,
        "do_sample": True,
        "use_local_models": False,
        "local_models_path": "/drive/models",
        "auto_detect_local": True,  # Automatically use local models if available
    }

    def __init__(self, js_instance):
        self.js = js_instance

    # --- lifecycle ---

    @classmethod
    async def create(cls, **overrides):
        import os

        ensure_runtime()

        from js import createLLM

        config = {**cls.DEFAULTS, **overrides}

        # Auto-detect local models if enabled
        if config.get("auto_detect_local", True) and not config.get("use_local_models", False):
            generator_model = config.get("generator_model", cls.DEFAULTS["generator_model"])
            local_model_path = f"{config['local_models_path']}/Xenova/{generator_model}"
            zip_path = f"{config['local_models_path']}/xenova-{generator_model}.zip"

            # Check for zip file first, then unzip if needed
            if os.path.exists(zip_path) and not os.path.exists(local_model_path):
                print(f"Found zipped model at {zip_path}, extracting...")
                import zipfile

                with zipfile.ZipFile(zip_path, "r") as zipf:
                    zipf.extractall(config["local_models_path"])
                print(f"Extracted to {config['local_models_path']}")

            if os.path.exists(local_model_path):
                print(f"Found local model at {local_model_path}, using local files")
                config["use_local_models"] = True
                # Convert filesystem path to emfs: URI for browser access
                # emfs: expects path relative to /drive, so models/Xenova/gpt2
                # Remove /drive/ prefix to get models/Xenova/gpt2
                emfs_path = local_model_path.replace("/drive/", "")
                config["emfs_model_uri"] = f"emfs:{emfs_path}"
                print(f"Using emfs URI: {config['emfs_model_uri']}")
            else:
                print(f"No local model found at {local_model_path}, will download from HuggingFace")

        instance = createLLM(config)
        await instance.init()

        return cls(instance)

    # --- core API ---

    async def generate(self, prompt: str):
        return await self.js.generate(prompt)

    async def embed(self, text: str):
        vec = await self.js.embed(text)
        return list(vec)

    # --- similarity ---

    async def similarity_search(self, query, docs):
        from js import cosineSimilarityBatch

        query_vec = await self.embed(query)

        doc_vecs = []
        for doc in docs:
            doc_vecs.append(await self.embed(doc))

        sims = cosineSimilarityBatch(query_vec, doc_vecs)

        results = list(zip(docs, sims))
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    # --- model export ---

    async def export_model_files(self, model_name=None, as_zip=True):
        """Export model files from browser cache to /drive for download.

        Args:
            model_name: Name of the model to export (default: current generator model)
            as_zip: If True, export as a zip file; if False, export as individual files
        """
        import os
        import zipfile

        if model_name is None:
            model_name = self.DEFAULTS["generator_model"]

        # Check if model is already exported
        if as_zip:
            zip_path = f"/drive/models/xenova-{model_name}.zip"
            if os.path.exists(zip_path):
                message = f"Model already exported at {zip_path}"
                print(message)
                return message
        else:
            model_dir = f"/drive/models/Xenova/{model_name}"
            if os.path.exists(model_dir) and os.listdir(model_dir):
                message = f"Model already exported at {model_dir}"
                print(message)
                return message

        result = await self.js.exportModelFiles(model_name)

        # Check if result is an object with files (Python-side export needed)
        if hasattr(result, "success") and result.success:
            print(f"Exporting {len(result.files)} files for {result.model}")

            if as_zip:
                # Export as zip file
                zip_path = f"/drive/models/xenova-{result.model}.zip"
                os.makedirs("/drive/models", exist_ok=True)

                # Deduplicate files by name, keeping the largest version
                files_by_name = {}
                for file_obj in result.files:
                    filename = file_obj.filename
                    data = file_obj.data
                    file_bytes = bytes(data.to_py())

                    if filename not in files_by_name or len(file_bytes) > len(files_by_name[filename]):
                        files_by_name[filename] = file_bytes

                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for filename, file_bytes in files_by_name.items():
                        # Store with Xenova/{model}/ prefix
                        arcname = f"Xenova/{result.model}/{filename}"
                        zipf.writestr(arcname, file_bytes)
                        print(f"Added to zip: {arcname} ({len(file_bytes)} bytes)")

                message = f"Successfully exported {len(files_by_name)} files to {zip_path}"
                print(message)
                return message
            else:
                # Export as individual files
                model_dir = f"/drive/models/Xenova/{result.model}"
                os.makedirs(model_dir, exist_ok=True)
                print(f"Created directory: {model_dir}")

                # Deduplicate files by name, keeping the largest version
                files_by_name = {}
                for file_obj in result.files:
                    filename = file_obj.filename
                    data = file_obj.data
                    file_bytes = bytes(data.to_py())

                    if filename not in files_by_name or len(file_bytes) > len(files_by_name[filename]):
                        files_by_name[filename] = file_bytes

                for filename, file_bytes in files_by_name.items():
                    filepath = f"{model_dir}/{filename}"
                    with open(filepath, "wb") as f:
                        f.write(file_bytes)

                    print(f"Exported: {filepath} ({len(file_bytes)} bytes)")

                message = f"Successfully exported {len(files_by_name)} files to {model_dir}"
                print(message)
                return message
        else:
            # String result from JS side
            print(f"Export result: {result}")
            return result
