# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch


class TestLLMDefaults:
    """Test LLM class defaults and configuration."""

    def test_defaults(self):
        from litelm import LLM

        assert LLM.DEFAULTS["generator_model"] == "gpt2"
        assert LLM.DEFAULTS["embedding_model"] == "all-MiniLM-L6-v2"
        assert LLM.DEFAULTS["max_new_tokens"] == 50
        assert LLM.DEFAULTS["temperature"] == 0.7
        assert LLM.DEFAULTS["top_k"] == 50
        assert LLM.DEFAULTS["do_sample"] is True
        assert LLM.DEFAULTS["use_local_models"] is False
        assert LLM.DEFAULTS["local_models_path"] == "/drive/models"
        assert LLM.DEFAULTS["auto_detect_local"] is True


class TestLLMGenerate:
    """Test LLM text generation."""

    async def test_generate(self):
        """Test generate method calls JS instance."""
        from litelm import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.generate = AsyncMock(return_value="Generated text")

        llm = LLM(mock_js_instance)
        result = await llm.generate("Test prompt")

        assert result == "Generated text"
        mock_js_instance.generate.assert_awaited_once_with("Test prompt")


class TestLLMEmbed:
    """Test LLM embedding generation."""

    async def test_embed(self):
        """Test embed method calls JS instance and returns list."""
        from litelm import LLM

        mock_js_instance = MagicMock()
        # JS returns a JS array-like object
        mock_vec = [0.1, 0.2, 0.3]
        mock_js_instance.embed = AsyncMock(return_value=mock_vec)

        llm = LLM(mock_js_instance)
        result = await llm.embed("Test text")

        assert result == [0.1, 0.2, 0.3]
        mock_js_instance.embed.assert_awaited_once_with("Test text")


class TestLLMSimilaritySearch:
    """Test LLM similarity search."""

    async def test_similarity_search(self):
        """Test similarity search with mocked JS function."""
        from litelm import LLM

        mock_js_instance = MagicMock()
        # Mock embeddings
        mock_js_instance.embed = AsyncMock(
            side_effect=[
                [1.0, 0.0],  # query
                [1.0, 0.0],  # doc1 - perfect match
                [0.0, 1.0],  # doc2 - orthogonal
                [0.7, 0.7],  # doc3 - partial match
            ]
        )

        llm = LLM(mock_js_instance)

        # Mock the JS cosineSimilarityBatch function
        mock_similarities = [1.0, 0.0, 0.7]

        import sys

        mock_js_module = MagicMock()
        mock_js_module.cosineSimilarityBatch = MagicMock(return_value=mock_similarities)
        sys.modules["js"] = mock_js_module

        # Mock to_js to just return the input (simulating successful conversion)
        mock_pyodide_module = MagicMock()
        mock_pyodide_module.ffi = MagicMock()
        mock_pyodide_module.ffi.to_js = MagicMock(side_effect=lambda x: x)

        try:
            # Mock pyodide.ffi module
            sys.modules["pyodide"] = mock_pyodide_module
            sys.modules["pyodide.ffi"] = mock_pyodide_module.ffi

            results = await llm.similarity_search("query", ["doc1", "doc2", "doc3"])

            # Results should be sorted by similarity (descending)
            assert len(results) == 3
            assert results[0] == ("doc1", 1.0)
            assert results[1] == ("doc3", 0.7)
            assert results[2] == ("doc2", 0.0)

            # Verify cosineSimilarityBatch was called
            mock_js_module.cosineSimilarityBatch.assert_called_once()
        finally:
            if "js" in sys.modules:
                del sys.modules["js"]
            if "pyodide" in sys.modules:
                del sys.modules["pyodide"]
            if "pyodide.ffi" in sys.modules:
                del sys.modules["pyodide.ffi"]


class TestLLMExportModelFiles:
    """Test LLM model export functionality."""

    async def test_export_model_files_already_exists_zip(self):
        """Test that export skips if zip already exists."""
        from litelm import LLM

        mock_js_instance = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing zip
            zip_path = os.path.join(tmpdir, "xenova-gpt2.zip")
            with open(zip_path, "w") as f:
                f.write("fake zip")

            llm = LLM(mock_js_instance)

            # Mock os.path.exists to return True for our test zip
            def mock_exists(path):
                if "xenova-gpt2.zip" in path:
                    return True
                return False

            with patch("os.path.exists", side_effect=mock_exists):
                result = await llm.export_model_files(model_name="gpt2", as_zip=True)

                assert "already exported" in result.lower()

    async def test_export_model_files_already_exists_dir(self):
        """Test that export skips if directory already exists."""
        from litelm import LLM

        mock_js_instance = MagicMock()
        llm = LLM(mock_js_instance)

        # Simply mock to return True and non-empty list
        with patch("os.path.exists", return_value=True), patch("os.listdir", return_value=["config.json"]):
            result = await llm.export_model_files(model_name="gpt2", as_zip=False)

            assert "already exported" in result.lower()

    async def test_export_model_files_string_result(self):
        """Test handling string result from JS side."""
        from litelm import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.exportModelFiles = AsyncMock(return_value="Export complete")

        llm = LLM(mock_js_instance)

        with patch("os.path.exists", return_value=False):
            result = await llm.export_model_files(model_name="gpt2", as_zip=True)

            assert result == "Export complete"

    async def test_export_model_files_with_js_files(self):
        """Test exporting model files when JS returns file objects."""
        from litelm import LLM

        mock_js_instance = MagicMock()

        # Mock JS result with files
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.model = "gpt2"

        # Mock file objects
        mock_file1 = MagicMock()
        mock_file1.filename = "config.json"
        mock_data1 = MagicMock()
        mock_data1.to_py = MagicMock(return_value=b'{"key": "value"}')
        mock_file1.data = mock_data1

        mock_result.files = [mock_file1]
        mock_js_instance.exportModelFiles = AsyncMock(return_value=mock_result)

        llm = LLM(mock_js_instance)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.path.exists", return_value=False), patch("os.makedirs"):
                # Patch zipfile to write to tmpdir
                original_init = zipfile.ZipFile.__init__

                def patched_init(self, file, *args, **kwargs):
                    if isinstance(file, str) and "/drive/models" in file:
                        file = os.path.join(tmpdir, os.path.basename(file))
                    return original_init(self, file, *args, **kwargs)

                with patch.object(zipfile.ZipFile, "__init__", patched_init):
                    result = await llm.export_model_files(model_name="gpt2", as_zip=True)

                    assert "Successfully exported" in result
                    assert "1 files" in result


class TestLLMAutoDetect:
    """Test auto-detection of local models."""

    async def test_auto_detect_zip_extraction(self):
        """Test that zip files are automatically extracted."""
        from litelm import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a zip file
            zip_path = os.path.join(tmpdir, "xenova-gpt2.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("Xenova/gpt2/config.json", "{}")
                zf.writestr("Xenova/gpt2/vocab.json", "{}")

            # Mock ensure_runtime and js module
            import sys

            mock_js_module = MagicMock()
            mock_js_module.createLLM = MagicMock(return_value=mock_js_instance)

            with patch("litelm.llm.ensure_runtime"):
                sys.modules["js"] = mock_js_module
                try:
                    await LLM.create(local_models_path=tmpdir, generator_model="gpt2")

                    # Verify the zip was extracted
                    extracted_dir = os.path.join(tmpdir, "Xenova", "gpt2")
                    assert os.path.exists(extracted_dir)
                    assert os.path.exists(os.path.join(extracted_dir, "config.json"))
                    assert os.path.exists(os.path.join(extracted_dir, "vocab.json"))

                    # Verify createLLM was called with use_local_models=True
                    config = mock_js_module.createLLM.call_args[0][0]
                    # Config might be dict or dict-like object after to_js conversion
                    if hasattr(config, "get"):
                        assert config.get("use_local_models") is True
                        assert "emfs_model_uri" in config
                    else:
                        # It's a plain dict in CPython tests
                        assert config["use_local_models"] is True
                        assert "emfs_model_uri" in config
                finally:
                    if "js" in sys.modules:
                        del sys.modules["js"]
