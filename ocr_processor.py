"""
Offline DeepSeek-VL OCR processor for invoice images.
Supports batch and single-image processing with local model caching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, __version__ as transformers_version


DEFAULT_MODEL = "deepseek-ai/deepseek-vl-1.3b-chat"
DEFAULT_PROMPT = "Extract all text from this invoice and format as structured data with clear field labels."


class DeepSeekOCROffline:
    """Offline DeepSeek-VL OCR processor with model caching support."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize OCR processor.

        Args:
            model_path: Hugging Face model ID or local path
            cache_dir: Local folder to cache/load model from (for offline mode)
            device: 'auto', 'cpu', or 'cuda'
            dtype: torch dtype (auto-selected if None)
        """
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.device = self._resolve_device(device)
        self.dtype = dtype or self._resolve_dtype(self.device)

        self._validate_runtime()
        self.tokenizer = None
        self.processor = None
        self.model = None

    def _validate_runtime(self) -> None:
        """Validate transformers version compatibility."""
        major_version = int(transformers_version.split(".", maxsplit=1)[0])
        if major_version >= 5:
            raise RuntimeError(
                "DeepSeek-VL requires transformers<5. "
                "Install compatible versions from requirements.txt"
            )

        try:
            from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
        except ImportError as exc:
            raise RuntimeError(
                "DeepSeek-VL not installed. "
                "Install: pip install -r requirements.txt"
            ) from exc

    def _resolve_device(self, requested: str) -> str:
        """Resolve device to use."""
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return requested

    def _resolve_dtype(self, device: str) -> torch.dtype:
        """Resolve data type based on device."""
        return torch.bfloat16 if device == "cuda" else torch.float32

    def load_model(self) -> None:
        """Load model and processor from cache or download."""
        from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor

        print(f"Loading processor from {self.model_path}...")
        processor_kwargs = {}
        if self.cache_dir:
            processor_kwargs["cache_dir"] = self.cache_dir

        self.processor = VLChatProcessor.from_pretrained(
            self.model_path, **processor_kwargs
        )
        self.tokenizer = self.processor.tokenizer

        print(f"Loading model on {self.device}...")
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
        }
        if self.cache_dir:
            model_kwargs["cache_dir"] = self.cache_dir

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **model_kwargs
        )
        self.model = self.model.to(self.device).eval()

    def _build_conversation(self, prompt: str) -> list[dict[str, str]]:
        """Build conversation format for processor."""
        return [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt}",
            },
            {"role": "Assistant", "content": ""},
        ]

    def extract_text(
        self,
        image_path: str | Path,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Extract text from image using offline DeepSeek-VL.

        Args:
            image_path: Path to image file
            prompt: Extraction prompt
            max_new_tokens: Max tokens to generate

        Returns:
            Extracted text
        """
        if self.model is None:
            self.load_model()

        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"Processing: {image_path.name}")
        image = Image.open(image_path).convert("RGB")

        conversation = self._build_conversation(prompt)
        prepare_inputs = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
        ).to(device=self.device, dtype=self.dtype)

        print("Running OCR inference...")
        with torch.inference_mode():
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            output_ids = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        prompt_token_count = prepare_inputs.input_ids.shape[1]
        generated_ids = output_ids[0, prompt_token_count:]
        if generated_ids.numel() == 0:
            generated_ids = output_ids[0]

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def process_batch(
        self,
        image_dir: str | Path,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 1024,
        output_dir: Optional[str | Path] = None,
    ) -> list[dict[str, Any]]:
        """
        Process multiple images in a directory.

        Args:
            image_dir: Directory containing images
            prompt: Extraction prompt
            max_new_tokens: Max tokens to generate
            output_dir: Directory to save results (optional)

        Returns:
            List of extraction results
        """
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {image_dir}")

        output_dir = Path(output_dir) if output_dir else image_dir / "ocr_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

        for image_file in sorted(image_dir.iterdir()):
            if image_file.suffix.lower() not in image_extensions:
                continue

            try:
                text = self.extract_text(image_file, prompt, max_new_tokens)

                result = {
                    "file_name": image_file.name,
                    "file_path": str(image_file),
                    "extracted_text": text,
                    "status": "success",
                }

                # Save individual result
                output_file = output_dir / f"{image_file.stem}_ocr.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                results.append(result)
                print(f"✓ {image_file.name}")

            except Exception as e:
                result = {
                    "file_name": image_file.name,
                    "file_path": str(image_file),
                    "error": str(e),
                    "status": "failed",
                }
                results.append(result)
                print(f"✗ {image_file.name}: {e}")

        # Save batch results
        batch_output = output_dir / "batch_results.json"
        with open(batch_output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nProcessed {len(results)} images")
        print(f"Results saved to: {output_dir}")

        return results


def main():
    """CLI interface for offline OCR."""
    parser = argparse.ArgumentParser(
        description="Offline DeepSeek-VL OCR processor for invoices"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single image
    single = subparsers.add_parser("single", help="Process single image")
    single.add_argument("image_path", help="Path to image file")
    single.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="Extraction prompt"
    )
    single.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens to generate"
    )
    single.add_argument("--cache-dir", help="Model cache directory for offline mode")
    single.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    single.add_argument("--output", help="Save output to file")

    # Batch processing
    batch = subparsers.add_parser("batch", help="Process image directory")
    batch.add_argument("image_dir", help="Directory containing images")
    batch.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="Extraction prompt"
    )
    batch.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens to generate"
    )
    batch.add_argument("--cache-dir", help="Model cache directory for offline mode")
    batch.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    batch.add_argument(
        "--output-dir", help="Directory to save results (default: image_dir/ocr_output)"
    )

    # Check environment
    check = subparsers.add_parser("check", help="Validate environment")
    check.add_argument("--cache-dir", help="Model cache directory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        ocr = DeepSeekOCROffline(
            cache_dir=getattr(args, "cache_dir", None),
            device=getattr(args, "device", "auto"),
        )

        if args.command == "check":
            print("[OK] DeepSeek-VL runtime is compatible")
            return

        if args.command == "single":
            text = ocr.extract_text(
                args.image_path,
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
            )
            print("\n--- Extracted Text ---")
            print(text)

            if args.output:
                Path(args.output).write_text(text, encoding="utf-8")
                print(f"\nSaved to: {args.output}")

        elif args.command == "batch":
            results = ocr.process_batch(
                args.image_dir,
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                output_dir=args.output_dir,
            )

    except Exception as e:
        print(f"Error: {e}", flush=True)
        raise


if __name__ == "__main__":
    main()

