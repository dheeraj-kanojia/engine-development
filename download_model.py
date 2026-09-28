#!/usr/bin/env python3
"""
Download DeepSeek-VL model for offline use.
Run this once to cache the model locally before offline processing.
"""

import argparse
from pathlib import Path


def download_model(cache_dir: str) -> None:
    """
    Download DeepSeek-VL 1.3B model to local cache.

    Args:
        cache_dir: Local folder to cache the model
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DeepSeek-VL 1.3B Model Downloader")
    print("=" * 60)
    print(f"\nDownload destination: {cache_path.absolute()}")
    print("Model size: ~2.7 GB")
    print("Estimated time: 5-10 minutes (depending on connection)\n")

    try:
        print("Validating runtime...")
        from transformers import AutoModelForCausalLM, __version__ as transformers_version
        from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor

        major_version = int(transformers_version.split(".", maxsplit=1)[0])
        if major_version >= 5:
            raise RuntimeError(
                f"Incompatible transformers version: {transformers_version}\n"
                "DeepSeek-VL requires transformers>=4.38.2,<5\n"
                "Run: pip install 'transformers>=4.38.2,<5'"
            )

        print(f"✓ transformers {transformers_version}")

        model_id = "deepseek-ai/deepseek-vl-1.3b-chat"

        print(f"\nDownloading processor...")
        VLChatProcessor.from_pretrained(model_id, cache_dir=str(cache_path))
        print(f"✓ Processor downloaded")

        print(f"\nDownloading model (this may take a few minutes)...")
        AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=str(cache_path),
        )
        print(f"✓ Model downloaded")

        print("\n" + "=" * 60)
        print("✓ SUCCESS: Model cached for offline use")
        print("=" * 60)

        print(f"\nModel location: {cache_path.absolute()}")
        print(f"\nTo run OCR offline, use:")
        print(f"  python test.py <image> --cache-dir {cache_path}")
        print(f"\nFor batch processing:")
        print(f"  python batch_ocr.py <images_dir> --cache-dir {cache_path}")
        print(f"\nFor web interface:")
        print(f"  python app_with_ocr.py --cache-dir {cache_path}")

        print(f"\nNo internet connection needed for inference! 🚀\n")

    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download DeepSeek-VL model for offline use"
    )
    parser.add_argument(
        "--cache-dir",
        default="./models/deepseek-vl-1.3b-chat",
        help="Local folder to cache model (default: ./models/deepseek-vl-1.3b-chat)",
    )

    args = parser.parse_args()
    download_model(args.cache_dir)


if __name__ == "__main__":
    main()

