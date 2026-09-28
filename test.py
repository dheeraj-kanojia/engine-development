from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, __version__ as transformers_version


DEFAULT_MODEL = "deepseek-ai/deepseek-vl-1.3b-chat"
DEFAULT_PROMPT = "Please perform OCR on this image and extract all text exactly as it appears."


def ensure_compatible_runtime():
    """Validate that the installed packages match DeepSeek-VL's supported API."""
    major_version = int(transformers_version.split(".", maxsplit=1)[0])
    if major_version >= 5:
        raise RuntimeError(
            "DeepSeek-VL is not compatible with transformers>=5 in this script. "
            "Install the versions from requirements.txt (transformers>=4.38.2,<5)."
        )

    try:
        from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek-VL is not installed. Install dependencies from requirements.txt "
            "to add the official DeepSeek-VL package."
        ) from exc

    return MultiModalityCausalLM, VLChatProcessor


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    return requested_device


def resolve_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


def build_conversation(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "User",
            "content": f"<image_placeholder>{prompt}",
        },
        {"role": "Assistant", "content": ""},
    ]


def load_components(model_path: str, device: str, dtype: torch.dtype, cache_dir: str | None = None):
    MultiModalityCausalLM, VLChatProcessor = ensure_compatible_runtime()

    print(f"Loading processor from {model_path}...")
    processor_kwargs = {}
    if cache_dir:
        processor_kwargs["cache_dir"] = cache_dir
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, **processor_kwargs)
    tokenizer = vl_chat_processor.tokenizer

    print(f"Loading model on {device} (this may take a few minutes)...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    model = model.to(device).eval()
    return tokenizer, vl_chat_processor, model


def run_ocr(
    image_path: Path,
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    device: str,
    cache_dir: str | None = None,
) -> str:
    dtype = resolve_dtype(device)
    tokenizer, vl_chat_processor, model = load_components(model_path, device, dtype, cache_dir)

    image = Image.open(image_path).convert("RGB")
    conversation = build_conversation(prompt)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[image],
        force_batchify=True,
    ).to(device=device, dtype=dtype)

    print("Running OCR inference...")
    with torch.inference_mode():
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        output_ids = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    prompt_token_count = prepare_inputs.input_ids.shape[1]
    generated_ids = output_ids[0, prompt_token_count:]
    if generated_ids.numel() == 0:
        generated_ids = output_ids[0]

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR against an image using the official DeepSeek-VL pipeline."
    )
    parser.add_argument("image_path", nargs="?", help="Path to the image to OCR.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL, help="Model id or local path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction sent to the model.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum generated tokens.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Where to run the model.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local folder to load the model from (for offline mode).",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Validate DeepSeek-VL imports and version compatibility without loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.check_env:
        ensure_compatible_runtime()
        print("DeepSeek-VL runtime check passed.")
        return

    if not args.image_path:
        raise SystemExit("Provide an image path, or use --check-env to validate dependencies only.")

    image_path = Path(args.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    device = resolve_device(args.device)
    extracted_text = run_ocr(
        image_path=image_path,
        model_path=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
        cache_dir=args.cache_dir,
    )

    print("\n--- Extracted Text ---")
    print(extracted_text)


if __name__ == "__main__":
    main()



