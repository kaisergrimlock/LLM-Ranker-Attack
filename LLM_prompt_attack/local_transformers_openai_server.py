"""Serve a cached Transformers model through the OpenAI chat-completions API.

This lightweight server is intended for sequential ranking evaluations on the
two P100 GPUs available on SEG's ``segsresap07``. It is not a replacement for a
high-throughput inference engine.
"""

from __future__ import annotations

import argparse
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse server configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-memory", default="13GiB")
    return parser.parse_args()


def create_app(args: argparse.Namespace) -> FastAPI:
    """Create an OpenAI-compatible server for one cached model."""
    state: dict[str, Any] = {"model": None, "tokenizer": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("This server requires two visible CUDA GPUs.")

        tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="balanced",
            max_memory={0: args.max_memory, 1: args.max_memory},
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        model.eval()
        state["model"] = model
        state["tokenizer"] = tokenizer
        print(f"Serving {args.served_model_name}: {model.hf_device_map}", flush=True)
        yield
        del state["model"]
        torch.cuda.empty_cache()

    app = FastAPI(lifespan=lifespan)

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": args.served_model_name, "object": "model"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        if body.get("model") != args.served_model_name:
            raise HTTPException(status_code=404, detail="Unknown model")
        if body.get("stream"):
            raise HTTPException(status_code=400, detail="Streaming is unsupported")

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(
                status_code=400, detail="messages must be a non-empty list"
            )

        tokenizer = state["tokenizer"]
        model = state["model"]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(model.get_input_embeddings().weight.device)
        max_tokens = min(int(body.get("max_tokens", 3)), 32)
        if max_tokens < 1:
            raise HTTPException(status_code=400, detail="max_tokens must be positive")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ],
            )

        generated = output[0, inputs["input_ids"].shape[1] :]
        content = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": args.served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": inputs["input_ids"].shape[1],
                    "completion_tokens": generated.shape[0],
                    "total_tokens": output.shape[1],
                },
            }
        )

    return app


if __name__ == "__main__":
    server_args = parse_args()
    import uvicorn

    uvicorn.run(create_app(server_args), host=server_args.host, port=server_args.port)
