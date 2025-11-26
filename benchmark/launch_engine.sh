#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_engine  —  Launch one of several LLM inference engines via Docker
#
# Usage:
#   ./launch_engine --engine=<engine> --model=<model>
#
#   <engine> ∈ { tgi | vllm | mii | sglang | lmdeploy | llamacpp}
#   <model>  is the Hugging Face model ID (e.g. “mistralai/Mistral-7B-Instruct-v0.3”)
#
# Example:
#   ./launch_engine --engine=tgi --model=mistralai/Mistral-7B-Instruct-v0.3
#
# Notes:
#   • Expects HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment.
#   • Always listens on 127.0.0.1:23333 inside the container→host.
#   • Uses $HOME/.cache/huggingface as cache Dir.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Parse arguments “--engine=…” and “--model=…” ────────────────────────────
ENGINE=""
MODEL=""

for ARG in "$@"; do
  case "$ARG" in
    --engine=*)
      ENGINE="${ARG#--engine=}"
      ;;
    --model=*)
      MODEL="${ARG#--model=}"
      ;;
    *)
      echo "Unknown argument: $ARG"
      echo "Usage: $0 --engine=<tgi|vllm|mii|sglang|lmdeploy|llamacpp> --model=<your-org/your-model-name>"
      exit 1
      ;;
  esac
done

if [[ -z "$ENGINE" || -z "$MODEL" ]]; then
  echo "Error: both --engine and --model must be provided."
  echo "Usage: $0 --engine=<tgi|vllm|mii|sglang|lmdeploy|llamacpp> --model=<your-org/your-model-name>"
  exit 1
fi

# ─── Common variables ───────────────────────────────────────────────────────
PORT=23333
CACHE_DIR="$HOME/.cache/huggingface"

# Ensure at least one token is set
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "Error: You must export HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment."
  exit 1
fi

# ─── Select and run the requested engine ────────────────────────────────────
case "$ENGINE" in

  # ──────────────────────────────────────────────────────────────
  # llama.cpp backend (2-step: convert HF -> GGUF, then server)
  # Example:
  #   ENGINE=llamacpp \
  #   MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2" \
  #   ./launch_engine.sh
  # ──────────────────────────────────────────────────────────────
  llamacpp)
    # ===== CONFIG =====
    MODEL_ID="${MODEL_ID:-${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}}"
    REVISION="${REVISION:-main}"

    # Where to store converted models
    OUT_DIR="${OUT_DIR:-$HOME/models}"

    # outtype is what convert-hf-to-gguf uses: f16, f32, q4_0, q4_k_m, q8_0, ...
    OUTTYPE="${OUTTYPE:-f16}"

    # llama.cpp server settings
    CTX="${CTX:-4096}"
    BATCH="${BATCH:-512}"
    N_GPU_LAYERS="${N_GPU_LAYERS:-99}"

    # Which server image to use (CPU vs CUDA)
    LLAMACPP_FULL_IMAGE="${LLAMACPP_FULL_IMAGE:-ghcr.io/ggerganov/llama.cpp:full}"
    LLAMACPP_SERVER_IMAGE="${LLAMACPP_SERVER_IMAGE:-ghcr.io/ggerganov/llama.cpp:server-cuda}"

    # ===== PREP =====
    command -v docker >/dev/null || { echo "docker not found"; exit 1; }
    command -v huggingface-cli >/dev/null || { echo "huggingface-cli not found (pip install -U huggingface-hub)"; exit 1; }

    SAFE_NAME="${MODEL_ID//\//__}"
    MODEL_ROOT="${OUT_DIR%/}/${SAFE_NAME}"
    HF_DIR="${MODEL_ROOT}/hf"
    mkdir -p "$HF_DIR"

    echo "[llamacpp] MODEL_ID=$MODEL_ID"
    echo "[llamacpp] REVISION=$REVISION"
    echo "[llamacpp] OUT_DIR=$OUT_DIR"
    echo "[llamacpp] OUTTYPE=$OUTTYPE"

    # Try to reuse an existing GGUF if present
    GGUF_FILE="$(compgen -G "$MODEL_ROOT"/*.gguf | head -n 1 || true)"

    if [ -z "$GGUF_FILE" ]; then
      echo "[llamacpp] No existing .gguf in $MODEL_ROOT – downloading HF model & converting"

      # ── Step 1: download HF model (safetensors etc.) ───────────────────────
      echo "[llamacpp] Downloading $MODEL_ID (rev=$REVISION) into $HF_DIR"
      huggingface-cli download \
        "$MODEL_ID" \
        --revision "$REVISION" \
        --local-dir "$HF_DIR" \
        --include "*" \
        --local-dir-use-symlinks False

      # ── Step 2: convert to GGUF using llama.cpp:full ──────────────────────
      echo "[llamacpp] Converting HF model to GGUF via $LLAMACPP_FULL_IMAGE (outtype=$OUTTYPE)"
      docker run --rm \
        -v "$HF_DIR":/repo \
        "$LLAMACPP_FULL_IMAGE" \
        --convert /repo \
        --outtype "$OUTTYPE"

      # The convert tool usually writes something like ggml-model-f16.gguf in /repo
      GGUF_IN_HF="$(ls -1 "$HF_DIR"/*.gguf 2>/dev/null | head -n 1)"
      if [ -z "$GGUF_IN_HF" ]; then
        echo "[llamacpp] ERROR: conversion did not produce a .gguf file in $HF_DIR"
        exit 1
      fi

      # Normalize filename in MODEL_ROOT
      TARGET_GGUF="$MODEL_ROOT/model-${OUTTYPE}.gguf"
      mv "$GGUF_IN_HF" "$TARGET_GGUF"
      GGUF_FILE="$TARGET_GGUF"
    fi

    echo "[llamacpp] Using GGUF file: $GGUF_FILE"

    # ===== RUN LLAMA.CPP SERVER (2nd container) =====
    echo "[llamacpp] Starting llama.cpp server from $LLAMACPP_SERVER_IMAGE on port $PORT"

    docker run --rm \
      --gpus all \
      -p "127.0.0.1:${PORT}:${PORT}" \
      -v "$MODEL_ROOT":/models \
      "$LLAMACPP_SERVER_IMAGE" \
        -m "/models/$(basename "$GGUF_FILE")" \
        -c "$CTX" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --n-gpu-layers "$N_GPU_LAYERS"
    ;;

  tgi)
    # ────────────────────────────────────────────────────────────────────────
    # TGI (Text Generation Inference) container:
    #
    # docker run --rm \
    #   --gpus all \
    #   -v "$HOME/.cache/huggingface:/data" \
    #   -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    #   -e HF_TOKEN="$HF_TOKEN" \
    #   -p 127.0.0.1:23333:23333 \
    #   ghcr.io/huggingface/text-generation-inference:3.3.1 \
    #     --model-id mistralai/Mistral-7B-Instruct-v0.3 \
    #     --trust-remote-code \
    #     --port 23333 \
    #     --max-client-batch-size 128
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --gpus all \
      -v "$HOME/.cache/huggingface:/data" \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HF_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      ghcr.io/huggingface/text-generation-inference:latest \
        --model-id "$MODEL" \
        --trust-remote-code \
        --port "$PORT" \
        --max-client-batch-size 512
    ;;

  vllm)
    # ────────────────────────────────────────────────────────────────────────
    # vLLM (OpenAI-compatible) container:
    #
    # docker run --rm \
    #   --runtime=nvidia --gpus all \
    #   -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    #   -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    #   -p 127.0.0.1:23333:23333 \
    #   --ipc=host \
    #   vllm/vllm-openai:latest \
    #     --model mistralai/Mistral-7B-Instruct-v0.3 \
    #     --port 23333

    #   
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --trust-remote-code \
        --max-model-len 4096 \
        --port "$PORT"
    ;;

  lmdeploy)
    # ────────────────────────────────────────────────────────────────────────
    # LMDeploy container:
    #
    # docker run --rm \
    #   --runtime=nvidia --gpus all \
    #   -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    #   -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    #   -p 127.0.0.1:23333:23333 \
    #   --ipc=host \
    #   openmmlab/lmdeploy:latest \
    #     lmdeploy serve api_server mistralai/Mistral-7B-Instruct-v0.3 \
    #     --server-port 23333
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      openmmlab/lmdeploy:latest \
        lmdeploy serve api_server "$MODEL" \
        --server-port "$PORT" \
        --cache-max-entry-count 0.5
    ;;

  sglang)
    # ────────────────────────────────────────────────────────────────────────
    # SGLang (Slim Graph Language) container:
    #
    # docker run --gpus all \
    #   -p 127.0.0.1:23333:23333 \
    #   -v ~/.cache/huggingface:/root/.cache/huggingface \
    #   --ipc=host \
    #   lmsysorg/sglang:latest \
    #   bash -c "\
    #     pip install --no-cache-dir protobuf sentencepiece --break-system-packages && \
    #     python3 -m sglang.launch_server \
    #       --model-path mistralai/Mistral-7B-Instruct-v0.3 \
    #       --host 0.0.0.0 \
    #       --port 23333 \
    #       --context-length 4096
    #   "
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --gpus all \
      -p 127.0.0.1:${PORT}:${PORT} \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      --ipc=host \
      -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
      lmsysorg/sglang:latest \
      bash -c "\
        pip install --no-cache-dir protobuf sentencepiece --break-system-packages && \
        python3 -m sglang.launch_server \
          --model-path $MODEL \
          --host 0.0.0.0 \
          --port $PORT \
        "
    ;;

  mii)
    # ────────────────────────────────────────────────────────────────────────
    # DeepSpeed-MII container:
    #
    # docker run --runtime=nvidia --gpus all \
    #   -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    #   -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    #   -p 127.0.0.1:23333:23333 \
    #   --ipc=host \
    #   slinusc/deepspeed-mii:latest \
    #   --model mistralai/Mistral-7B-Instruct-v0.3 \
    #   --port 23333
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      slinusc/deepspeed-mii:latest \
      --model "$MODEL" \
      --port "$PORT"
    ;;

  *)
    echo "Error: unsupported engine '$ENGINE'."
    echo "Please choose one of: tgi, vllm, lmdeploy, sglang, mii."
    exit 1
    ;;
esac
