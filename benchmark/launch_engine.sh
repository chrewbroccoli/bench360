#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_engine  —  Launch LLM inference engines via Docker
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Parse arguments ─────────────────────────────────────────────────────────
ENGINE=""
MODEL=""

for ARG in "$@"; do
  case "$ARG" in
    --engine=*) ENGINE="${ARG#--engine=}" ;;
    --model=*)  MODEL="${ARG#--model=}" ;;
    *)
      echo "Unknown argument: $ARG"
      echo "Usage: $0 --engine=<engine> --model=<model>"
      exit 1
      ;;
  esac
done

if [[ -z "$ENGINE" || -z "$MODEL" ]]; then
  echo "Error: both --engine and --model must be provided."
  exit 1
fi

# ─── Common variables ───────────────────────────────────────────────────────
PORT=23333
CACHE_DIR="$HOME/.cache/huggingface"
  OUT_DIR="${OUT_DIR:-$HOME/models}"
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

# ─── Engine Selection ───────────────────────────────────────────────────────
case "$ENGINE" in

llamacpp)
  # ─── Configuration ────────────────────────────────────────────────────────
  MODEL_ROOT="${OUT_DIR%/}/${MODEL//\//__}"
  HF_DIR="${MODEL_ROOT}/hf"

  LLAMACPP_SERVER_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"
  LLAMACPP_FULL_IMAGE="ghcr.io/ggml-org/llama.cpp:full"

  # ─── 0. Check for existing GGUF or HF weights ─────────────────────────────
  # Logic:
  # 1. Look for ANY .gguf file (e.g., custom names like "qwen-q6.gguf").
  # 2. If no GGUF is found, check for HF config.json.
  # 3. If neither exists, download from Hugging Face.

  # Find the first .gguf file in the directory (if any)
  EXISTING_GGUF=$(find "${MODEL_ROOT}" -maxdepth 1 -name "*.gguf" -print -quit)

  if [[ -z "${EXISTING_GGUF}" && ! -f "${HF_DIR}/config.json" ]]; then
    echo "[llamacpp] Neither HF weights nor any GGUF found in ${MODEL_ROOT}"
    echo "[llamacpp] Downloading '${MODEL}' via huggingface-cli..."

    # Ensure huggingface-cli exists on host
    if ! command -v huggingface-cli >/dev/null 2>&1; then
      echo "[llamacpp] Error: huggingface-cli not found on PATH."
      echo "Install it with: pip install -U 'huggingface_hub[cli]'"
      exit 1
    fi

    mkdir -p "${HF_DIR}"

    # Build auth args if token present
    HF_AUTH_ARGS=()
    if [[ -n "${HF_TOKEN}" ]]; then
      HF_AUTH_ARGS+=(--token "${HF_TOKEN}")
    fi

    # Optional: allow overriding revision (branch/tag/commit) via HF_REVISION
    HF_REVISION="${HF_REVISION:-}"
    HF_REV_ARGS=()
    if [[ -n "${HF_REVISION}" ]]; then
      HF_REV_ARGS+=(--revision "${HF_REVISION}")
    fi

    # Download the repo snapshot into HF_DIR
    huggingface-cli download "${MODEL}" \
      "${HF_AUTH_ARGS[@]}" \
      "${HF_REV_ARGS[@]}" \
      --local-dir "${HF_DIR}" \
      --local-dir-use-symlinks False

    echo "[llamacpp] Download complete."
  else
    echo "[llamacpp] Found existing HF weights OR a GGUF file. Skipping download."
  fi

  # ─── 1. Convert ONLY if we don't have a GGUF yet ──────────────────────────
  # Re-check for GGUF variable (in case we just downloaded and need to convert)
  EXISTING_GGUF=$(find "${MODEL_ROOT}" -maxdepth 1 -name "*.gguf" -print -quit)

  if [[ -z "${EXISTING_GGUF}" ]]; then
    echo "[llamacpp] No GGUF found. Converting HF weights to F16 GGUF..."

    docker run --rm \
      -v "$(realpath "$HF_DIR")":/input \
      -v "$(realpath "$MODEL_ROOT")":/output \
      "$LLAMACPP_FULL_IMAGE" \
      --convert \
      --outfile /output/model-f16.gguf \
      --outtype f16 \
      /input

    # Update variable to point to the file we just created
    EXISTING_GGUF="${MODEL_ROOT}/model-f16.gguf"
    echo "[llamacpp] F16 conversion complete."
  fi

  # ─── 2. Start Server ──────────────────────────────────────────────────────
  # We extract the filename so we can pass it to the docker container relative path
  GGUF_FILENAME=$(basename "$EXISTING_GGUF")

  echo "[llamacpp] Starting server on port $PORT..."
  echo "[llamacpp] Loading model: $GGUF_FILENAME"

  docker run --rm --gpus all \
    -p "127.0.0.1:${PORT}:${PORT}" \
    -v "$(realpath "$MODEL_ROOT")":/models \
    -v "$(pwd)/benchmark/grammars":/grammars \
    "$LLAMACPP_SERVER_IMAGE" \
    -m "/models/${GGUF_FILENAME}" \
    -c 32768 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    -ngl 99 \
    --flash-attn on \
    --no-mmap \
    --grammar-file /grammars/json.gbnf
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
        --max-model-len 16384 \
        --port "$PORT"
    ;;

  vllm-vl)
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
    # ───────────────────    docker run --rm \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --quantization fp8 \
        --trust-remote-code \
        --max-model-len 32000 \
        --port "$PORT" \
        --gpu-memory-utilization 0.96 \
        #--chat-template-content-format string─────────────────────────────────────────────────────

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
    exit 1
    ;;
esac