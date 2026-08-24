#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-$BASE/LLM-Ranker-Attack}"
ENV="${ENV:-$PROJECT/.conda-bedrock}"
PYTHON="${PYTHON:-$ENV/bin/python}"
IR_DATASETS_HOME="${IR_DATASETS_HOME:-$BASE/.cache/ir_datasets}"
COLLECTION_FILE="${MSMARCO_COLLECTION_FILE:-$IR_DATASETS_HOME/msmarco-passage/collection.tsv}"
INPUT_DIR="${MSMARCO_INDEX_INPUT:-$BASE/index-input/msmarco-passage-jsonl}"
INPUT_MARKER="${INPUT_DIR}.complete"
INDEX_PATH="${MSMARCO_INDEX:-$BASE/indexes/msmarco-v1-passage}"
THREADS="${THREADS:-16}"

export JAVA_HOME="${JAVA_HOME:-$ENV/lib/jvm}"
export JVM_PATH="${JVM_PATH:-$JAVA_HOME/lib/server/libjvm.so}"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$JVM_PATH" ]]; then
  echo "Java runtime library not found: $JVM_PATH" >&2
  exit 1
fi
if [[ ! -f "$COLLECTION_FILE" ]]; then
  echo "MS MARCO collection not found: $COLLECTION_FILE" >&2
  echo "Set MSMARCO_COLLECTION_FILE to the extracted collection.tsv path." >&2
  exit 1
fi
if [[ -e "$INDEX_PATH" ]]; then
  echo "Index path already exists; refusing to overwrite: $INDEX_PATH" >&2
  exit 1
fi

if [[ ! -f "$INPUT_MARKER" ]]; then
  echo "Converting collection TSV to Anserini JSONL shards..."
  "$PYTHON" "$PROJECT/retrieval/convert_msmarco_tsv_to_jsonl.py" \
    --input "$COLLECTION_FILE" \
    --output-dir "$INPUT_DIR"
fi

mkdir -p "$(dirname "$INDEX_PATH")"

echo "Collection: $COLLECTION_FILE"
echo "Input dir:  $INPUT_DIR"
echo "Index:      $INDEX_PATH"
echo "Threads:    $THREADS"

"$PYTHON" -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$INPUT_DIR" \
  --index "$INDEX_PATH" \
  --generator DefaultLuceneDocumentGenerator \
  --threads "$THREADS"

echo "Index completed: $INDEX_PATH"
