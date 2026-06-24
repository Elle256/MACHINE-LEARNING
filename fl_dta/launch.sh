#!/usr/bin/env bash
# =============================================================================
# launch.sh  –  FL-DTA full experiment pipeline
#
# Thứ tự chạy:
#   1. Cài dependencies
#   2. Centralized baseline (upper bound)
#   3. FedAvg IID
#   4. FedAvg Non-IID (kinase-based)
#   5. FedAvg Non-IID Dirichlet (alpha=0.5 và alpha=0.1)
#   6. Phân tích & vẽ biểu đồ
#
# Usage:
#   chmod +x launch.sh
#   ./launch.sh                        # chạy tất cả trên davis
#   ./launch.sh --dataset kiba         # chạy trên kiba
#   ./launch.sh --skip_install         # bỏ qua bước pip install
#   ./launch.sh --gpu 0                # chỉ định GPU
# =============================================================================

set -euo pipefail   # dung ngay neu co loi
# Force UTF-8 trên Windows (tránh cp1252 encoding error)
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8


# ─── Parse flags ─────────────────────────────────────────────────────────────
DATASET="davis"
GPU="0"
SKIP_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)    DATASET="$2";      shift 2 ;;
        --gpu)        GPU="$2";          shift 2 ;;
        --skip_install) SKIP_INSTALL=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
FIGURES_DIR="$SCRIPT_DIR/figures"
LOG_DIR="$SCRIPT_DIR/logs"
DATA_DIR="$SCRIPT_DIR/data/raw"
CONFIG="$SCRIPT_DIR/configs/default.yaml"

mkdir -p "$RESULTS_DIR" "$FIGURES_DIR" "$LOG_DIR"

# ─── Logging helper ──────────────────────────────────────────────────────────
log() { echo -e "\n\033[1;36m[$(date '+%H:%M:%S')] $*\033[0m"; }
ok()  { echo -e "\033[1;32m✓ $*\033[0m"; }
err() { echo -e "\033[1;31m✗ $*\033[0m" >&2; exit 1; }

# ─── Detect device ───────────────────────────────────────────────────────────
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    log "GPU detected → sử dụng CUDA (GPU $GPU)"
else
    DEVICE="cpu"
    log "Không tìm thấy GPU → sử dụng CPU"
fi

# ─── 0. Install dependencies ─────────────────────────────────────────────────
if [ "$SKIP_INSTALL" = false ]; then
    log "Cài đặt dependencies..."

    # torch-geometric cần cài với đúng wheel URL theo torch version + cuda tag
    TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
    CUDA_TAG=$(python -c "import torch; v=torch.version.cuda; print('cu'+''.join(v.split('.')) if v else 'cpu')" 2>/dev/null || echo "cpu")

    if [ -n "$TORCH_VER" ]; then
        log "Torch $TORCH_VER | CUDA tag: $CUDA_TAG"
        PYGE_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
        pip install -q torch-scatter torch-sparse torch-geometric -f "$PYGE_URL" \
            || pip install -q torch-geometric   # fallback nếu wheel không match
    else
        pip install -q torch-geometric
    fi

    # Các package còn lại (rdkit đã đổi tên từ rdkit-pypi)
    pip install -q \
        rdkit \
        "numpy>=1.24.0" \
        "pandas>=2.0.0" \
        "scikit-learn>=1.2.0" \
        "pyyaml>=6.0" \
        "matplotlib>=3.7.0" \
        "seaborn>=0.12.0" \
        || err "pip install thất bại"

    ok "Dependencies đã cài xong"
fi

# ─── Common hyperparameters ──────────────────────────────────────────────────
# (Được chọn theo chuẩn DeepDTA paper + FL best practices)

SEED=42
BATCH_SIZE=512         # centralized batch size
NUM_CLIENTS=5          # số pharma clients (5 = mô phỏng thực tế hợp lý)
ROUNDS=100             # số FL communication rounds
LOCAL_EPOCHS=5         # số epochs local mỗi round (đủ để hội tụ, không overfit)
LOCAL_BS=256           # local batch size
LR=0.001               # Adam lr (paper DeepDTA dùng 0.001)
WEIGHT_DECAY=1e-4
FRACTION_FIT=1.0       # tất cả clients tham gia mỗi round (N=5, nhỏ nên dùng full)
LOG_EVERY=10           # log mỗi 10 rounds

# ─── 1. Centralized baseline ─────────────────────────────────────────────────
log "Bắt đầu [1/5]: Centralized baseline (${DATASET})..."

python "$SCRIPT_DIR/train_centralized.py" \
    --dataset        "$DATASET"       \
    --epochs         100              \
    --batch_size     "$BATCH_SIZE"    \
    --lr             "$LR"            \
    --weight_decay   "$WEIGHT_DECAY"  \
    --seed           "$SEED"          \
    --device         "$DEVICE"        \
    --data_dir       "$DATA_DIR"      \
    --results_dir    "$RESULTS_DIR"   \
    --config         "$CONFIG"        \
    --log_every      5                \
    2>&1 | tee "$LOG_DIR/centralized_${DATASET}.log"

ok "Centralized xong."

# ─── 2. FedAvg – IID ─────────────────────────────────────────────────────────
log "Bắt đầu [2/5]: FedAvg IID (${DATASET}, ${NUM_CLIENTS} clients)..."

python "$SCRIPT_DIR/train_federated.py" \
    --dataset        "$DATASET"       \
    --partition      iid              \
    --num_clients    "$NUM_CLIENTS"   \
    --rounds         "$ROUNDS"        \
    --local_epochs   "$LOCAL_EPOCHS"  \
    --local_bs       "$LOCAL_BS"      \
    --lr             "$LR"            \
    --weight_decay   "$WEIGHT_DECAY"  \
    --fraction_fit   "$FRACTION_FIT"  \
    --batch_size     "$BATCH_SIZE"    \
    --seed           "$SEED"          \
    --device         "$DEVICE"        \
    --data_dir       "$DATA_DIR"      \
    --results_dir    "$RESULTS_DIR"   \
    --config         "$CONFIG"        \
    --log_every      "$LOG_EVERY"     \
    2>&1 | tee "$LOG_DIR/fedavg_iid_${DATASET}.log"

ok "FedAvg IID xong."

# ─── 3. FedAvg – Non-IID kinase-based ────────────────────────────────────────
log "Bắt đầu [3/5]: FedAvg Non-IID kinase (${DATASET}, ${NUM_CLIENTS} clients)..."

python "$SCRIPT_DIR/train_federated.py" \
    --dataset        "$DATASET"       \
    --partition      non_iid          \
    --num_clients    "$NUM_CLIENTS"   \
    --rounds         "$ROUNDS"        \
    --local_epochs   "$LOCAL_EPOCHS"  \
    --local_bs       "$LOCAL_BS"      \
    --lr             "$LR"            \
    --weight_decay   "$WEIGHT_DECAY"  \
    --fraction_fit   "$FRACTION_FIT"  \
    --batch_size     "$BATCH_SIZE"    \
    --seed           "$SEED"          \
    --device         "$DEVICE"        \
    --data_dir       "$DATA_DIR"      \
    --results_dir    "$RESULTS_DIR"   \
    --config         "$CONFIG"        \
    --log_every      "$LOG_EVERY"     \
    2>&1 | tee "$LOG_DIR/fedavg_noniid_${DATASET}.log"

ok "FedAvg Non-IID kinase xong."

# ─── 4. FedAvg – Non-IID Dirichlet alpha=0.5 (mild heterogeneity) ────────────
log "Bắt đầu [4/5]: FedAvg Dirichlet α=0.5 (${DATASET})..."

python "$SCRIPT_DIR/train_federated.py" \
    --dataset        "$DATASET"       \
    --partition      dirichlet        \
    --alpha          0.5              \
    --num_clients    "$NUM_CLIENTS"   \
    --rounds         "$ROUNDS"        \
    --local_epochs   "$LOCAL_EPOCHS"  \
    --local_bs       "$LOCAL_BS"      \
    --lr             "$LR"            \
    --weight_decay   "$WEIGHT_DECAY"  \
    --fraction_fit   "$FRACTION_FIT"  \
    --batch_size     "$BATCH_SIZE"    \
    --seed           "$SEED"          \
    --device         "$DEVICE"        \
    --data_dir       "$DATA_DIR"      \
    --results_dir    "$RESULTS_DIR"   \
    --config         "$CONFIG"        \
    --log_every      "$LOG_EVERY"     \
    2>&1 | tee "$LOG_DIR/fedavg_dirichlet05_${DATASET}.log"

ok "FedAvg Dirichlet α=0.5 xong."

# ─── 5. FedAvg – Non-IID Dirichlet alpha=0.1 (severe heterogeneity) ──────────
log "Bắt đầu [5/5]: FedAvg Dirichlet α=0.1 (${DATASET}, severe Non-IID)..."

python "$SCRIPT_DIR/train_federated.py" \
    --dataset        "$DATASET"       \
    --partition      dirichlet        \
    --alpha          0.1              \
    --num_clients    "$NUM_CLIENTS"   \
    --rounds         "$ROUNDS"        \
    --local_epochs   "$LOCAL_EPOCHS"  \
    --local_bs       "$LOCAL_BS"      \
    --lr             "$LR"            \
    --weight_decay   "$WEIGHT_DECAY"  \
    --fraction_fit   "$FRACTION_FIT"  \
    --batch_size     "$BATCH_SIZE"    \
    --seed           "$SEED"          \
    --device         "$DEVICE"        \
    --data_dir       "$DATA_DIR"      \
    --results_dir    "$RESULTS_DIR"   \
    --config         "$CONFIG"        \
    --log_every      "$LOG_EVERY"     \
    2>&1 | tee "$LOG_DIR/fedavg_dirichlet01_${DATASET}.log"

ok "FedAvg Dirichlet α=0.1 xong."

# ─── 6. Phân tích & vẽ biểu đồ ──────────────────────────────────────────────
log "Phân tích kết quả và vẽ biểu đồ..."

python "$SCRIPT_DIR/analyze_results.py" \
    --results_dir "$RESULTS_DIR"        \
    --output_dir  "$FIGURES_DIR"        \
    2>&1 | tee "$LOG_DIR/analyze.log"

ok "Biểu đồ đã lưu tại: $FIGURES_DIR"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              FL-DTA – Tất cả experiments XONG           ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Dataset    : %-43s║\n" "$DATASET"
printf "║  Device     : %-43s║\n" "$DEVICE"
printf "║  Clients    : %-43s║\n" "$NUM_CLIENTS"
printf "║  Rounds     : %-43s║\n" "$ROUNDS"
printf "║  Results    : %-43s║\n" "$RESULTS_DIR"
printf "║  Figures    : %-43s║\n" "$FIGURES_DIR"
printf "║  Logs       : %-43s║\n" "$LOG_DIR"
echo "╚══════════════════════════════════════════════════════════╝"
