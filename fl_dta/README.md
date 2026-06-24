# FL-DTA: Federated Learning for Drug-Target Affinity Prediction

## Tổng quan

Project này benchmark **Federated Learning (FL)** trên bài toán **Drug-Target Affinity (DTA)** prediction, sử dụng dataset Davis/KIBA. Mỗi client mô phỏng một tổ chức dược phẩm độc lập, train cục bộ và aggregate bằng FedAvg hoặc FedProx.

## Câu hỏi nghiên cứu

1. **FL vs Centralized**: FL có giữ được performance so với centralized training không?
2. **Non-IID impact**: Non-IID data (mỗi lab chuyên một loại kinase) ảnh hưởng thế nào?

## Đóng góp

- Benchmark FL đầu tiên trên bài toán DTA
- Phân tích communication rounds vs accuracy trade-off
- So sánh IID vs Non-IID partition strategy
- So sánh FedAvg vs FedProx trên Non-IID data

## Cấu trúc project

```
fl_dta/
├── configs/
│   └── default.yaml          # Hyperparameters toàn bộ experiment
├── data/
│   ├── dataset.py            # Davis/KIBA loader & preprocessing
│   └── partition.py          # IID & Non-IID partitioning
├── models/
│   ├── drug_encoder.py       # GNN encoder cho drug (SMILES → graph)
│   ├── protein_encoder.py    # CNN encoder cho protein sequence
│   └── deep_dta.py           # DeepDTA model tổng hợp
├── federation/
│   ├── server.py             # FL Server: aggregate, evaluate
│   ├── client.py             # FL Client: LocalUpdate (FedAvg & FedProx)
│   └── aggregation.py        # FedAvg weighted aggregation
├── utils/
│   ├── metrics.py            # MSE, CI, rm2 metrics
│   ├── logger.py             # WandB + CSV logging
│   └── seed.py               # Reproducibility
├── experiments/
│   └── run_experiment.py     # Entry point chạy experiment
├── train_centralized.py      # Baseline centralized training
├── train_federated.py        # Federated training loop
└── analyze_results.py        # Plot & so sánh kết quả
```

## Cài đặt

```bash
pip install torch torch-geometric torch-scatter torch-sparse
pip install rdkit-pypi pandas numpy scikit-learn pyyaml wandb matplotlib seaborn
```

## Chạy experiment

```bash
# Centralized baseline
python train_centralized.py --dataset davis --epochs 100

# Federated (IID)
python train_federated.py --dataset davis --algorithm fedavg --partition iid --num_clients 5 --rounds 100

# Federated (Non-IID)
python train_federated.py --dataset davis --algorithm fedavg --partition non_iid --num_clients 5 --rounds 100

# FedProx (Non-IID)
python train_federated.py --dataset davis --algorithm fedprox --partition non_iid --mu 0.01 --num_clients 5 --rounds 100

# Phân tích kết quả
python analyze_results.py --results_dir results/
```
