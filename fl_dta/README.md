# FL-DTA: Federated Learning for Drug-Target Affinity Prediction

## Tổng quan

Project này benchmark **Federated Learning (FL)** trên bài toán **Drug-Target Affinity (DTA)** prediction, sử dụng dataset Davis/KIBA. Mỗi client mô phỏng một tổ chức dược phẩm độc lập, train cục bộ và aggregate bằng FedAvg.

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
pip install torch torch-geometric torch-scatter torch-sparse
pip install rdkit-pypi pandas numpy scikit-learn pyyaml wandb matplotlib seaborn