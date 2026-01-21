import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from utils.train import evaluate_and_collect

# ===========================
# 1. 实验配置
# ===========================

SEED = 42

DATASET_NAME = "college_msg"
# 可选：
# "bitcoin_alpha"
# "bitcoin_otc"
# "college_msg"
# "wiki_talk"

DATA_PATHS = {
    "bitcoin_alpha": "data/bitcoin_alpha/soc-sign-bitcoin-alpha.csv",
    "bitcoin_otc": "data/bitcoin_otc/soc-sign-bitcoin-otc.csv",
    "college_msg": "data/email/CollegeMsg.txt",
    "wiki_talk": "data/wiki/wiki-talk-temporal.txt"
}

WINDOW_SIZE = 3          # 1 = GCN / GCN+T1, 3 = MSDGAD
USE_TEMPORAL = True      # False = GCN, True = Temporal
HIDDEN_DIM = 32

TIME_WINDOW_DAYS = 14
TRAIN_RATIO = 0.7
EPOCHS = 30
LR = 1e-3

RUN_BASELINES = False     # 是否运行 LOF / iForest / DynGEM

# ===========================
# 2. 固定随机种子
# ===========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ===========================
# 3. 数据加载
# ===========================

def load_dataset():
    if DATASET_NAME == "bitcoin_alpha":
        from utils.preprocess import load_bitcoin_alpha
        return load_bitcoin_alpha(
            DATA_PATHS["bitcoin_alpha"],
            time_window=TIME_WINDOW_DAYS * 24 * 3600
        )

    elif DATASET_NAME == "bitcoin_otc":
        from utils.preprocess import load_bitcoin_alpha
        return load_bitcoin_alpha(
            DATA_PATHS["bitcoin_otc"],
            time_window=TIME_WINDOW_DAYS * 24 * 3600
        )

    elif DATASET_NAME == "college_msg":
        from utils.preprocess_email import load_college_msg
        return load_college_msg(
            DATA_PATHS["college_msg"],
            time_window=TIME_WINDOW_DAYS * 24 * 3600
        )
    elif DATASET_NAME == "wiki_talk":
        from utils.preprocess_wiki_talk import load_wiki_talk
        return load_wiki_talk(
            DATA_PATHS["wiki_talk"],
            time_window=TIME_WINDOW_DAYS * 24 * 3600
        )
    else:
        raise ValueError(f"Unknown dataset {DATASET_NAME}")

# ===========================
# 4. 主流程
# ===========================

def main():
    print("=" * 70)
    print(f"Dataset      : {DATASET_NAME}")
    print(f"Window size  : {WINDOW_SIZE}")
    print(f"Use temporal : {USE_TEMPORAL}")
    print(f"Hidden dim   : {HIDDEN_DIM}")
    print("=" * 70)

    # ---------- load data ----------
    graphs, edge_labels = load_dataset()

    from utils.dataset import DynamicGraphDataset
    dataset = DynamicGraphDataset(
        graphs,
        edge_labels,
        window_size=WINDOW_SIZE
    )

    T = len(dataset)
    train_len = int(TRAIN_RATIO * T)

    train_set = Subset(dataset, range(train_len))
    test_set = Subset(dataset, range(train_len, T))

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0]
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0]
    )

    print(f"Total snapshots : {len(graphs)}")
    print(f"Train samples  : {len(train_set)}")
    print(f"Test samples   : {len(test_set)}")

    # ---------- build model ----------
    from models.msdgad import MSDGAD
    
    in_dim = graphs[0].ndata["feat"].shape[1]
    
    model = MSDGAD(
        in_dim=in_dim,
        hid_dim=HIDDEN_DIM,
        use_temporal=USE_TEMPORAL
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---------- training ----------
    from utils.train import train_epoch, evaluate
    test_auc_list = []
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer)
        test_auc = evaluate(model, test_loader)
        test_auc_list.append(test_auc)
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Test AUC {test_auc:.4f}"
        )
    np.savez(f'training_curve_{DATASET_NAME}.npz',test_auc=test_auc_list)
    
    # ---------- evaluation ----------
    print("=" * 70)
    print("Model training finished.")
    print("=" * 70)
    
    # ---------- score distribution ----------
    evaluate_and_collect(
        model,
        test_loader,
        save_path=f'score_distribution_{DATASET_NAME}.npz',
        max_samples=20000
    )
    
    # ===========================
    # 5. Baselines（完全独立）
    # ===========================

    if RUN_BASELINES:
        print("\nRunning baselines...")
        print("-" * 70)

        from baselines.lof import run_lof
        from baselines.iforest import run_iforest
        from baselines.dyn_gem import run_dyn_gem

        lof_auc = run_lof(graphs, edge_labels, train_ratio=TRAIN_RATIO)
        print(f"LOF AUC      : {lof_auc:.4f}")

        if_auc = run_iforest(graphs, edge_labels, train_ratio=TRAIN_RATIO)
        print(f"IForest AUC  : {if_auc:.4f}")

        dyn_auc = run_dyn_gem(graphs, edge_labels, train_ratio=TRAIN_RATIO)
        print(f"DynGEM AUC   : {dyn_auc:.4f}")

        print("-" * 70)
        print("Baselines finished.")

    print("=" * 70)
    print("All experiments completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
