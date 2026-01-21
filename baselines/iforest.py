# baselines/iforest_timeseries.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def run_iforest(graphs, edge_labels, train_ratio=0.7):
    
    T = len(graphs)
    split = int(train_ratio * T)
    
    train_graphs = graphs[:split]
    
    test_graphs = graphs[split:]
    test_edge_labels = edge_labels[split:]
    
    # print(f"时间序列划分：训练集 {split} 个时间步，测试集 {T-split} 个时间步")
    
    def extract_features(g):
        src, dst = g.edges()
        deg = g.in_degrees().numpy()
       
        X_deg = np.stack([deg[src.numpy()], deg[dst.numpy()]], axis=1)
        return X_deg
    
    train_features = []
    for t in range(len(train_graphs)):
        X = extract_features(train_graphs[t])
        train_features.append(X)
    
    train_X = np.vstack(train_features)
    
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    
    clf = IsolationForest(
        n_estimators=10,           
        max_samples=16,            
        contamination=0.01,          
        random_state=123,
        n_jobs=-1
    )
    clf.fit(train_X_scaled)
    #print(f"训练完成，训练样本数: {train_X_scaled.shape[0]}")
    
    test_scores = []
    test_labels = []
    
    for t in range(len(test_graphs)):
        
        X_test = extract_features(test_graphs[t])
        X_test_scaled = scaler.transform(X_test)  
        
        scores = -clf.score_samples(X_test_scaled)
        
        test_scores.extend(scores)
        test_labels.extend(test_edge_labels[t].numpy())
    
    auc = roc_auc_score(test_labels, test_scores)
    # print(f"测试集AUC: {auc:.4f}")
    
    return auc

def run_iforest_cross_validation(graphs, edge_labels, n_splits=5):
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    auc_scores = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(graphs)):
        print(f"\n=== 第 {fold+1} 折交叉验证 ===")
        print(f"训练集大小: {len(train_index)}, 测试集大小: {len(test_index)}")
        
        assert max(train_index) < min(test_index), "时间序列顺序错误！"
        
        train_graphs = [graphs[i] for i in train_index]
        test_graphs = [graphs[i] for i in test_index]
        test_edge_labels = [edge_labels[i] for i in test_index]
        
        fold_auc = run_iforest(
            train_graphs + test_graphs, 
            [None] * len(train_graphs) + test_edge_labels,  
            train_ratio=len(train_graphs)/(len(train_graphs)+len(test_graphs))
        )
        auc_scores.append(fold_auc)
    
    print(f"\n平均AUC: {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.4f})")
    return np.mean(auc_scores)