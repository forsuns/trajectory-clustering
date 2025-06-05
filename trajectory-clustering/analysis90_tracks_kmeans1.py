import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import json

# -------------------- 설정 --------------------
data_dir = './data'
plot_dir = './results90'
os.makedirs(plot_dir, exist_ok=True)

case = '0'
case_str = str(case).zfill(1)
track_file = f"{data_dir}/track{case_str}.txt"

# -------------------- 데이터 로딩 --------------------
track_cols = ['time', 'n', 'x', 'y', 'z', 'ip']
track_df = pd.read_csv(track_file, delim_whitespace=True, header=None, names=track_cols, dtype=str)
if track_df.iloc[0]['x'].lower().strip() == 'x':
    track_df = track_df.iloc[1:]

track_df['x'] = pd.to_numeric(track_df['x'], errors='coerce')
track_df['y'] = pd.to_numeric(track_df['y'], errors='coerce')
track_df['time'] = pd.to_numeric(track_df['time'], errors='coerce')
track_df['n'] = pd.to_numeric(track_df['n'], errors='coerce')
track_df = track_df[(track_df['x'] != 0) & (track_df['y'] != 0)].dropna(subset=['x', 'y'])

# -------------------- 궤적 통계량 --------------------
traj_stats = []
for pid, group in track_df.groupby('n'):
    group = group.sort_values('time')
    path = group[['x', 'y']].values
    if len(path) < 5:
        continue
    start, end = path[0], path[-1]
    dist = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    direct = euclidean(start, end)
    straightness = direct / dist if dist else 0
    displacement_ratio = direct / dist if dist else 0
    duration = group['time'].max() - group['time'].min()
    centroid = path.mean(axis=0)
    rad_avg = np.mean(np.linalg.norm(path - centroid, axis=1))
    vecs = np.diff(path, axis=0)
    angles = [np.arccos(np.clip(np.dot(vecs[i-1]/np.linalg.norm(vecs[i-1]), vecs[i]/np.linalg.norm(vecs[i])), -1, 1))
              for i in range(1, len(vecs)) if np.linalg.norm(vecs[i-1]) > 0 and np.linalg.norm(vecs[i]) > 0]
    curvature = np.mean(angles) if angles else 0
    traj_stats.append([pid, centroid[0], centroid[1], dist, direct, straightness, displacement_ratio, duration, rad_avg, curvature])

traj_stats_df = pd.DataFrame(traj_stats, columns=['id', 'x', 'y', 'dist', 'direct', 'straightness', 'displacement_ratio', 'duration', 'rad_avg', 'curvature'])
feature_cols = ['dist', 'displacement_ratio', 'duration', 'rad_avg', 'curvature']
traj_stats_df = traj_stats_df.dropna(subset=feature_cols)

# -------------------- 클러스터링 --------------------
scaler = StandardScaler()
features_scaled = scaler.fit_transform(traj_stats_df[feature_cols])
kmeans = KMeans(n_clusters=3, random_state=4002).fit(features_scaled)
traj_stats_df['label'] = kmeans.labels_

# -------------------- 클러스터 대표값 --------------------
cluster_means = traj_stats_df.groupby('label')[feature_cols].mean()
cluster_means.to_csv(f"{plot_dir}/case{case_str}_cluster_feature_means.txt", sep='\t')

# -------------------- 유형 정의 (중복 방지 순차 지정) --------------------
type_map = {}
used = set()

# Type A: 체류시간이 가장 짧은 클러스터
a = cluster_means['duration'].idxmin()
type_map[a] = 'Type A (Advective)'
used.add(a)

# Type B: 이동거리가 가장 큰 클러스터 중 미사용
b = cluster_means.loc[~cluster_means.index.isin(used), 'dist'].idxmax()
type_map[b] = 'Type B (Loop)'
used.add(b)

# Type C: 중심 반경이 가장 큰 클러스터 중 미사용
c = cluster_means.loc[~cluster_means.index.isin(used), 'rad_avg'].idxmax()
type_map[c] = 'Type C (Spreading)'

traj_stats_df['type'] = traj_stats_df['label'].map(type_map)

# 저장용 역매핑 기준
with open(f"{plot_dir}/case{case_str}_cluster_type_mapping.json", 'w') as f:
    json.dump({int(k): v for k, v in type_map.items()}, f, indent=2)

# -------------------- PCA 시각화 --------------------
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(features_scaled)
traj_stats_df['PCA1'] = pca_coords[:, 0]
traj_stats_df['PCA2'] = pca_coords[:, 1]

# ▶ loading 저장
loading_matrix = pd.DataFrame(
    pca.components_.T,
    index=feature_cols,
    columns=['PCA1_loading', 'PCA2_loading']
)
loading_matrix.to_csv(f"{plot_dir}/case{case_str}_pca_loadings.txt", sep='\t')

# ▶ PCA 시각화
fig, ax = plt.subplots(figsize=(10, 8))
colors = {'Type A (Advective)': 'red', 'Type B (Loop)': 'blue', 'Type C (Spreading)': 'green'}
for ttype, tgroup in traj_stats_df.groupby('type'):
    ax.scatter(tgroup['PCA1'], tgroup['PCA2'], label=ttype, s=30, alpha=0.7, color=colors.get(ttype, 'gray'))
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title(f"Case {case_str} - PCA by Cluster & Type")
ax.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/case{case_str}_pca_cluster_type.png")
plt.close()

# -------------------- 결과 저장 --------------------
columns_order = ['id', 'x', 'y', 'label', 'type'] + feature_cols + ['PCA1', 'PCA2']
traj_stats_df.to_csv(f"{plot_dir}/case{case_str}_trajectory_with_type.txt",
                     sep='\t', columns=columns_order, index=False, encoding='utf-8')

print("✅ case{case_str} 클러스터링, 유형 분류 및 PCA loading 저장 완료!")
