import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

from matplotlib import rcParams
from scipy.io import loadmat
from matplotlib.patheffects import withStroke

rcParams['font.size'] = 18

xlim = (260000, 300000)
ylim = (3944000, 3984000)

def get_case_basemap(tag, basemap_data):
    index = {'0': 0, '1': 1, '2': 2}.get(tag, 0)
    xx = basemap_data[f'xx{index}']
    yy = basemap_data[f'yy{index}']
    return xx, yy

def add_basemap_overlay(ax, xx, yy, island_data, xlim, ylim, north=True):
    ax.fill(xx, yy, color=[1, 0.88, 0.21], edgecolor='black', linewidth=0.5, zorder=2)
    for _, row in island_data.iterrows():
        if xlim[0] <= row['inX'] <= xlim[1] and ylim[0] <= row['inY'] <= ylim[1]:
            ax.text(row['inX'], row['inY'], row['inames'],
                    fontsize=18, color='black', ha='center', va='center',
                    path_effects=[withStroke(linewidth=3, foreground='white')])
    if north:
        add_north_arrow(ax, x=0.95, y=0.95, width=0.04, height=0.05, fontsize=14)

def add_north_arrow(ax, x=0.9, y=0.9, width=0.02, height=0.1, label="N", fontsize=14,
                    color='black', edgecolor='black'):
    shadow_effect = withStroke(linewidth=3, foreground="gray", alpha=0.6)
    ax.annotate('', xy=(x, y), xytext=(x, y - height),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor=color, edgecolor=edgecolor,
                                linewidth=0.5, width=width * 100,
                                headwidth=width * 300, headlength=height * 200))
    ax.text(x, y + height * 0.2, label, fontsize=fontsize, color=color,
            ha='center', va='center', transform=ax.transAxes, path_effects=[shadow_effect])

def add_scale_bar(ax, length=1000, location=(0.1, 0.05), linewidth=3, fontsize=12):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bar_x = xlim[0] + (xlim[1] - xlim[0]) * location[0]
    bar_y = ylim[0] + (ylim[1] - ylim[0]) * location[1]
    ax.plot([bar_x, bar_x + length], [bar_y, bar_y], color='k', lw=linewidth)
    ax.text(bar_x + length / 2, bar_y + (ylim[1] - ylim[0]) * 0.01,
            f'{length/1000:.0f} km' if length >= 1000 else f'{length:.0f} m',
            ha='center', va='bottom', fontsize=fontsize)

# ⬇️ island_data 로딩
island_data = pd.read_csv('./islandNames.txt', delim_whitespace=True, header=None,
                          names=["inX", "inY", "inames"])

mat_data = loadmat(f'./data/basemap.mat')

##############################################################################

# -------------------- 설정 --------------------
data_dir = './data'
plot_dir = './results90'
os.makedirs(plot_dir, exist_ok=True)

cases = ['0', '1', '2']  # case0도 포함하여 도면 일관성 유지
feature_cols = ['dist', 'displacement_ratio', 'duration', 'rad_avg', 'curvature']

# -------------------- 기준 로딩 --------------------
with open(f"{plot_dir}/case0_cluster_type_mapping.json", 'r') as f:
    label_to_type = json.load(f)
label_to_type = {int(k): v for k, v in label_to_type.items()}

cluster_means = pd.read_csv(f"{plot_dir}/case0_cluster_feature_means.txt", sep='\t', index_col=0)
cluster_labels = cluster_means.index.values

# -------------------- 함수: 통계량 계산 --------------------
def compute_trajectory_stats(track_df):
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
        traj_stats.append([pid, centroid[0], centroid[1], dist, direct, displacement_ratio, duration, rad_avg, curvature])
    df = pd.DataFrame(traj_stats, columns=['id', 'x', 'y', 'dist', 'direct', 'displacement_ratio', 'duration', 'rad_avg', 'curvature'])
    return df

# -------------------- 각 case별 적용 --------------------
for case in cases:
    xx, yy = get_case_basemap(case, mat_data)
    case_str = str(case).zfill(1)
    track_file = f"{data_dir}/track{case_str}.txt"
    track_cols = ['time', 'n', 'x', 'y', 'z', 'ip']
    track_df = pd.read_csv(track_file, delim_whitespace=True, header=None, names=track_cols, dtype=str)
    if track_df.iloc[0]['x'].lower().strip() == 'x':
        track_df = track_df.iloc[1:]
    track_df['x'] = pd.to_numeric(track_df['x'], errors='coerce')
    track_df['y'] = pd.to_numeric(track_df['y'], errors='coerce')
    track_df['time'] = pd.to_numeric(track_df['time'], errors='coerce')
    track_df['n'] = pd.to_numeric(track_df['n'], errors='coerce')
    track_df = track_df[(track_df['x'] != 0) & (track_df['y'] != 0)].dropna(subset=['x', 'y'])

    stats_df = compute_trajectory_stats(track_df)

    if stats_df.empty:
        print(f"⚠️ case{case_str}의 유효한 궤적 데이터가 없습니다.")
        continue

    scaler = StandardScaler()
    stats_scaled = scaler.fit_transform(stats_df[feature_cols])
    cluster_scaled = scaler.transform(cluster_means[feature_cols])

    labels = []
    for row in stats_scaled:
        dists = np.linalg.norm(cluster_scaled - row, axis=1)
        closest = cluster_means.index[np.argmin(dists)]
        labels.append(int(closest))

    stats_df['label'] = labels
    stats_df['type'] = stats_df['label'].map(label_to_type)

    if stats_df['type'].isna().all():
        print(f"⚠️ case{case_str} 유형 매핑 실패: label_to_type 매핑을 확인하세요.")
        continue

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(stats_scaled)
    stats_df['PCA1'] = pca_coords[:, 0]
    stats_df['PCA2'] = pca_coords[:, 1]

    stats_df.to_csv(f"{plot_dir}/case{case_str}_trajectory_with_type.txt", sep='\t', index=False)

    import matplotlib.cm as cm
    import seaborn as sns
    color_map = dict(zip(sorted(stats_df['type'].unique()), cm.get_cmap('tab10').colors))

    # ▶ 1. PCA 시각화
    fig, ax = plt.subplots(figsize=(10, 8))
    for ttype, tgroup in stats_df.groupby('type'):
        ax.scatter(tgroup['PCA1'], tgroup['PCA2'], label=ttype, s=30, alpha=0.6, color=color_map.get(ttype, 'gray'))
    ax.set_title(f"Case {case_str} - PCA by Assigned Type (from Case 0)")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/case{case_str}_type_pca_scatter.png")
    plt.close()

    # ▶ 2. 유형별 개수 바차트
    fixed_order = ['Type A (Advective)', 'Type B (Loop)', 'Type C (Spreading)']
    type_counts = stats_df['type'].value_counts().reindex(fixed_order, fill_value=0)
    type_percents = type_counts / type_counts.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = type_counts.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title(f"Case {case_str} - Count of Trajectory Types")
    ax.set_xlabel("Trajectory Type")
    ax.set_ylabel("Number of Particles")
    plt.xticks(rotation=45)

    for i, (count, percent) in enumerate(zip(type_counts, type_percents)):
        ax.text(i, count + max(type_counts) * 0.01, f"{percent:.1f}%", ha='center', va='bottom', fontsize=18)

    plt.ylim([0,1000])
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/case{case_str}_type_count_bar.png")
    plt.close()

    print(f"✅ case{case_str} 유형 분류 및 도면 생성 완료 (기준: case0)")
    
    
    # ▶ 대표 궤적 시각화 (유형별 중심값 기반 - 모든 case)
    from matplotlib.patheffects import withStroke

    label_colors = {
        'Type A (Advective)': 'red',
        'Type B (Loop)': 'blue',
        'Type C (Spreading)': 'green'
        }

    fig, ax = plt.subplots(figsize=(10, 8))
    
    ########### base map start ###########
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_autoscale_on(False)
    add_basemap_overlay(ax, xx, yy, island_data, xlim, ylim)
    add_scale_bar(ax, length=1000, location=(0.1, 0.05))
    ########### base map end  ###########
    
    #ax.set_aspect('equal')
    ax.set_title(f"Case {case_str} - Representative Trajectories by Type")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    for ttype, tgroup in stats_df.groupby('type'):
        cluster_mean = tgroup[feature_cols].mean().values
        tgroup['center_dist'] = np.linalg.norm(tgroup[feature_cols].values - cluster_mean, axis=1)
        reps = tgroup.nsmallest(5, 'center_dist')
        for pid in reps['id'].astype(int):
            traj = track_df[track_df['n'] == pid].sort_values('time')
            ax.plot(traj['x'], traj['y'], color=label_colors.get(ttype, 'gray'), linewidth=1.5,
                    alpha=0.7, label=ttype if pid == reps['id'].iloc[0] else None)

    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/case{case_str}_representative_paths.png")
    plt.close()
    
    # ▶ 유형별 설명변수 평균 비교 테이블 저장 및 시각화
    type_feature_summary = stats_df.groupby('type')[feature_cols].mean()
    type_feature_summary = type_feature_summary.reindex(fixed_order)  # 순서 고정
    table_path = f"{plot_dir}/case{case_str}_type_feature_means.txt"
    type_feature_summary.to_csv(table_path, sep='\t', float_format='%.4f')

    # ▶ 시각화 (바그래프 형태)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        type_feature_summary.T,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        annot_kws={"size": 14},  # 셀 내부 숫자 크기
        cbar_kws={'label': 'Feature Mean'},
        ax=ax
        )
    ax.set_title(f"Case {case_str} - Feature Mean per Trajectory Type")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/case{case_str}_type_feature_means_heatmap.png", dpi=300)
    plt.close()