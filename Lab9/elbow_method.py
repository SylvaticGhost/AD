import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def _calc_decrease_diff(decreases):
    diff = [decreases[i] - decreases[i + 1] for i in range(len(decreases) - 1)]
    print(f"Різниця між спадами: {[f'{d:.2f}%' for d in diff]}")
    return diff

def _vizualize_elbow(clusters_count, inertias, decreases):
    plt.figure(figsize=(8, 5))
    plt.plot(clusters_count, inertias, marker='o', linewidth=2, markersize=8, label='Inertia')

    for i in range(1, len(clusters_count)):
        plt.annotate(f"{decreases[i]:.1f}%",
                     xy=((clusters_count[i - 1] + clusters_count[i]) / 2, (inertias[i - 1] + inertias[i]) / 2),
                     xytext=(0, -15), textcoords='offset points',
                     ha='center', va='top', fontsize=9)

    plt.xlabel('К-сть кластерів (k)')
    plt.ylabel('inertias')
    plt.title('Метод ліктя для визначення оптимальної кількості кластерів')
    plt.grid(True)
    plt.legend()
    plt.show()

def elbow(Z,df, criterion):
    min_dist = np.min(Z[:, 2])
    max_dist = np.max(Z[:, 2])
    thresholds = np.linspace(min_dist, max_dist, 10)
    thresholds = [t for t in thresholds if t != 0]

    clusters_count = []
    inertias = []
    decreases = []

    for thresh in thresholds:
        clusters = fcluster(Z, t=thresh, criterion=criterion)
        unique_clusters = np.unique(clusters)
        k = len(unique_clusters)

        if k in clusters_count:
            continue

        clusters_count.append(k)
        inertia = 0

        for cluster in unique_clusters:
            cluster_data = df.iloc[np.where(clusters == cluster)]
            if not cluster_data.empty:
                centroid = cluster_data.mean()
                inertia += np.sum(((cluster_data - centroid) ** 2).sum(axis=1))

        inertias.append(inertia)
        decrease = np.abs((inertias[-2] - inertias[-1]) / inertias[-2] * 100) if len(clusters_count) > 1 else 0
        decreases.append(decrease)
        print(f"Поріг: {thresh:.2f} | k: {k}, Inertia: {inertia:.2f}, Зміна: {decrease:.2f}%")

    decreases_diff = _calc_decrease_diff(decreases)

    elbow_idx = np.argmin(np.abs(decreases_diff))
    optimal_threshold = thresholds[elbow_idx]
    optimal_k = clusters_count[elbow_idx]
    print(f"Оптимальний поріг: {optimal_threshold:.2f}, Кількість кластерів: {optimal_k}")

    _vizualize_elbow(clusters_count, inertias, decreases)

    return optimal_k, optimal_threshold

def elbow_k(Z, df, max_k):
    clusters_count = []
    inertias = []
    decreases = []

    for k in range(1, max_k + 1):
        clusters = fcluster(Z, t=k, criterion='maxclust')
        if k in clusters_count:
            continue

        clusters_count.append(k)
        inertia = 0

        for cluster in np.unique(clusters):
            cluster_data = df[clusters == cluster]
            if len(cluster_data) > 0:
                centroid = cluster_data.mean(axis=0)
                inertia += np.sum(np.sum((cluster_data - centroid) ** 2, axis=1))

        inertias.append(inertia)
        decrease = np.abs((inertias[-2] - inertias[-1]) / inertias[-2] * 100) if len(clusters_count) > 1 else 0
        decreases.append(decrease)
        print(f"k: {k}, Inertia: {inertia:.2f}, Зміна: {decrease:.2f}%")

    decreases_diff = _calc_decrease_diff(decreases)
    elbow_idx = np.argmin(np.abs(decreases_diff))
    optimal_k = clusters_count[elbow_idx]
    print(f"Оптимальна кількість кластерів: {optimal_k}")

    _vizualize_elbow(clusters_count, inertias, decreases)

    return optimal_k

def elbow_KMeans(data, max_k):
    inertia = []
    k_range = range(1, max_k)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        inertia.append(kmeans.inertia_)

    decreases = [((inertia[i] - inertia[i + 1])/inertia[i]) * 100 for i in range(len(inertia) - 1)]
    decreases_diff = _calc_decrease_diff(decreases)

    elbow_idx = np.argmin(np.abs(decreases_diff))
    optimal_k = k_range[elbow_idx]
    print(f"Оптимальна кількість кластерів: {optimal_k}")

    return optimal_k