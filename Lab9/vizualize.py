import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_clusters(df, dispersion_df, method_name, cluster_column='Cluster'):
    plt.figure(figsize=(12, 16))
    k = len(np.unique(df[cluster_column]))
    colors = plt.cm.rainbow(np.linspace(0, 1, k))
    for cluster, color in zip(np.unique(df[cluster_column]), colors):
        mask = df[cluster_column] == cluster
        disp_df_row = dispersion_df[dispersion_df['Cluster'] == cluster]
        dispersion = disp_df_row['Dispersion'].values[0]
        plt.scatter(df[mask]['All'], df[mask].index,
                    label=f'Кластер {cluster}', color=color)

        center = float(disp_df_row['Center'].values[0])
        y_positions = np.arange(len(df[mask].index))
        y_min = min(y_positions) if len(y_positions) > 0 else 0
        y_max = max(y_positions) if len(y_positions) > 0 else 0
        plt.axvline(x=center, color=color, linestyle='--',
                    alpha=0.5, linewidth=1)

        plt.text(center, 0, f'Центр: {center}',
                 rotation=90, verticalalignment='bottom',
                 color='gray', fontweight='bold')

    plt.title(f'{method_name} | Кластеризація країн за кількістю поїздок')
    plt.xlabel('Поїздки')
    plt.ylabel('Країни')
    plt.legend()
    plt.tight_layout()
    plt.show()


def start_visualization(df):
    plt.figure(figsize=(12, 16))
    plt.scatter(df['All'], df.index)
    plt.xlabel('Кількість поїздок')
    plt.ylabel('Країна')
    plt.title('Кількість поїздок по країнам')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def visualize_core_clusters(df, dispersion_df, cluster_column='Cluster'):

    plt.figure(figsize=(12, 18))
    sns.set(style="dark")

    df_sorted = df.sort_values('All', ascending=True)

    scatter = sns.scatterplot(
        data=df_sorted,
        x='All',
        y='Country',
        hue=cluster_column,
        style='is_core',
        palette='deep',
        s=100
    )

    k = np.unique(df_sorted[cluster_column])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(k)))
    for c in range(len(k)):
        cluster_data = df_sorted[df_sorted[cluster_column] == c]
        center = dispersion_df[dispersion_df['Cluster'] == c]['Center'].values[0]
        color = colors[c]
        plt.axvline(x=center, color=color, linestyle='--', alpha=0.5, linewidth=1)
        plt.text(center, 0, f'Центр: {center}', rotation=90, verticalalignment='bottom',
                 color='gray', fontweight='bold')


    plt.title("метод двоступеневої кластеризації | Кластеризація країн за кількістю поїздок")
    plt.xlabel("Кількість поїздок")
    plt.ylabel("Країна")
    plt.legend(title='Кластер')
    plt.tight_layout()
    plt.show()

def visualize_mean_center_clusters(df, dispersion_df, cluster_column='Cluster'):
    plt.figure(figsize=(12, 18))
    sns.scatterplot(x='All', y='Country', hue=cluster_column, data=df, palette='deep', style='is_center', s=100)

    centers_values = df[df['is_center'] == True]['All'].values
    for center in centers_values:
        plt.axvline(x=center, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        plt.text(center, 0, f'Центр: {center}', rotation=90, verticalalignment='bottom',
                 color='gray', fontweight='bold')

    plt.title('метод середньої відстані | Кластеризація країн за кількістю поїздок')
    plt.xlabel('Кількість поїздок')
    plt.ylabel('Країна')
    plt.show()
