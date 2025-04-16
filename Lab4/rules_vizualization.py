import matplotlib.pyplot as plt
import seaborn as sns

def graph_rep(df, alg):
    rule_labels = [f"{', '.join(ant)} → {', '.join(con)}" for ant, con in
                   zip(df['antecedents'], df['consequents'])]

    plt.figure(figsize=(12, 8))

    plt.scatter(range(len(rule_labels)), df['confidence'],
                s=df['support'] * 5000, alpha=0.6,
                c=df['conviction'], cmap='viridis')
    plt.colorbar(label='Переконливість (Conviction)')

    plt.xticks(range(len(rule_labels)), rule_labels, rotation=90)
    plt.ylabel('Достовірність (Confidence)')
    plt.xlabel('Правила')
    plt.title(f'Асоціативні правила. Алгоритм: {alg}')
    plt.tight_layout()
    plt.show()

def heat_map(df, alg):
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning,
                            message="ast.Str is deprecated")
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                            message="Attribute s is deprecated")
    df['antecedents_str'] = df['antecedents'].apply(lambda x: ', '.join(x))
    df['consequents_str'] = df['consequents'].apply(lambda x: ', '.join(x))

    matrix_df = df.pivot(index='antecedents_str', columns='consequents_str', values='conviction')

    matrix_df.index.name = 'Antecedents'
    matrix_df.columns.name = 'Consequents'

    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix_df, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Асоціативні правила. Алгоритм: ' + alg)
    plt.tight_layout()
    plt.show()