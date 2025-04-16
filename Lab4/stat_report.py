def stat_report(dataframes):
    import pandas as pd
    import numpy as np
    from helpers import sets_to_lists
    from tabulate import tabulate

    for _, df in dataframes:
        if 'antecedents' in df.columns and isinstance(df['antecedents'].iloc[0], frozenset):
            sets_to_lists(df['antecedents'])
        if 'consequents' in df.columns and isinstance(df['consequents'].iloc[0], frozenset):
            sets_to_lists(df['consequents'])

    all_rules = set()
    for _, df in dataframes:
        for ant, con in zip(df['antecedents'], df['consequents']):
            # Convert to frozenset for hashability
            ant_frozen = frozenset(ant)
            con_frozen = frozenset(con)
            all_rules.add((ant_frozen, con_frozen))

    index = pd.MultiIndex.from_tuples(
        [(', '.join(sorted(ant)), ', '.join(sorted(con))) for ant, con in all_rules],
        names=['antecedents', 'consequents']
    )

    metrics = ['support', 'confidence', 'lift', 'conviction']
    metric_dfs = {metric: pd.DataFrame(index=index, columns=[name for name, _ in dataframes])
                  for metric in metrics}

    for algo_name, df_algo in dataframes:
        for _, row in df_algo.iterrows():
            ant_str = ', '.join(sorted(row['antecedents']))
            con_str = ', '.join(sorted(row['consequents']))

            for metric in metrics:
                if metric in df_algo.columns:
                    metric_dfs[metric].loc[(ant_str, con_str), algo_name] = row[metric]

    rule_counts = pd.Series({name: len(df) for name, df in dataframes},
                            name="Number of rules")

    avg_metrics = {}
    for metric in metrics:
        avg_metrics[f"Avg {metric}"] = pd.Series(
            {name: df[metric].mean() if metric in df.columns else np.nan
             for name, df in dataframes}
        )

    common_rules = {}
    algo_names = [name for name, _ in dataframes]
    for i, name1 in enumerate(algo_names):
        for name2 in algo_names[i+1:]:
            common = metric_dfs['support'][[name1, name2]].dropna().shape[0]
            common_rules[f"{name1} ∩ {name2}"] = common

    print("\n=== SUPPORT VALUES BY ALGORITHM ===")
    print(tabulate(metric_dfs['support'].reset_index(), headers='keys', tablefmt='fancy_grid', showindex=False))

    print("\n=== CONFIDENCE VALUES BY ALGORITHM ===")
    print(tabulate(metric_dfs['confidence'].reset_index(), headers='keys', tablefmt='fancy_grid', showindex=False))

    print("\n=== LIFT VALUES BY ALGORITHM ===")
    print(tabulate(metric_dfs['lift'].reset_index(), headers='keys', tablefmt='fancy_grid', showindex=False))

    print("\n=== CONVICTION VALUES BY ALGORITHM ===")
    print(tabulate(metric_dfs['conviction'].reset_index(), headers='keys', tablefmt='fancy_grid', showindex=False))

    # Print summary statistics
    print("\n=== RULE COUNTS BY ALGORITHM ===")
    rule_counts_df = pd.DataFrame(rule_counts).reset_index()
    rule_counts_df.columns = ['Algorithm', 'Number of Rules']
    print(tabulate(rule_counts_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    print("\n=== AVERAGE METRICS BY ALGORITHM ===")
    avg_metrics_df = pd.DataFrame(avg_metrics)
    avg_metrics_df.index.name = 'Algorithm'
    print(tabulate(avg_metrics_df.reset_index(), headers='keys', tablefmt='fancy_grid', showindex=False))

    print("\n=== COMMON RULES BETWEEN ALGORITHMS ===")
    common_rules_df = pd.DataFrame(common_rules.items(), columns=['Algorithm Pair', 'Common Rules'])
    print(tabulate(common_rules_df, headers='keys', tablefmt='fancy_grid', showindex=False))