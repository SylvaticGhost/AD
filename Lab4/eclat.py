def eclat_algorithm(df, min_support, min_confidence=0.5):
    transactions = []
    for i, row in df.iterrows():
        transaction = set(df.columns[row == 1])
        transactions.append(transaction)

    item_supports = {}
    for transaction in transactions:
        for item in transaction:
            if item in item_supports:
                item_supports[item] += 1
            else:
                item_supports[item] = 1

    n_transactions = len(transactions)
    for item in item_supports:
        item_supports[item] /= n_transactions

    frequent_items = {item: support for item, support in item_supports.items() if support >= min_support}

    result = [(frozenset([item]), support) for item, support in frequent_items.items()]

    k = 2
    while True:
        prev_itemsets = [itemset for itemset, _ in result if len(itemset) == k-1]

        if not prev_itemsets:
            break

        candidates = []
        for i in range(len(prev_itemsets)):
            for j in range(i+1, len(prev_itemsets)):
                itemset1 = list(prev_itemsets[i])
                itemset2 = list(prev_itemsets[j])

                if itemset1[:-1] == itemset2[:-1]:
                    candidate = frozenset(list(prev_itemsets[i]) + [itemset2[-1]])
                    if len(candidate) == k:
                        candidates.append(candidate)

        k_itemsets = []
        for candidate in candidates:
            support = sum(1 for transaction in transactions if candidate.issubset(transaction)) / n_transactions
            if support >= min_support:
                k_itemsets.append((candidate, support))

        # Add k-itemsets to result
        result.extend(k_itemsets)
        k += 1

        if not k_itemsets:
            break

    rules = []
    for itemset, support in result:
        if len(itemset) >= 2:
            for r in range(1, len(itemset)):
                for antecedent in _get_subsets(itemset, r):
                    consequent = itemset - antecedent

                    antecedent_support = next(s for i, s in result if i == antecedent)
                    confidence = support / antecedent_support

                    if confidence >= min_confidence:
                        consequent_support = next(s for i, s in result if i == consequent)
                        lift = confidence / consequent_support

                        # Calculate conviction: (1 - consequent_support) / (1 - confidence)
                        if confidence < 1:
                            conviction = (1 - consequent_support) / (1 - confidence)
                        else:
                            conviction = float('inf')  # Conviction is infinite when confidence = 1

                        rules.append((antecedent, consequent, support, confidence, lift, conviction))

    return result, rules

def _get_subsets(itemset, length):
    from itertools import combinations
    return [frozenset(subset) for subset in combinations(itemset, length)]