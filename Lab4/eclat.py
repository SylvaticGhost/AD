def eclat_algorithm(df, min_support):
    """
    Implementation of ECLAT algorithm

    Parameters:
    df : pandas DataFrame with binary values (0/1) for each item
    min_support : minimum support threshold

    Returns:
    list of tuples (itemset, support)
    """
    # Convert DataFrame to list of sets where each set contains the items present in a transaction
    transactions = []
    for i, row in df.iterrows():
        transaction = set(df.columns[row == 1])
        transactions.append(transaction)

    # Count support for each item
    item_supports = {}
    for transaction in transactions:
        for item in transaction:
            if item in item_supports:
                item_supports[item] += 1
            else:
                item_supports[item] = 1

    # Convert to relative support
    n_transactions = len(transactions)
    for item in item_supports:
        item_supports[item] /= n_transactions

    # Filter items by min_support
    frequent_items = {item: support for item, support in item_supports.items() if support >= min_support}

    # Generate frequent itemsets
    result = [(frozenset([item]), support) for item, support in frequent_items.items()]

    # Generate k+1 itemsets from k itemsets
    k = 2
    while result:
        # Get k-1 itemsets
        prev_itemsets = [itemset for itemset, _ in result if len(itemset) == k-1]

        # Generate candidate k-itemsets
        candidates = []
        for i in range(len(prev_itemsets)):
            for j in range(i+1, len(prev_itemsets)):
                itemset1 = list(prev_itemsets[i])
                itemset2 = list(prev_itemsets[j])

                # If first k-2 items are the same, we can join them
                if itemset1[:-1] == itemset2[:-1]:
                    candidate = frozenset(list(prev_itemsets[i]) + [itemset2[-1]])
                    if len(candidate) == k:
                        candidates.append(candidate)

        # Calculate support for candidates
        k_itemsets = []
        for candidate in candidates:
            support = sum(1 for transaction in transactions if candidate.issubset(transaction)) / n_transactions
            if support >= min_support:
                k_itemsets.append((candidate, support))

        # Add k-itemsets to result
        result.extend(k_itemsets)
        k += 1

        # If no k-itemsets were found, we're done
        if not k_itemsets:
            break

    return result