_to_list = lambda x: list(x)
_col_to_str = lambda x: ', '.join(x)

def sets_to_lists(df_column):
    df_column.update(df_column.apply(_to_list))

def collection_to_string(df_column):
    df_column.update(df_column.apply(_col_to_str))