def df_list_to_readable(df, columns):
    copy_df = df.copy()
    for column in columns:
        copy_df[column] = df[column].apply(
            lambda x: '//'.join(list(map(str, x))))
    return copy_df


def readable_to_df_list(df, columns):
    def transform_back(x):
        try:
            return list(map(eval, x.split('//')))
        except:
            try:
                return x.split('//')
            except:
                return x

    copy_df = df.copy()
    for column in columns:
        copy_df[column] = df[column].apply(transform_back)
    return copy_df