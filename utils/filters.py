def filter_meta(df_meta, topics=None, orientations=None):
    # filter by the settings set on the config.yaml
    if topics:
        df_meta = df_meta[df_meta["Topic"].isin(topics)]
    if orientations:
        df_meta = df_meta[df_meta["Party_orientation"].isin(orientations)]
    return df_meta


def filter_years(years_folders, years=None):
    # filtering years folders, setting is set on the config.yaml
    if years:
        years_folders = list(set(years) & set(years_folders))
    return years_folders
