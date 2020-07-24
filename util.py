from pandas import read_csv

def load_txt_df(fp, var, st_aug, _slice=None):
    cols = ['SUBJECT_ID', 'HADM_ID', var]
    types = {
        'SUBJECT_ID':int,
        'HADM_ID':object,
        var:str
    }
    if st_aug:
        cols.append('SEMTYPES')
        types['SEMTYPES'] = str
    df = read_csv(
        fp,
        usecols=cols,
        dtype=types,
        nrows=_slice
    )
    if sum(isna(df[var])) > 0:
        df[var] = df[var].fillna('')

    return df
