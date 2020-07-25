import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm


def main(args):
    # get data
    print('===========')
    print('Reading data...')
    adm_df = pd.read_csv(args.adm_fp,
                         usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME'],
                         dtype={'SUBJECT_ID':str, 'HADM_ID':str},
                         parse_dates=['ADMITTIME'])

    note_cols = ['SUBJECT_ID', 'HADM_ID', 'TEXT', 'TERM', 'CUI', 'SEMTYPES']
    types = {}
    for n in note_cols:
        types[n] = str
    notes_df = pd.read_csv(args.notes_fp, usecols=note_cols, dtype=types)

    # split the data into "observation" and "prediction" periods
    # PHYSIONET WEBSITE: "dates were shifted into the future by a random offset
    # for each individual patient in a consistent manner to preserve intervals,
    # resulting in stays which occur sometime between the years 2100 and 2200.
    # Time of day, day of the week, and approximate seasonality were conserved
    # during date shifting"
    # so it's possible that admissions that were close to each other in the same
    # year had their order shifted at the month level, but I think it's fairly
    # safe to assume they can be ordered as they are
    print('Processing admissions...')
    readm_df = pd.DataFrame()
    adm_labels = []
    sorted_by_date = adm_df.\
        sort_values('SUBJECT_ID').\
        groupby('SUBJECT_ID').\
        apply(lambda x: x.sort_values('ADMITTIME')).\
        drop('SUBJECT_ID', axis=1)
    patient_ids = sorted_by_date.index.get_level_values(0).drop_duplicates()
    readm_df['SUBJECT_ID'] = patient_ids
    for subj in tqdm(patient_ids):
        times = list(sorted_by_date.loc[subj, 'ADMITTIME'])
        n_adm = len(times)
        if n_adm == 1:
            adm_labels.append(0)
        else:
            interval = (times[n_adm-1]-times[n_adm-2]).total_seconds()
            if interval < 15552e3 and interval > 0:
                adm_labels.append(1)
            elif interval <= 0:
                continue
            else:
                adm_labels.append(0)
    readmission_prevalence = len(adm_labels[adm_labels == 1])/len(adm_labels)
    readm_df['READM'] = adm_labels

    # write the percentage of readmissions to a text file
    with open(args.prev_fp, 'w+') as rp_out:
        rp_out.write('''READMISSION PREPROCESSING
Binary 6-month readmission prediction variable calculated from {}
Readmission Prevalence: {:.1f}%
            '''.format(args.adm_fp, 100*readmission_prevalence))

    readm_df.to_csv(args.out_fp, index=False)

    print('===========')


if __name__ == '__main__':
    parser = ArgumentParser(prog='make_readmission_variable.py',
        description='Generate labels for the readmission prediction task')
    parser.add_argument('adm_fp', type=str, help='path to admissions .csv file')
    parser.add_argument('notes_fp', type=str, help='''path to note events .csv file
        (cleaned text version)''')
    parser.add_argument('--out_fp', default='patient_readmission_labels.csv',
        help='path to output admissions .csv')
    parser.add_argument('--prev_fp', default='readmission_prevalence.txt',
        help='path to text file with the readmission prevalence percentage')

    main(parser.parse_args())
