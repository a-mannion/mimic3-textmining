import pandas as pd
from argparse import ArgumentParser
from warnings import simplefilter


def main(args):
    assert((args.frac > 0.0) and (args.frac <= 0.4))

    ndf = pd.read_csv('../data/mimic_experiment_writes/all_data.csv',
                      index_col=0)
    adf = pd.read_csv(args.rdf_in, index_col=0) # Readmissions dataset

    print('Initial data: Note events dataset: {:d} entries, {:d} admissions of {:d} patients'\
        .format(len(ndf), len(ndf.HADM_ID.drop_duplicates()), len(ndf.SUBJECT_ID.drop_duplicates())))
    print('Readmission: {:d} entries, {:d} patients'.format(len(adf), len(adf.index.drop_duplicates())))

    # remove patients with no notes
    adf = adf[adf.SUBJECT_ID.isin(ndf.SUBJECT_ID)]
    print('Patients with no notes removed: now {:d} in readmissions set'.format(len(adf)))

    # stratified sampling done by sampling separately from readm=1 and readm=0 subsets
    readm_df = adf[adf.READM == 1]
    n_patients = len(adf)
    readm_prev = len(readm_df)/n_patients
    print('{:d} readmitted patients, giving {:.2f}% prevalence'.format(len(readm_df), 100*readm_prev))
    holdout_size = int(args.frac*n_patients)
    holdout_readm_size = int(readm_prev*holdout_size)
    holdout_readm = readm_df.sample(holdout_readm_size, random_state=args.seed)
    holdout_not = adf[adf.READM == 0].sample(holdout_size-holdout_readm_size, random_state=args.seed)
    holdout_patients = pd.concat((holdout_readm, holdout_not)).index.get_level_values(0)
    print('{:d} holdout readmitted patients, {:d} not readmitted, total {:d} (should be {:d})'\
        .format(holdout_readm_size, len(holdout_not), len(holdout_patients), holdout_size))

    adf[~adf.SUBJECT_ID.isin(holdout_patients)].\
        to_csv(args.ddir+'readmission_train.csv')
    adf[adf.SUBJECT_ID.isin(holdout_patients)].\
        to_csv(args.ddir+'readmission_test.csv')

    ndf[~ndf.SUBJECT_ID.isin(holdout_patients)].to_csv(args.ddir+'notes_train.csv', index=False)
    ndf[ndf.SUBJECT_ID.isin(holdout_patients)].to_csv(args.ddir+'notes_test.csv', index=False)


if __name__ == '__main__':
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=pd.errors.DtypeWarning)

    parser = ArgumentParser()
    parser.add_argument('rdf_in', type=str)
    parser.add_argument('ddir', type=str)
    parser.add_argument('--frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=18520)

    main(parser.parse_args())
