import re
from argparse import ArgumentParser
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from pandas import DataFrame, read_csv
from sys import stdout


def main(args):
    print('==========')
    cols = ['SUBJECT_ID', 'HADM_ID', 'TEXT']
    dtypes = {}
    for c in cols:
        dtypes[c] = str
    ndf_iter = read_csv(args.input_fp, usecols=cols, dtype=dtypes, chunksize=1)

    clean_df = DataFrame()
    notes, patients, admissions = [], [], []
    print('Iterating over dataset...\nNotes cleaned:')
    i = 0
    for chunk in ndf_iter:
        patients.append(chunk.SUBJECT_ID.values[0])
        admissions.append(chunk.HADM_ID.values[0])
        text = re.sub('[^a-zA-Z.]', ' ', chunk.TEXT.values[0].lower())
        words = word_tokenize(re.sub(r'\s+', ' ', text))
        notes.append(' '.join([w for w in words if w not in stopwords.words('english')]))
        i += 1
        stdout.write('\r')
        stdout.flush()
        stdout.write(str(i))
        stdout.flush()

    clean_df['SUBJECT_ID'] = patients
    clean_df['HADM_ID'] = admissions
    clean_df['TEXT'] = notes

    clean_df.to_csv(args.notes_output_name, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_fp', type=str, help='path to NOTEEVENTS file')
    parser.add_argument('-n', dest='notes_output_name', type=str, default='cleaned-noteevents.csv',
        help='path for output .csv file containing cleaned notes')

    main(parser.parse_args())
