import re
import nltk
from argparse import ArgumentParser
from nltk.corpus import stopwords
from pandas import read_csv
from quickumls import QuickUMLS
from operator import itemgetter
from tqdm import tqdm


def main(args):
    print('=============')
    if args.granularity not in ['N', 'S', 'W']:
        raise TypeError('Invalid value for the granularity - should be N, S, or W')

    print('Reading MIMIC-III data...')
    if args.skiplims is None:
        notes_df = read_csv(args.noteevents_fp)
    else:
        to_skip = []
        for i in range(0, len(args.skiplims), 2):
            to_skip += [j for j in range(args.skiplims[i], args.skiplims[i+1])]
        notes_df = read_csv(args.noteevents_fp, skiprows=to_skip)

    print('Preprocessing notes ...')
    parsed_list = []
    for note in tqdm(notes_df['TEXT']):
        note = note.lower()
        note = re.sub('[^a-zA-Z.]', ' ', note)
        note = re.sub(r'\s+', ' ', note)

        # For finer granularity than entire notes, they are tokenized so that we
        # can iterate over sentences or words
        if args.granularity != 'N':
            note = nltk.sent_tokenize(note)
            if args.granularity == 'W':
                for i in range(len(note)):
                    note[i] = re.sub('[.]', '', note[i])
                    note = [nltk.word_tokenize for sentence in note]
                    for i in range(len(note)):
                        note[i] = [word for word in note_[i] if word not in stopwords.words('english')]

        parsed_list.append(note)

    print('Matching with UMLS corpus...')
    # initialise QuickUMLS string matching object
    matcher = QuickUMLS(args.qumls_fp, threshold=args.thresh, similarity_name=args.sim)

    # useful to define these two here so the mapping loop isn't too verbose
    qumls_getter = lambda n: matcher.match(n, best_match=False, ignore_syntax=False)
    # this gets the maximum similarity score and its index in the list for that ngram
    simscore_getter = lambda l: max(enumerate([d['similarity'] for d in l]), key=itemgetter(1))

    ALL = args.attr == 'all'

    if ALL:
        # make a dictionary which will have the columns to be added to the dataframe
        names = ['term', 'cui', 'semtypes']
        attrs = {}
        for name in names:
            attrs[name] = []
    else:
        mapped_corpus = []
    if args.keep_similarity: similarity_scores = []

    for note in tqdm(parsed_list):
        if ALL:
            # note-level mini-version of the dictionary "attrs" to collect the attributes for each note
            sub_attr = {}
            for name in names:
                sub_attr[name] = []
        else:
            single_attr_list = []
        if args.keep_similarity: sim_list = []
        if args.granularity == 'N':
            res = qumls_getter(note)
            for l in res:
                ss = simscore_getter(l)
                if ALL:
                    for name in names:
                        sub_attr[name].append(l[ss[0]][name])
                else:
                    single_attr_list.append(l[ss[0]][args.attr])
                if args.keep_similarity: sim_list.append(ss[1])
        else:
            for s in note:
                if args.granularity != 'W':
                    res = qumls_getter(s)
                    for l in res:
                        ss = simscore_getter(l)
                        if ALL:
                            for name in names:
                                sub_attr[name].append(l[ss[0]][name])
                        else:
                            single_attr_list.append(l[ss[0]][args.attr])
                        if args.keep_similarity: sim_list.append(ss[1])
                else:
                    for w in s:
                        res = qumls_getter(w)[0]
                        ss = simscore_getter(res)
                        if ALL:
                            for name in names:
                                sub_attr[name].append(res[ss[0]][name])
                        else:
                            single_attr_list.append(res[ss[0]][args.attr])
                        if args.keep_similarity: sim_list.append(ss[1])
        if ALL:
            if args.filter_semtypes_file is not None:
                irrelevant_type_ids = [i[:-1] for i in open(args.filter_semtypes_file, 'r')]
                indices_to_remove = []
                for st_set in sub_attr['semtypes']:
                    if all(st in irrelevant_type_ids for st in st_set):
                        indices_to_remove.append(sub_attr['semtypes'].index(st_set))
                for name in names:
                    sub_attr[name] = [st for st in sub_attr[name] if sub_attr[name].index(st) not in indices_to_remove]
            for name in names:
                mapped_note = ''
                for a in sub_attr[name]:
                    if name == 'semtypes':
                        for a_ in a:
                            mapped_note += a_+' '
                    else:
                        mapped_note += a+' '
                attrs[name].append(mapped_note)
        else:
            mapped_note = ''
            for word in single_attr_list:
                mapped_note += word
                mapped_note += ' '
            mapped_corpus.append(mapped_note)

    print('Matching finished!')

    print('Writing .csv file...')
    if ALL:
        for name, mapped_corpus in attrs.items():
            notes_df[name.upper()] = mapped_corpus
        if args.keep_similarity: notes_df['SIM_SCORE'] = sim_list
    else:
        notes_df[args.attr.upper()] = mapped_corpus

    if args.outfilepath[-4:] != '.csv': args.outfilepath += '.csv'
    notes_df.to_csv(args.outfilepath, index=False)

    print('Done!')
    print('=============')


if __name__ == '__main__':
    parser = ArgumentParser(prog='mimicnotes2umls.py', description='''Reads MIMIC-III NOTEEVENTS file
        and maps relevant terms to UMLS concepts''')
    parser.add_argument('noteevents_fp', type=str, help='path to NOTEEVENTS.csv')
    parser.add_argument('qumls_fp', type=str, help='path to QuickUMLS data')
    parser.add_argument('-g', dest='granularity', type=str, default='N', help='''granularity at which to pass
        terms to the UMLS matcher. Three possible values; N (note - default), S (sentence), or W (word)''')
    parser.add_argument('-t', dest='thresh', type=float, default=0.7, help='''Score threshold for the QuickUMLS
        matching function, default 0.7''')
    parser.add_argument('-s', dest='sim', type=str, default='jaccard', help='''String specifying the type of
        similarity measure to be passed to the QuickUMLS matcher, default "jaccard"''')
    parser.add_argument('-a', dest='attr', type=str, default='all', choices=['term', 'cui', 'semtypes', 'all'],
        help='''Attribute of QuickUMLS return list to extract - the default is "all"''')
    parser.add_argument('-o', dest='outfilepath', type=str, default='umls_noteevents', help='''Optional output file
        path for the cleaned csv''')
    parser.add_argument('-r', dest='skiplims', nargs='+', type=int, help='''To only use a subset of the MIMIC-III notes
        file, list start and end points (line numbers) delimiting sections of the file for pandas.read_csv to skip when
        reading it''')
    parser.add_argument('-ks', dest='keep_similarity', action='store_true', help='''Include this flag to keep the
        similarity scores of each UMLS concept found to be used in the aggregation step''')
    parser.add_argument('-ff', dest='filter_semtypes_file', type=str, default=None, help='''Include this
        flag to create another output file with certain semantic types removed (along with the path to a file
        containing the list of semantic type identifiers to NOT use) - only works for "all" option of the -a flag.''')

    main(parser.parse_args())
