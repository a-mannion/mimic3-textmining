import pandas as pd
import argparse
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.sparse import vstack
from joblib import dump
from collections import OrderedDict
from tqdm import tqdm
from numpy import asarray, mean
from util import load_txt_df


def aggregate_embeddings(id_vector, tfidf_matrix):
    # I wasn't sure how pandas groupby+add would deal with scipy.sparse.csc
    # matrices so I just added them into an ordered dictionary...
    assert(len(id_vector) == tfidf_matrix.shape[0])
    tfidf_map = OrderedDict()
    for patient, vector in tqdm(zip(id_vector, tfidf_matrix), total=len(id_vector)):
        if patient not in tfidf_map:
            tfidf_map[patient] = vector
        else:
            tfidf_map[patient] += vector

    # ... and then stacked the dictionary values back to a matrix
    print('stacking matrix...')
    X = tfidf_map.popitem(False)[1]
    for x in tqdm(tfidf_map.values()):
        X = vstack((X, x))

    return X


def add_augmented_var(df):
    df[args.var+'_ST'] = df[args.var]+df['SEMTYPES']


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_train', type=str)
    parser.add_argument('n_test', type=str)
    parser.add_argument('r_train', type=str)
    parser.add_argument('r_test', type=str)
    parser.add_argument('var', type=str)
    parser.add_argument('--st_aug', action='store_true')
    parser.add_argument('--out_fp', type=str,
        default='~/data/mimic_experiment_writes/bowsgdresults.txt')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_model', action='store_true')

    return parser.parse_args()


def main():
    print('=======================')
    global args
    args = parse_arguments()

    print('Loading data...')
    warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
    notes_train = load_txt_df(
        fp=args.n_train,
        var=args.var,
        st_aug=args.st_aug
    )
    readm_train = pd.read_csv(args.r_train, index_col=0)
    notes_test = load_txt_df(
        fp=args.n_test,
        var=args.var,
        st_aug=args.st_aug
    )
    readm_test = pd.read_csv(args.r_test, index_col=0)

    if args.st_aug:
        print('Adding semantic types to '+args.var)
        add_augmented_var(notes_train)
        add_augmented_var(notes_test)
        args.var += '_ST'

    # stack text data together to train embeddings
    all_notes = pd.concat((notes_train, notes_test))

    # double check for NaNs
    if sum(pd.isna(all_notes[args.var])) > 0:
        all_notes[args.var] = all_notes[args.var].fillna('')

    # compute bag-of-words embeddings
    print('Calculating BoW Matrix...')
    bow_matrix = CountVectorizer(analyzer=lambda s: s.split(' '))\
        .fit(all_notes[args.var]).transform(all_notes[args.var])

    # normalisation via term frequency-inverse document frequency
    print('TF-IDF...')
    tfidf_matrix = TfidfTransformer().fit(bow_matrix).transform(bow_matrix)

    print('aggregating training set embeddings...')
    X_train = aggregate_embeddings(
        notes_train.SUBJECT_ID.values,
        tfidf_matrix[:len(notes_train)]
    )

    # cross-validation grid search for the best-scoring model
    print('Testing different SVM models...\n')
    grid_sgd = GridSearchCV(
        SGDClassifier(random_state=args.seed),
        param_grid=[{'alpha':[10**i for i in range(-4, 1)],
                     'penalty':['l2', 'elasticnet']}],
        refit=True,
        n_jobs=-1,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    )
    gridsearch_res = grid_sgd.fit(X_train, readm_train.READM.values)

    # write out details of the most optimal model
    details = '''SVM classification with SGD on BoW embeddings of MIMIC-III {} variable
Best estimator\n{}\nArea under ROC: {:.3f}'''\
        .format(args.var, gridsearch_res.estimator, gridsearch_res.best_score_)
    print('-- Training results --')
    print(details)

    # pickle the model
    if args.save_model:
        dump(gridsearch_res, args.out_fp[:args.out_fp.rindex('.')+1]+'joblib')

    print('aggregating test set embeddings...')
    X_test = aggregate_embeddings(
        notes_test.SUBJECT_ID.values,
        tfidf_matrix[len(notes_train):]
    )

    # make predictions
    print('Running chosen SVM model on test set...')
    test_pred = gridsearch_res.predict(X_test)
    test_score = gridsearch_res.decision_function(X_test)
    true_labels = readm_test.READM.values
    test_acc = mean(asarray((test_pred == true_labels), dtype=int))
    test_prec = precision_score(true_labels, test_pred)
    test_rec = recall_score(true_labels, test_pred)
    test_f1 = f1_score(true_labels, test_pred)
    test_auroc = roc_auc_score(true_labels, test_score)
    testres_str = '''\n--- TEST RESULTS ---\nAccuracy {:.4f}\nPrecision {:.4f}
Recall {:.4f}\nF1 {:.4f}\nAUROC: {:.4f}'''\
        .format(test_acc, test_prec, test_rec, test_f1, test_auroc)
    print(testres_str)

    with open(args.out_fp, 'w+') as out:
        out.write(details)
        out.write(testres_str)

    print('=======================')


if __name__ == '__main__':
    main()
