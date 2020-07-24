import numpy as np
import torch
import torch.utils.data as data
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from torch.nn import CrossEntropyLoss
from pandas import DataFrame, read_csv, isna
from nltk import word_tokenize
from collections import OrderedDict
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from pytorch_lightning import LightningModule
from transformers import BertTokenizer, BertForSequenceClassification
from math import ceil
from random import sample


class W2VEmbedAggregate(object):
    '''Object that implements scikit-learn style methods to compute Word2Vec embeddings
    and aggregate across notes/patients'''

    def __init__(self, patient_ids, **kwargs):
        self.patient_ids = patient_ids
        self.arg_names = ['sg', 'size', 'window', 'min_count', 'alpha', 'iter', 'workers']

    def set_params(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.arg_names)

    def fit(self, text, y=None):
        '''Trains a Word2Vec model: only to be called internally in the fit() method of the grid search'''
        w2v_kwargs = {}
        w2v_kwargs.update((k, v) for k, v in self.__dict__.items() if k in self.arg_names)
        self.embedding = Word2Vec(text, **w2v_kwargs)

        return self

    def transform(self, text, assign_to_attr=True):
        '''Aggregates over all word embeddings in each note'''
        word_vectors = self.embedding.wv
        dim = word_vectors.vector_size
        n = len(text)

        # aggregate the word vectors for each note (concatenation)
        _concat_aggreg = lambda arr: np.concatenate(
            (np.mean(arr, 0), np.max(arr, 0), np.min(arr, 0))
        )
        X = np.empty((n, dim*3))
        for i in range(n):
            note_list = []
            for word in text[i]:
                try:
                    note_list.append(word_vectors[word])
                except KeyError:
                    continue
            if len(note_list) == 0:
                # even if no word vectors are found the output matrix still needs to be the right shape
                X[i,:] = np.zeros(dim*3)
            else:
                note_array = np.array(note_list).reshape((len(note_list), dim))
                X[i, :] = _concat_aggreg(note_array)

        if assign_to_attr:
            self.note_level_aggregations = X

        return X


class MIMICWord2VecReadmissionPredictor(object):
    '''Implementation class for the ML pipeline that goes from the cleaned/annotated
    text -> word embeddings -> SVM classification'''

    def __init__(self, txtvar, st_aug, seed=1, train_chunksize=1e5, test_chunksize=1e3, db=False):
        self.txtvar = txtvar
        self.st_aug = st_aug
        self.seed = seed
        self.train_chunksize = train_chunksize
        self.test_chunksize = test_chunksize
        self.db = db

    def _load_data(self, corpus_fp, readm_fp, chunksize, adapt_for_gridsearch=False):
        cols = ['SUBJECT_ID', 'HADM_ID', self.txtvar]
        dtypes = {}
        for c in cols:
            dtypes[c] = str
        if self.st_aug:
            cols.append('SEMTYPES')
            dtypes['SEMTYPES'] = str
        corpus_df = read_csv(
            corpus_fp,
            index_col=0,
            usecols=cols,
            dtype=dtypes,
            chunksize=chunksize
        )
        readm_df = read_csv(readm_fp, index_col=0)
        patient_ids = []
        text = []
        readm_df = read_csv(readm_fp, index_col=0)
        for chunk in corpus_df:
            chunk = chunk[chunk.index.isin(readm_df.index)]
            patient_ids += chunk.index.get_level_values(0).tolist()
            if sum(isna(chunk[self.txtvar])) > 0:
                chunk[self.txtvar] = chunk[self.txtvar].fillna('')
            if self.st_aug:
                chunk = chunk.assign(
                    **{self.txtvar:chunk[self.txtvar]+chunk.SEMTYPES.fillna('')}
                )
                chunk.drop('SEMTYPES', axis=1, inplace=True)
            for note in chunk[self.txtvar]:
                text.append(word_tokenize(note))
            if self.db:
                break

        # labels have to be the same length as train data for the pipeline
        # make sure that labels are ordered according to the patient IDs in the text dataset
        readm_df = readm_df[readm_df.index.isin(patient_ids)]
        input_labels = readm_df.READM.values
        readm_ordering_map = {}
        for p_id, label in zip(readm_df.index, input_labels):
            readm_ordering_map[p_id] = label

        # order the labels to be correspond to the output of patient aggregations
        labels = []
        unique_id_list = list(OrderedDict.fromkeys(patient_ids))
        for p_id in unique_id_list:
            labels.append(input_labels[readm_df.index.tolist().index(p_id)])

        if adapt_for_gridsearch:
            gridsearch_labels = []
            for p_id in patient_ids:
                try:
                    gridsearch_labels.append(readm_ordering_map[p_id])
                except KeyError:
                    continue

            return patient_ids, text, labels, gridsearch_labels
        else:
            return patient_ids, text, labels

    def _load_train_data(self, corpus_fp, readm_fp, chunksize, adapt_for_gridsearch):
        self.train_patient_ids, self.train_text, self.train_labels, self.gridsearch_labels =\
            self._load_data(
                corpus_fp=corpus_fp, readm_fp=readm_fp, chunksize=chunksize, adapt_for_gridsearch=adapt_for_gridsearch
            )

    def _load_test_data(self, corpus_fp, readm_fp):
        self.test_patient_ids, self.test_text, self.test_labels =\
            self._load_data(corpus_fp=corpus_fp, readm_fp=readm_fp, chunksize=1e3)

    def choose_params(self, corpus_fp, readm_fp, n_jobs, use_multithreading=False):
        self._load_train_data(
            corpus_fp=corpus_fp, readm_fp=readm_fp, chunksize=self.train_chunksize, adapt_for_gridsearch=True
        )
        pipeline = Pipeline(
            steps = [
                ('embed_agg', W2VEmbedAggregate(patient_ids=self.train_patient_ids)),
                ('clf', SGDClassifier(random_state=self.seed))
            ]
        )
        lr_grid = [10**i for i in range(-4, -1)]
        param_grid = [
            {
                'embed_agg__sg':[1, 0],
                'embed_agg__size':[100, 200],
                'embed_agg__window':[5, 7, 9],
                'embed_agg__alpha':lr_grid,
                'clf__alpha':lr_grid
            }
        ]
        grid_w2v_sgd = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            refit=True,
            n_jobs=n_jobs,
            scoring='roc_auc',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        )
        if use_multithreading:
            from sklearn.utils import parallel_backend

            with parallel_backend('threading'):
                gridsearch_res = grid_w2v_sgd.fit(self.train_text, self.gridsearch_labels)
        else:
            gridsearch_res = grid_w2v_sgd.fit(self.train_text, self.gridsearch_labels)

        self.w2v_agg_model = gridsearch_res.best_estimator_.named_steps['embed_agg']
        self.clf = gridsearch_res.best_estimator_.named_steps['clf']

    def _patient_aggregation(self, note_vectors):
        dim = self.w2v_agg_model.embedding.wv.vector_size
        patient_to_nv_map = OrderedDict()
        for p_id, vector in zip(self.train_patient_ids, note_vectors):
            if p_id not in patient_to_nv_map:
                patient_to_nv_map[p_id] = vector
            else:
                patient_to_nv_map[p_id] = np.vstack((patient_to_nv_map[p_id], vector))
        X = np.empty((len(patient_to_nv_map), dim))
        i = 0
        for array in patient_to_nv_map.values():
            X[i,:] = np.mean(array, 0)
            i += 1

        return X

    def train(self, text=None, labels=None):
        if text is not None and labels is not None:
            # TODO assume we're running the full pipeline from scratch and instantiate a new embed-and-aggregate object
            pass
        else:
            # get note-level vectors and take the mean over them for each patient
            X = self._patient_aggregation(
                self.w2v_agg_model.note_level_aggregations
            )

            # run a finer-tuned search for the best learning rate for the classifier using patient-level classifications
            current_lr = self.clf.alpha
            lr_grid = [current_lr*x for x in [0.5, 1.0, 2.0]]
            grid_sgd = GridSearchCV(
                SGDClassifier(random_state=self.seed),
                param_grid={'alpha':lr_grid},
                refit=True,
                n_jobs=-1,
                scoring='roc_auc',
                cv=StratifiedKFold(n_splits=5, random_state=self.seed)
            )
            alpha_gridsearch_res = grid_sgd.fit(X, self.train_labels)
            if alpha_gridsearch_res.best_estimator_.alpha != current_lr:
                self.clf.alpha = alpha_gridsearch_res.best_estimator_.alpha

    def test(self, corpus_fp, readm_fp, out_fp=None, save_model_fp=None):
        self._load_test_data(corpus_fp, readm_fp)
        self.w2v_agg_model.embedding.train(self.test_text)
        X = self._patient_aggregation(
            self.w2v_agg_model.transform(self.test_text, assign_to_attr=False)
        )
        test_pred = self.clf.predict(X)
        test_scores = self.clf.decision_function(X)
        test_prec = precision_score(self.test_labels, test_pred, average='weighted')
        test_recall = recall_score(self.test_labels, test_pred, average='weighted')
        test_f1 = f1_score(self.test_labels, test_pred, average='weighted')
        test_auroc = roc_auc_score(self.test_labels, test_scores, average='weighted')

        if out_fp is not None:
            with open(out_fp, 'w+') as model_desc:
                model_desc.write(
                    '''Word2Vec-based test results\nVariable {}\nSemantic types: {}\n\n---\nPipeline\n---{}\n{}\n
W2V Params:\n{}, LR {:.4f}, dim {:d}, window {:d}, epochs {:d}
\n---\nScores\n---\nPrecision {:.4f}, Recall {:.4f}, F1 = {:.4f}, AUROC {:.4f}'''\
                        .format(self.txtvar, 'yes' if self.st_aug else 'no', self.w2v_agg_model, self.clf,
                            'skip-gram' if self.w2v_agg_model.embedding.sg == 1 else 'CBOW', self.w2v_agg_model.alpha,
                            self.w2v_agg_model.size, self.w2v_agg_model.window, self.w2v_agg_model.iter, test_prec,
                            test_recall, test_f1, test_auroc
                        )
                )

        if save_model_fp is not None:
            self.w2v_agg_model.embedding(save_model_fp)


class EncodedDataset(data.Dataset):
    '''Pytorch-inherited dataset class that tokenizes the text batch-by-batch as
    it is passed to the dataloader to save RAM'''

    def __init__(self, df, bert_model, txtvar):
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

        # encoding has to be done during the initialisation, because torch expects all of the tensors output by __getitem__() to
        # be of the same size, so I can't return a stack of sequences for a patient index, it has to be a single sequence for the
        # sequence index
        input_ids, attn_masks, labels, patient_ids = [], [], [], []
        for patient, note, label in zip(df.SUBJECT_ID, df[txtvar], df.READM):
            encoded_dict = tokenizer.encode_plus(
                note,
                add_special_tokens=True,
                max_length=512,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
                return_overflowing_tokens=True
            )
            input_ids.append(encoded_dict['input_ids'].reshape(512))
            attn_masks.append(encoded_dict['attention_mask'].reshape(512))
            patient_ids.append(patient)
            labels.append(label)

            if 'overflowing_tokens' in encoded_dict.keys():
                overflow = encoded_dict['overflowing_tokens']
                n_overflow = len(overflow)
                # split the overflowing tokens into sequences of size 512 and label them
                # with the current subject identifier and label
                for i in range(ceil(n_overflow/512)):
                    # duplicate ID and label for each sequence
                    patient_ids.append(patient)
                    labels.append(label)
                    seq = overflow[512*i:512*(i+1)]
                    if len(seq) == 512:
                        attn_mask = torch.ones(512)
                    else:
                        short_seq_len = n_overflow-i*512
                        seq += [0 for i in range(512-short_seq_len)]
                        attn_mask = torch.cat(
                            (torch.ones(short_seq_len), torch.zeros(512-short_seq_len))
                        )
                    input_ids.append(torch.tensor(seq))
                    attn_masks.append(attn_mask.long())

        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.labels = labels
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(self.patient_ids[idx]) != list:
            self.patient_ids[idx] = [self.patient_ids[idx]]

        return {
            'input_ids' : self.input_ids[idx],
            'attn_masks' : self.attn_masks[idx],
            'labels' : torch.tensor(self.labels[idx]),
            'patient_ids' : torch.tensor([int(p) for p in self.patient_ids[idx]])
        }


class MIMICBERTReadmissionPredictor(LightningModule):
    '''This class implements model hooks into the Pytorch-Lightning framework, basically
    a wrapper around Pytorch module functionality'''

    def __init__(self, **kwargs):
        super().__init__()

        # if parse_version(pl.__version__) < parse_version('0.8.1'):
        #     raise RuntimeError('''This implementation requires Pytorch-Lightning version
        #         0.8.1 or later''')

        params = [
            'n_train_fp', 'r_train_fp', 'n_test_fp', 'r_test_fp', # data file paths
            'val_frac', 'batch_size', 'threads', # implementation arguments
            'bert_model', 'txtvar', 'seqlen', 'st_aug', 'scale_factor', # language-model arguments
            'test_metric_avg', # how to tell sklearn to account for class imbalance in the test set
            'db', # boolean - debug mode
            'write_test_results_to',
            'update_all_params', # bool: update the entire BERT model rather than just fine-tuning the final layer
            'verbose' #bool
        ]

        # default arguments
        self.scale_factor = 2.0
        self.threads = torch.get_num_threads()
        self.test_metric_avg = 'weighted'
        self.db = False
        self.write_test_results_to = None
        self.update_all_params = False
        self.verbose = False

        # load input arguments
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in params)

        self.model = BertForSequenceClassification.from_pretrained(self.bert_model)
        self.loss = CrossEntropyLoss(reduction='none')

        if not self.update_all_params:
            # tells the optimiser not to propagate the gradient over the entire model
            for name, param in self.model.named_parameters():
                if name.startswith('embeddings'):
                    param.requires_grad = False

        # this function is not in versions <0.8.1
        self.save_hyperparameters('epochs', 'lr', 'momentum')

    def setup(self, stage):

        def _load_txt_df(fp): # to be put in an external util script later
            cols = ['SUBJECT_ID', 'HADM_ID', self.txtvar]
            types = {
                'SUBJECT_ID':int,
                'HADM_ID':object,
                self.txtvar:str
            }
            if self.st_aug:
                cols.append('SEMTYPES')
                types['SEMTYPES'] = str
            df = read_csv(
                fp,
                usecols=cols,
                dtype=types,
                nrows=self.batch_size if self.db else None
            )
            if sum(isna(df[self.txtvar])) > 0:
                df[self.txtvar] = df[self.txtvar].fillna('')

            return df

        def _dataframe_setup(nfp, rfp, split=True):
            if self.verbose:
                print('Reading data from .csv...')
            text_df = _load_txt_df(nfp)
            labelled_text = text_df.merge(read_csv(rfp, index_col=0), on='SUBJECT_ID', how='left')

            # add semantic type codes if specified
            if self.st_aug:
                labelled_text[self.txtvar] += labelled_text['SEMTYPES']
                labelled_text[self.txtvar] = labelled_text[self.txtvar].apply(str)
                labelled_text.drop('SEMTYPES', axis=1, inplace=True)

            if split:
                # do random split of patients list to ensure notes for the same patient don't get split between the training and validation sets
                all_patients = labelled_text.SUBJECT_ID.drop_duplicates().tolist()
                n_val_patients = int(len(all_patients)*self.val_frac)
                val_patients = sample(all_patients, n_val_patients)
                val_df = labelled_text[labelled_text.SUBJECT_ID.isin(val_patients)]
                train_df = labelled_text[~labelled_text.SUBJECT_ID.isin(val_df.SUBJECT_ID)]

                return train_df, val_df
            else:
                return labelled_text

        if stage == 'fit':
            if self.verbose:
                print('Loading training & validation datasets...')
            self.train_df, self.val_df = _dataframe_setup(self.n_train_fp, self.r_train_fp)
        if stage == 'test':
            if self.verbose:
                print('Loading test dataset...')
            self.test_df = _dataframe_setup(self.n_test_fp, self.r_test_fp, split=False)

    def forward(self, input_ids, attn_masks):
        logits, = self.model(input_ids, attn_masks.float())

        return logits

    def train_dataloader(self):
        train_ds = EncodedDataset(self.train_df, self.bert_model, self.txtvar)
        return data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            sampler=data.RandomSampler(train_ds),
            num_workers=self.threads
        )

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'], batch['attn_masks'])
        loss = self.loss(logits, batch['labels']).mean()

        return {'loss':loss, 'log':{'train_loss':float(loss)}}

    def val_dataloader(self):
        val_ds = EncodedDataset(self.val_df, self.bert_model, self.txtvar)
        return data.DataLoader(
            val_ds,
            batch_size=self.batch_size,
            sampler=data.SequentialSampler(val_ds),
            num_workers=self.threads
        )

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'], batch['attn_masks'])
        loss = self.loss(logits, batch['labels'])
        acc = (logits.argmax(-1) == batch['labels']).float()

        return {'loss':loss, 'acc':acc}

    def validation_epoch_end(self, outputs):
        loss = torch.cat([o['loss'] for o in outputs], 0).mean()
        acc = torch.cat([o['acc'] for o in outputs], 0).mean()
        out = {'loss':loss, 'acc':acc}

        return {**out, 'log':out}

    def test_dataloader(self):
        test_ds = EncodedDataset(self.test_df, self.bert_model, self.txtvar)
        return data.DataLoader(
            test_ds,
            batch_size=self.batch_size,
            sampler=data.SequentialSampler(test_ds),
            num_workers=self.threads
        )

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'], batch['attn_masks'])
        scaled_logits, patient_labels = self._aggreg_subseq_logits(logits, batch['labels'], batch['patient_ids'])
        loss = self.loss(scaled_logits, patient_labels)
        predictions = scaled_logits.argmax(-1)
        acc = (predictions == patient_labels).float()
        labels_np = patient_labels.detach().numpy()
        predictions_np = predictions.detach().numpy()
        prec = precision_score(labels_np, predictions_np, average=self.test_metric_avg)
        recall = recall_score(labels_np, predictions_np, average=self.test_metric_avg)
        f1 = f1_score(labels_np, predictions_np, average=self.test_metric_avg)
        auroc = roc_auc_score(labels_np, predictions_np, average=self.test_metric_avg)

        return {'loss':loss, 'acc':acc, 'prec':prec, 'recall':recall, 'f1':f1, 'auroc':auroc, 'log':{'test_loss':loss}}

    def test_epoch_end(self, outputs):
        loss = torch.cat([o['loss'] for o in outputs]).mean()
        acc = torch.cat([o['acc'] for o in outputs]).mean()
        _mean_output = lambda metric, out: np.mean([o[metric] for o in out])
        prec, recall, f1, auroc = map(
            _mean_output,
            ('prec', 'recall', 'f1', 'auroc'),
            tuple([outputs for i in range(4)])
        )
        out = {'loss':float(loss), 'acc':float(acc), 'prec':prec, 'recall':recall, 'f1':f1, 'auroc':auroc}
        if self.write_test_results_to is not None:
            with open(self.write_test_results_to, 'w+') as test_log:
                test_log.write('-- TEST RESULTS --\n')
                for k, v in out.items():
                    test_log.write('{} : {:.4f}\n'.format(k, v))

        return {**out, 'log':out}

    def configure_optimizers(self):
        optim = SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum
        )
        sched = CyclicLR(
            optim,
            base_lr=1e-8,
            max_lr=self.hparams.lr
        )
        return [optim], [sched]

    def _aggreg_subseq_logits(self, logits, labels, patient_ids):
        '''
        This aggregates across the estimated readmission probability for each of the
        subsequences associated with each patient and outputs a readmission probability
        for each patient along with the reduced list of labels with which to calculate
        the test metrics
        '''
        def _scale(logits, c):
            factor = len(logits)/c
            return (np.max(logits)+np.mean(logits)*factor)/(1+factor)

        df = DataFrame()
        df['id'] = patient_ids
        df['r'] = labels
        df['p1'] = logits[:, 0].detach().numpy()
        df['p2'] = logits[:, 1].detach().numpy()

        scaled_logits1, scaled_logits2 = map(
            lambda v: df.loc[:, ['id', v]].groupby('id').agg(_scale, c=self.scale_factor)[v].values, ('p1', 'p2')
        )
        scaled_logits1, scaled_logits2 = map(
            lambda arr: torch.tensor(arr).reshape([len(arr), 1]), (scaled_logits1, scaled_logits2)
        )
        output_logits = torch.cat((scaled_logits1, scaled_logits2), dim=1)
        output_labels = df.loc[:, ['id', 'r']].drop_duplicates().r.values

        return output_logits, torch.tensor(output_labels)
