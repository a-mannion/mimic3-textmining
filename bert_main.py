import argparse
from lib import MIMICBERTReadmissionPredictor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch import cuda, save
import os
from time import time


def run(txtvar, st_aug, data_dir, logdir, n_gpus):
    msg1 = 'Training {} model on {} variable'.format(args.bert_model, txtvar)
    test_outpath = args.bert_model+txtvar
    if st_aug:
        msg1 += ' with semantic types'
        test_outpath += '_st'
    test_outpath += '.txt'
    print('=========')
    print(msg1)
    s = time()
    model = MIMICBERTReadmissionPredictor(
        n_train_fp=os.path.join(data_dir, 'notes_train_seeded.csv'),
        r_train_fp=os.path.join(data_dir, 'readmission_train_seeded.csv'),
        n_test_fp=os.path.join(data_dir, 'notes_test_seeded.csv'),
        r_test_fp=os.path.join(data_dir, 'readmission_test_seeded.csv'),
        epochs=4,
        val_frac=0.2,
        batch_size=args.batch,
        lr=0.01,
        momentum=0.5,
        bert_model=args.bert_model,
        txtvar=txtvar,
        seqlen=512,
        st_aug=st_aug,
        db=args.debug,
        write_test_results_to=os.path.join(logdir, test_outpath),
        verbose=args.verbose
    )
    trainer = Trainer(
        default_root_dir=logdir,
        gpus=(n_gpus if cuda.is_available() else 0),
        max_epochs=1 if args.debug else args.epochs,
        logger=(TensorBoardLogger(logdir, name='tb') if args.log else None),
        fast_dev_run=args.debug,
        distributed_backend='ddp',
        accumulate_grad_batches=args.grad_accum
    )
    print('GPUs used;')
    if cuda.is_available and n_gpus > 0:
        for i in range(n_gpus):
            print(cuda.get_device_name(i))
    else:
        print('None')
    trainer.fit(model)

    print('Pickling model...')
    out = 'mimic_'+args.bert_model
    if st_aug: out += '_st'
    save(model.state_dict(), os.path.join(data_dir, out+'.pt'))

    trainer.test(model)

    e = time()
    print('Training, pickling & testing time: {:.4f}'.format(e-s))
    print('=========\n\n')


def main():
    # TODO: make these into cl args
    data_dir = '/home/mrim/manniona/data/mimic_experiment_writes/tenpercentsplit'
    logdir = '/home/mrim/manniona/data/bertmodel_logs'

    if args.gpus == -1 and not args.debug:
        n_gpus = cuda.device_count()
    elif args.debug:
        n_gpus = 0
    else:
        n_gpus = args.gpus

    run(
        txtvar=args.txtvar,
        st_aug=args.st_aug,
        data_dir=data_dir,
        logdir=logdir,
        n_gpus=n_gpus
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bert_model', type=str)
    parser.add_argument('txtvar', type=str)
    parser.add_argument('--st_aug', action='store_true')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    global args
    args = parser.parse_args()

    main()
