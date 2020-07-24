import argparse
import warnings
from lib import MIMICWord2VecReadmissionPredictor
from os.path import join


def main(args):
    print('================')
    if args.data_dir is not None:
        for fparg in ['train_txt_fp', 'test_txt_fp', 'train_readm_fp',
                      'test_readm_fp', 'out_fp', 'model_fp']:
            args.__setattr__(fparg, join(args.data_dir, args.__getattribute__(fparg)))

    model = MIMICWord2VecReadmissionPredictor(
        txtvar=args.txtvar,
        st_aug=args.st,
        db=args.db
    )

    print('Running Parameter Grid Search...\n')
    model.choose_params(
        args.train_txt_fp,
        args.train_readm_fp,
        n_jobs=args.workers,
        use_multithreading=args.multithread
    )

    print('Training on patient dataset...\n')
    model.train()

    print('Testing...\n')
    model.test(
        args.test_text_fp,
        args.test_readm_fp,
        args.out_fp,
        args.model_fp
    )

    print('Done!')
    if args.out_fp is not None:
        print('see {} for results'.format(args.out_fp))
    print('================')


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('txtvar', type=str)
    parser.add_argument('-st', action='store_true')
    parser.add_argument('-data_dir', type=str)
    parser.add_argument('train_txt_fp', type=str)
    parser.add_argument('test_txt_fp', type=str)
    parser.add_argument('train_readm_fp', type=str)
    parser.add_argument('test_readm_fp', type=str)
    parser.add_argument('-db', action='store_true')
    parser.add_argument('-out_fp', type=str)
    parser.add_argument('-model_fp', type=str)
    parser.add_argument('-workers', type=int, default=-1)
    parser.add_argument('-multithread', action='store_true')

    main(parser.parse_args())
