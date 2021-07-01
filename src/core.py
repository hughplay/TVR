import argparse

import pipeline
from dataset import preprocess_parser, preprocess
from utils.config import Config
from utils.watch import watch_parser, watch


def prepare_parser():
    parser = argparse.ArgumentParser(
        description='The core script of experiment management.')
    subparsers = parser.add_subparsers(dest='command')

    add_preprocess_parser(subparsers)
    add_train_parser(subparsers)
    add_test_parser(subparsers)
    add_watch_parser(subparsers)

    return parser


def add_preprocess_parser(subparsers):
    parser = subparsers.add_parser('preprocess', description='Preprocessing')
    preprocess_parser(parser)


def add_train_parser(subparsers):
    parser = subparsers.add_parser('train', description='Training.')
    parser.add_argument('config', help='The path of a config file.')
    parser.add_argument(
        '--device', default=argparse.SUPPRESS, help='Training device.')
    parser.add_argument(
        '--ckpt', default='latest',
        help='Continue training from this checkpoint.')
    parser.add_argument(
        '--pretrain', action='store_true',
        help='Whether use a checkpoint as a pretrained model')
    parser.add_argument(
        '--restart', action='store_true', help='Training from start.')
    parser.add_argument(
        '--test', action='store_true', help='Testing after training done.')
    parser.add_argument(
        '--test_ckpt', default='best', help='Use this checkpoint for testing')
    parser.add_argument(
        '--parallel', action='store_true', help='Do not use this argument.')


def train(args):
    config = Config(args.config)
    if args.parallel:
        if not config.runner.startswith('Parallel'):
            config.runner = 'Parallel' + config.runner
    else:
        if config.runner.startswith('Parallel'):
            config.runner = config.runner[len('Parallel'):]
    config.merge_args(args)
    Runner = getattr(pipeline, config.runner)
    r = Runner(config)
    r.init_as_trainer(restart=args.restart)
    r.train(args.ckpt, pretrain=args.pretrain)
    if args.test:
        r.init_as_tester()
        r.test(args.test_ckpt)


def add_test_parser(subparsers):
    parser = subparsers.add_parser('test', description='Testing.')
    parser.add_argument('config', help='The path of a config file.')
    parser.add_argument(
        '--device', default=argparse.SUPPRESS, help='Training device.')
    parser.add_argument(
        '--ckpt', default='best', help='Use this checkpoint for testing.')


def test(args):
    config = Config(args.config)
    config.merge_args(args)
    Runner = getattr(pipeline, config.runner)
    r = Runner(config)
    r.init_as_tester()
    r.test(args.ckpt)


def add_watch_parser(subparsers):
    parser = subparsers.add_parser(
        'watch', description='Waiting for execution.')
    watch_parser(parser)


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    if args.command == 'preprocess':
    preprocess(
        args.dataroot, args.force, args.split, args.split_name, args.width,
        args.height)
    elif args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'watch':
        watch(args)
    else:
        pass
