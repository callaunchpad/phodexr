import argparse

parser = argparse.ArgumentParser(description='CLI for handling ML operations')

parser.add_argument('mode', metavar='mode', type=str, help='mode with which to run')

args = parser.parse_args()

if args.mode == 'train_cnn':
    print('training cnn mode')
elif args.mode == 'train_clip':
    print('training clip mode')
else:
    print('mode not recognized')