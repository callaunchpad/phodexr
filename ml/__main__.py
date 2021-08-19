import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI for handling ML operations')

    parser.add_argument('mode', metavar='mode', type=str, help='mode with which to run')

    parser.add_argument('--batch_size', metavar='batch_size', type=int, help='batch size')
    parser.add_argument('--epochs', metavar='epochs', type=int, help='num epochs')
    parser.add_argument('--lr', metavar='lr', type=int, help='learning rate')

    args = parser.parse_args()

    if args.mode == 'train_cnn':
        print('training cnn mode')
    elif args.mode == 'train_clip':
        print('training clip mode')
    else:
        print('mode not recognized')
