import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI for handling ML operations')

    parser.add_argument('mode', metavar='mode', type=str, help='mode with which to run')

    parser.add_argument('--epochs', metavar='epochs', type=int, default=2, help='num epochs')
    parser.add_argument('--batch_size', metavar='batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--lr', metavar='lr', type=float, default=0.001, help='learning rate')

    args = parser.parse_args()

    if args.mode == 'train_cnn_cifar10':
        print('[*] Training SimpleCNN On Cifar10')
        from ml.trainer.cifar10 import train_cnn_cifar10

        # leave the model so you can use it after training finishes
        model = train_cnn_cifar10(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    elif args.mode == 'test_distilbert_tokenizer':
        print('[*] Testing DistilBERT Tokenizer')
        
    elif args.mode == 'train_clip':
        print('training clip mode')
    else:
        print('mode not recognized')
