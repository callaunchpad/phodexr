import argparse
import torch

torch.manual_seed(17)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI for handling ML operations')

    parser.add_argument('mode', metavar='mode', type=str, help='mode with which to run')

    parser.add_argument('--epochs', metavar='epochs', type=int, default=2, help='num epochs')
    parser.add_argument('--batch_size', metavar='batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--lr', metavar='lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--debug', dest='debug', action='store_true', help='use_debug_set')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    if args.mode == 'train_cnn_cifar10':
        print('[*] Training SimpleCNN On Cifar10')
        from ml.trainer.cifar10 import train_cnn_cifar10

        # leave the model so you can use it after training finishes
        model = train_cnn_cifar10(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    elif args.mode == 'test_distilbert_tokenizer':
        print('[*] Testing DistilBERT Tokenizer')
        from ml.testing.distilbert_tokenizer import test_distilbert_tokenizer

        test_distilbert_tokenizer(epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == 'train_resnet_cifar10':
        print('[*] Training RESNET on Cifar10')
        from ml.trainer.cifar10 import train_resnet_cifar10

        model = train_resnet_cifar10(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    elif args.mode == 'train_clip_baseline':
        print('[*] Training CLIP mode')
        from ml.trainer.clip_mcoco_baseline import train_clip_mcoco_baseline

        # leave the model so we can use it after training finishes
        nlp_head, vision_head = train_clip_mcoco_baseline(epochs=args.epochs,
                                                          batch_size=args.batch_size,
                                                          learning_rate=args.lr,
                                                          debug=args.debug)
    elif args.mode == 'test_clip':
        print('[*] Testing CLIP')
        from ml.testing.test_clip import test_clip
        
        res = test_clip()
    else:
        print('mode not recognized')
