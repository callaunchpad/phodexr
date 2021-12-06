import argparse
import torch

torch.manual_seed(17)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI for handling ML operations')

    parser.add_argument('mode', metavar='mode', type=str, help='mode with which to run')

    parser.add_argument('--epochs', metavar='epochs', type=int, default=2, help='num epochs')
    parser.add_argument('--batch_size', metavar='batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--lr', metavar='lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--optimizer', metavar='optimizer', type=str, default='adam', help='optimizer to use for the run')
    parser.add_argument('--decay_epoch', metavar='decay_epoch', type=int, default=5, help='num epochs until decay lr')
    parser.add_argument('--vision_weights', metavar='vision_weights', type=str, default='', help='path to vision weights')
    parser.add_argument('--nlp_weights', metavar='nlp_weights', type=str, default='', help='path to nlp weights')
    parser.add_argument('--unfreeze_nlp', dest='unfreeze_nlp', action='store_true', help='unfreeze nlp weights')
    parser.set_defaults(unfreeze_nlp=False)
    parser.add_argument('--debug', dest='debug', action='store_true', help='use debug set')
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
        nlp_head, vision_head, nlp_optim, vision_optim = train_clip_mcoco_baseline(epochs=args.epochs,
                                                                                    batch_size=args.batch_size,
                                                                                    learning_rate=args.lr,
                                                                                    optimizer=args.optimizer,
                                                                                    decay_epoch=args.decay_epoch,
                                                                                    vision_weights=args.vision_weights,
                                                                                    nlp_weights=args.nlp_weights,
                                                                                    unfreeze_nlp=args.unfreeze_nlp,
                                                                                    debug=args.debug)
    elif args.mode == 'test_clip':
        print('[*] Testing CLIP')
        from ml.testing.test_clip import test_clip
        
        res = test_clip()
    else:
        print('mode not recognized')

def save_model_optim(model, optimizer, path):
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict() 
    }, path)