##################################################
# Imports
##################################################

import sys
import pytorch_lightning as pl
from vit import Classifier

from config import get_args
from dataloaders import get_dataloaders

def get_trainer(args):
    if args.mode == 'train':
        trainer = pl.Trainer(gpus=1, max_epochs=args.epochs)

    elif args.mode == 'validation':
        trainer = pl.Trainer(max_epochs=args.epochs)

    elif args.mode == 'test':
        trainer = pl.Trainer(max_epochs=args.epochs)

    else:
        raise Exception(f'Mode "{args.mode}" not supported.')

    return trainer

def main(args):

    # Dataloaders
    dls = get_dataloaders(args)

    # Model
    model = Classifier(args)

    # Trainer
    trainer = get_trainer(args)

    if args.mode == 'train':
        trainer.fit(model, dls['train_aug'], dls['validation'])
        trainer.test(model=None, test_dataloaders=dls['validation'])

    elif args.mode == 'validation':
        trainer.test(model=model, test_dataloaders=dls['validation'])

    elif args.mode == 'test':
        trainer.test(model=model, test_dataloaders=dls['validation'])

    else:
        raise Exception(f'Mode "{args.mode}" not supported.')

if __name__ == '__main__':

    # Args
    args = get_args(sys.stdin)
    print(args)

    # Run main
    main(args)
