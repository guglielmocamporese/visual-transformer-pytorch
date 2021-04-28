##################################################
# Imports
##################################################

import argparse
import json

def get_args(stdin):

    parser = argparse.ArgumentParser(stdin)
    parser.add_argument('--patch_size', type=int, default=4, help='The size of the input patches.')
    #parser.add_argument('--num_layers', type=int, default=2, help='The number of layers in the encoder.')
    #parser.add_argument('--h_dim', type=int, default=256, help='The hidden dimensionality of the model.')
    #parser.add_argument('--num_heads', type=int, default=8, help='The number of the heads in the multihead attention.')
    parser.add_argument('--num_classes', type=int, default=10, help='The number of classes in the dataset.')
    #parser.add_argument('--d_ff', type=int, default=2048, help='The dimensionality of the feed forward layer.')
    parser.add_argument('--max_time_steps', type=int, default=1000, help='Maximum number of tme steps of the model.')
    parser.add_argument('--use_clf_token', type=bool, default=True, help='Whether to use the class token in the model.')
    parser.add_argument('--batch_size', type=int, default=256, help='The batch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of workers.')
    parser.add_argument('--model_checkpoint', type=str, default='', help='The model checkpoint path.')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs for the training.')
    parser.add_argument('--lr', type=float, default=3e-3, help='The learning rate.')
    parser.add_argument('--mode', type=str, default='validation', help='The mode of the experiment, in "train", "validation" or "test".')
    parser.add_argument('--model', type=str, default='vit-base', help='The model architecture, in "vit-base", "vit-large", "vit-huge".')
    parser.add_argument('--dropout', type=float, default=0.1, help='The dropout rate.')
    parser.add_argument('--dropout_emb', type=float, default=0.1, help='The dropout rate after the positional embedding.')
    args = parser.parse_args()

    args.__class__.__repr__ = lambda x: 'Input args: ' + json.dumps(x.__dict__, indent=4)
    return args
