import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument('--root_path', type=str, default='/ssd1/dsets/activitynet_captions')
    parser.add_argument('--model_path', type=str, default='../models', help='Path to read models from when training / testing')
    parser.add_argument('--model_save_path', type=str, default='../models', help='Path to save models to when training')
    parser.add_argument('--framepath', type=str, default='frames')
    parser.add_argument('--annpaths', nargs='+', type=str, default=['train_fps.json', 'val_1_fps.json', 'val_2_fps.json'], help='Path to build vocabulary')

    # configurations of 3D CNN
    parser.add_argument('--enc_pretrain_path', type=str, help='Pretrained model of feature extracting module (.pth)')
    parser.add_argument('--norm_value', default=1, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--cnn_name', default='resnet', type=str, help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--cnn_depth', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--conv1_t_size', default=7, type=int, help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride', default=1, type=int, help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool', action='store_true', help='If true, the max pooling after conv1 is removed.')

    # configurations of captioning module
    parser.add_argument('--dec_pretrain_path', type=str, help='Pretrained model of captioning module (.pth)')
    parser.add_argument('--rnn_name', type=str, default='LSTM')
    parser.add_argument('--rnn_layers', type=int, default=3)
    parser.add_argument('--rnn_dropout', type=int, default=0.1)
    parser.add_argument('--max_seqlen', type=int, default=30)
    parser.add_argument('--lstm_memory', type=int, default=512)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--emb_init', type=str, default='../wordvectors/glove.6B.300d.txt')

    # training config
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--token_level', action='store_true')
    parser.add_argument('--n_cpu', type=int, default=0)
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--single_gpu', dest='dataparallel', action='store_false')

    # hyperparams
    parser.add_argument('--feature_size', type=int, default=512)
    parser.add_argument('--min_freq', type=int, default=5)
    parser.add_argument('--imsize', type=int, default=224)
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)

    # evaluation config
    parser.add_argument('--submission_path', type=str, default='submission.json')
    parser.add_argument('--json_path', type=str, default=None)

    args = parser.parse_args()

    return args
