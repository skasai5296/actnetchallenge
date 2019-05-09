import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_classes',
        default=400,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)'
    )
    parser.add_argument(
        '--n_finetune_classes',
        default=400,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
    )
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--ft_begin_index',
        default=0,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--norm_value',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--modelname',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--modeldepth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--root_path', type=str, default='/ssd1/dsets/activitynet_captions')
    parser.add_argument('--model_path', type=str, default='../models')
    parser.add_argument('--meta_path', type=str, default='videometa_train.json')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--framepath', type=str, default='frames')
    parser.add_argument('--annpath', type=str, default='train.json')
    parser.add_argument('--cnnmethod', type=str, default='resnet')
    parser.add_argument('--rnnmethod', type=str, default='LSTM')
    parser.add_argument('--vocabpath', type=str, default='vocab.json')
    parser.add_argument('--start_from_ep', type=int, default=0)
    parser.add_argument('--lstm_pretrain_ep', type=int, default=20)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--lstm_stacks', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=18)
    parser.add_argument('--imsize', type=int, default=224)
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--lstm_memory', type=int, default=512)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--emb_init', type=str, default='../wordvectors/glove.6B.300d.txt')
    parser.add_argument('--feature_size', type=int, default=512)
    parser.add_argument('--max_seqlen', type=int, default=30)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=int, default=0.9)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--token_level', action='store_true')
    parser.add_argument('--cuda', action='store_false')
    parser.add_argument('--dataparallel', action='store_false')

    args = parser.parse_args()

    return args
