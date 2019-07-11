# nohup python train.py --lstm_stacks 1 --max_epochs 50 --vocabpath vocab.json --lstm_pretrain_ep 0 > ../logs/word_none_langpretrain0_resnet10_noinit.out &!
# nohup python train.py --lstm_stacks 1 --max_epochs 50 --token_level --vocabpath vocab_char.json --lstm_pretrain_ep 0 --max_seqlen 200 > ../logs/char_langpretrain0_resnet10_noinit.out &!
# nohup python train.py --lstm_stacks 1 --max_epochs 50 --lstm_pretrain_ep 10 > ../logs/word_none_langpretrain10_resnet10_noinit.out
# nohup python train.py --lstm_stacks 1 --max_epochs 50 --token_level --vocabpath vocab_char.json --lstm_pretrain_ep 10 --max_seqlen 200 --lr 8e-5 --bs 64 > ../logs/char_none_langpretrain10_resnet10_noinit.out &!
# nohup python train.py --lstm_stacks 1 --max_epochs 500 --token_level --vocabpath vocab_char.json --lstm_pretrain_ep 50 --max_seqlen 150 --lr 1e-4 --bs 64 > ../logs/char_none_langpretrain50_resnet10_noinit.out &!
# failed to progress from the an the an ...
# nohup python train.py --max_epochs 1000 --token_level --vocabpath vocab_char.json --lstm_pretrain_ep 100 --max_seqlen 200 --lr 8e-5 --bs 64 > ../logs/char_none_langpretrain100_resnet10_noinit.out &!

#nohup python train.py --max_epochs 1000 --token_level --vocabpath vocab_char.json --lstm_pretrain_ep 100 --max_seqlen 200 --lr 8e-5 --bs 64 --emb_init ../wordvectors/glove.840B.300d-char.txt \
#--embedding_size 300 > ../logs/char_glove_langpretrain100_resnet10_noinit.out &!

# 150 epochs, <BOS> a man is seen a and the camera <EOS>
# nohup python train.py --max_epochs 1000 --vocabpath vocab.json --lstm_pretrain_ep 0 --max_seqlen 30 --lr 1e-4 --bs 32 --emb_init ../wordvectors/glove.840B.300d-char.txt \
# --embedding_size 300 > ../logs/word_glove_langpretrain0_resnet10_noinit.out &!

#nohup python train.py --max_epochs 1000 --vocabpath vocab.json --lstm_pretrain_ep 10 --max_seqlen 30 --lr 1e-4 --bs 32 --feature_size 8192 --emb_init ../wordvectors/glove.840B.300d-char.txt \
#--embedding_size 300 --modelname resnet --modeldepth 18 --resnet_shortcut A --pretrain_path ../models/resnet-18-kinetics.pth > ../logs/word_glove_langpretrain10_resnet18.out &!

# nohup python train.py --max_epochs 1000 --vocabpath vocab.json --lstm_pretrain_ep 0 --max_seqlen 30 --lr 1e-4 --bs 32 --feature_size 8192 --emb_init ../wordvectors/glove.840B.300d-char.txt \
# --embedding_size 300 --modelname resnet --modeldepth 18 --resnet_shortcut A --pretrain_path ../models/resnet-18-kinetics.pth > ../logs/word_glove_langpretrain10_resnet18.out &!

# nohup python train.py --max_epochs 1000 --vocabpath vocab_pad.json --lstm_pretrain_ep 0 --max_seqlen 30 --lr 1e-4 --bs 32 --feature_size 8192  \
# --langmethod Transformer --embedding_size 512 --lstm_memory 512 --modelname resnet --modeldepth 18 --resnet_shortcut A \
# --clip_len 16 --pretrain_path ../models/resnet-18-kinetics.pth > ../logs/word_transpretrain0_resnet18.out &!

# python train.py --max_epochs 1000 --vocabpath vocab_pad.json --lstm_pretrain_ep 0 --max_seqlen 30 --lr 1e-4 --bs 8 --feature_size 8192  \
# --langmethod Transformer --embedding_size 512 --lstm_memory 512 --modelname resnet --modeldepth 18 --resnet_shortcut A \
# --clip_len 64 --pretrain_path ../models/resnet-18-kinetics.pth

# python train.py --max_epochs 1000 --vocabpath vocab_pad.json --lstm_pretrain_ep 0 --max_seqlen 30 --lr 1e-4 --bs 8 --feature_size 2048  \
# --langmethod Transformer --embedding_size 512 --lstm_memory 512 --modelname resnext --modeldepth 101 --resnet_shortcut A \
# --clip_len 64 --pretrain_path ../models/save_105.pth

# python train.py --max_epochs 200 --vocabpath vocab_pad.json --lstm_pretrain_ep 0 --max_seqlen 30 --lr 1e-4 --bs 16 --feature_size 2048  \
# --langmethod Transformer --embedding_size 512 --lstm_memory 512 --modelname resnext --modeldepth 101 --resnet_shortcut A --n_cpu 8 \
# --weight_decay 1e-4 --clip_len 64 --pretrain_path ../models/save_105.pth --start_from_ep 0 > ../logs/honban_bs8_wd1e-4_lr1e-4.out

# python train.py --max_epochs 200 --vocabpath vocab_pad.json --lstm_pretrain_ep 0 --max_seqlen 30 --lr 1e-3 --bs 8 --feature_size 2048  \
# --langmethod LSTM --embedding_size 300 --lstm_memory 512 --modelname resnext --modeldepth 101 --resnet_shortcut A --n_cpu 8 \
# --weight_decay 1e-4 --emb_init ../wordvectors/glove.6B.300d.txt --clip_len 64 --pretrain_path ../models/save_105.pth --start_from_ep 0

# python train.py --max_epochs 1000 --vocabpath vocab_pad.json --lstm_pretrain_ep 0 --max_seqlen 30 --lr 1e-4 --bs 8 --feature_size 2048  \
# --langmethod Transformer --embedding_size 512 --lstm_memory 512 --modelname resnext --modeldepth 101 --resnet_shortcut A --n_cpu 8 \
# --weight_decay 1e-4 --clip_len 64 --pretrain_path ../models/save_105.pth --start_from_ep 0

# for debugging
# python train.py --max_epochs 1000 --vocabpath vocab_pad.json --lstm_pretrain_ep 0 --max_seqlen 30 --lr 1e-4 --bs 64 --feature_size 8192  \
# --langmethod Transformer --embedding_size 512 --lstm_memory 512 --modelname resnet --modeldepth 18 --resnet_shortcut A --pretrain_path ../models/resnet-18-kinetics.pth

python train_lstm.py \
--cnn_name resnext --cnn_depth 101 --rnn_name LSTM --rnn_layers 3 --max_seqlen 30 --annpaths train_fps.json val_1_fps.json val_2_fps.json \
--emb_init ../wordvectors/glove.6B.300d.txt --max_epochs 20 --n_cpu 8 --feature_size 2048 --lstm_memory 512 --embedding_size 300 \
--min_freq 5 --imsize 224 --clip_len 32 --batch_size 8 --lr 1e-3 --momentum 0.9 --weight_decay 1e-4 --patience 10 \
--enc_pretrain_path ../models/save_105.pth \
> ../logs/config1.log 2>&1


# --- possible arguments ---
#     # configurations of dataset (paths)
#     parser.add_argument('--root_path', type=str, default='/ssd1/dsets/activitynet_captions')
#     parser.add_argument('--model_path', type=str, default='../models', help='Path to read models from when training / testing')
#     parser.add_argument('--model_save_path', type=str, default='../models', help='Path to save models to when training')
#     parser.add_argument('--framepath', type=str, default='frames')
#     parser.add_argument('--annpaths', nargs='+', type=str, default=['train.json', 'val_1.json', 'val_2.json'], help='Path to build vocabulary')
# 
#     # configurations of 3D CNN
#     parser.add_argument('--enc_pretrain_path', default='', type=str, help='Pretrained model of feature extracting module (.pth)')
#     parser.add_argument('--norm_value', default=1, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
#     parser.add_argument('--cnn_name', default='resnet', type=str, help='(resnet | preresnet | wideresnet | resnext | densenet | ')
#     parser.add_argument('--cnn_depth', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
#     parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
#     parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
#     parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
#     parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
# 
#     # configurations of captioning module
#     parser.add_argument('--dec_pretrain_path', default='', type=str, help='Pretrained model of captioning module (.pth)')
#     parser.add_argument('--rnn_name', type=str, default='LSTM')
#     parser.add_argument('--rnn_layers', type=int, default=3)
#     parser.add_argument('--max_seqlen', type=int, default=30)
#     parser.add_argument('--emb_init', type=str, default='../wordvectors/glove.6B.300d.txt')
# 
#     # training config
#     parser.add_argument('--max_epochs', type=int, default=20)
#     parser.add_argument('--log_every', type=int, default=10)
#     parser.add_argument('--token_level', action='store_true')
#     parser.add_argument('--n_cpu', type=int, default=8)
#     parser.add_argument('--no_cuda', dest='cuda', action='store_false')
#     parser.add_argument('--single_gpu', dest='dataparallel', action='store_false')
# 
#     # hyperparams
#     parser.add_argument('--feature_size', type=int, default=512)
#     parser.add_argument('--lstm_memory', type=int, default=512)
#     parser.add_argument('--embedding_size', type=int, default=512)
#     parser.add_argument('--min_freq', type=int, default=5)
#     parser.add_argument('--imsize', type=int, default=224)
#     parser.add_argument('--clip_len', type=int, default=16)
#     parser.add_argument('--batch_size', type=int, default=1)
#     parser.add_argument('--lr', type=float, default=1e-2)
#     parser.add_argument('--momentum', type=float, default=0.9)
#     parser.add_argument('--weight_decay', type=float, default=1e-4)
#     parser.add_argument('--patience', type=int, default=10)
# 
#     # evaluation config
#     parser.add_argument('--submission_path', type=str, default='submission.json')
#     parser.add_argument('--json_path', type=str, default=None)
