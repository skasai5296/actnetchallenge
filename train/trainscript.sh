# nohup python train.py --lstm_stacks 1 --max_epochs 50 --vocabpath vocab.json --lstm_pretrain_ep 0 > ../logs/default.out &!
# nohup python train.py --lstm_stacks 1 --max_epochs 50 --token_level --vocabpath vocab_char.json --lstm_pretrain_ep 0 --max_seqlen 200 > ../logs/char_default.out &!
# nohup python train.py --lstm_stacks 1 --max_epochs 50 --lstm_pretrain_ep 10 > ../logs/default_pretrain10.out
# success an a an a the an a ...
# nohup python train.py --lstm_stacks 1 --max_epochs 50 --token_level --vocabpath vocab_char.json --lstm_pretrain_ep 10 --max_seqlen 200 --lr 8e-5 --bs 64 > ../logs/char_default_pretrain10.out &!
# nohup python train.py --lstm_stacks 1 --max_epochs 500 --token_level --vocabpath vocab_char.json --lstm_pretrain_ep 50 --max_seqlen 150 --lr 1e-4 --bs 64 > ../logs/char_lstm2_pretrain50.out &!
nohup python train.py --max_epochs 1000 --token_level --vocabpath vocab_char.json --lstm_pretrain_ep 100 --max_seqlen 200 --lr 8e-5 --bs 64 > ../logs/char_default_pretrain100.out &!
