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

nohup python train.py --max_epochs 1000 --vocabpath vocab_pad.json --lstm_pretrain_ep 0 --max_seqlen 30 --lr 1e-4 --bs 64 --feature_size 8192  \
--langmethod Transformer --embedding_size 512 --lstm_memory 512 --modelname resnet --modeldepth 18 --resnet_shortcut A --pretrain_path ../models/resnet-18-kinetics.pth > ../logs/word_glove_transpretrain0_resnet18.out &!
