python eval.py --resnet_shortcut A --feature_size 8192 --embedding_size 512 --lstm_memory 512 --bs 32 --batch_size 32 --langmethod Transformer --vocabpath vocab_pad.json --model_ep 17
# python eval.py --resnet_shortcut A --feature_size 8192 --embedding_size 512 --lstm_memory 512 --bs 32 --langmethod Transformer --vocabpath vocab_pad.json \
# --root_path /ssd2/dsets/activitynet_captions --meta_path videometa_val.json --mode val --model_ep 1000
#python eval.py --resnet_shortcut A --feature_size 8192 --embedding_size 512 --lstm_memory 512 --bs 32 --langmethod Transformer --vocabpath vocab_pad.json \
#--root_path /ssd2/dsets/activitynet_captions --meta_path videometa_test.json --mode test --model_ep 1000

