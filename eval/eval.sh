# python eval.py --resnet_shortcut A --feature_size 8192 --embedding_size 512 --lstm_memory 512 --bs 32 --batch_size 32 --langmethod Transformer --vocabpath vocab_pad.json \
# --submission_path submission_train.json --model_ep 114
# python eval.py --resnet_shortcut A --feature_size 2048 --embedding_size 512 --lstm_memory 512 --bs 16 --batch_size 16 --langmethod Transformer --vocabpath vocab_pad.json \
# --clip_len 64 --modelname resnext --modeldepth 101 --root_path /ssd2/dsets/activitynet_captions --meta_path videometa_val.json --mode val --n_classes 400 --submission_path submission_val.json --model_ep 7
# python eval.py --resnet_shortcut A --feature_size 2048 --embedding_size 512 --lstm_memory 512 --bs 16 --batch_size 16 --langmethod Transformer --vocabpath vocab_pad.json \
# --clip_len 64 --modelname resnext --modeldepth 101 --root_path /ssd2/dsets/activitynet_captions --meta_path videometa_val.json --mode val --n_classes 400 --submission_path submission_val2.json --model_ep 7
# python eval.py --resnet_shortcut A --feature_size 2048 --embedding_size 512 --lstm_memory 512 --bs 8 --batch_size 16 --langmethod Transformer --vocabpath vocab_pad.json \
# --clip_len 64 --modelname resnext --modeldepth 101 --root_path /ssd2/dsets/activitynet_captions --meta_path videometa_val.json --mode val --n_classes 400 --submission_path submission_val3.json --model_ep 7

# python eval.py --resnet_shortcut A --feature_size 2048 --embedding_size 512 --lstm_memory 512 --bs 8 --batch_size 8 --langmethod Transformer --vocabpath vocab_pad.json \
# --clip_len 64 --modelname resnext --modeldepth 101 --root_path /ssd2/dsets/activitynet_captions --meta_path videometa_val.json --mode val --n_classes 400 --submission_path submission_val3.json --model_ep 2

python eval.py --resnet_shortcut A --feature_size 2048 --embedding_size 512 --lstm_memory 512 --bs 16 --batch_size 16 --langmethod Transformer --vocabpath vocab_pad.json \
--clip_len 64 --modelname resnext --modeldepth 101 --root_path /ssd2/dsets/activitynet_captions --meta_path videometa_test.json --mode test --n_classes 400 --submission_path submission_test.json --model_ep 1
