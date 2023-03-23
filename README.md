# FNDG
Fake News Detection using Graph

## Run BERT classification on server
` conda activate hcmut_env`

`nohup python train_bert_mlp.py --batch_size 8 --dataset 'Liar' > ../Result/liar_log.txt`

`python eval_fakeBERT.py --dataset 'FND'`
