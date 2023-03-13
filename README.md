# FNDG
Fake News Detection using Graph

## Run BERT classification
` conda activate hcmut_env`

Local: `python bert_mlp.py --epoch 1 --token_length 512 --batch_size 16 --dataset 'FND' --lr 2e-5 > ../Result/log.txt`

Server: `nohup python bert_mlp.py --epoch 1 --token_length 512 --batch_size 16 --dataset 'FND' --lr 2e-5 > ../Result/log.txt`
