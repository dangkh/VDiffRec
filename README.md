# VDiffRec
 
python  main.py  --dataset=ml-1m_clean --data_path=../datasets/ml-1m_clean/ --batch_size=400  --emb_size=10 --noise_scale=0.01 --mean_type=eps --steps=100 --noise_min=0.005 --noise_max=0.01 --sampling_steps=10 --reweight=1 --log_name=log --round=1 --lr1=0.0003 --lamda=0.03 

python  main.py  --dataset=ml-1m_clean --data_path=../datasets/ml-1m_clean/ --batch_size=400  --emb_size=10  --log_name=log --round=1 --lr1=0.003

python  main.py  --cuda --dataset=baby --data_path=../datasets/baby/ --batch_size=400  --emb_size=10  --log_name=log --round=1 --lr1=0.0003 --mean_type=eps