python main.py --reweight=1 --log_name=log --round=1 --dataset=ml-1m_noisy --data_path=../datasets/ml-1m_noisy/
!python main.py --n_cate 1 --cuda --lamda 0.05 --dataset=ml-1m_clean --data_path=../datasets/ml-1m_clean/ --batch_size=400  --emb_size=300 --noise_scale 0.1 --mean_type=eps --steps=10 --noise_min=0.01 --noise_max=0.05 --sampling_steps=0 --reweight=1 --log_name=log --round=1 --gpu=0
!python main.py --n_cate 1 --cuda --lamda 0.05 --dataset=ml-1m_clean --data_path=../datasets/ml-1m_clean/ --batch_size=400  --emb_size=32 --noise_scale 0.1 --mean_type=eps --steps=10 --noise_min=0.1 --noise_max=0.9 --sampling_steps=0 --reweight=1 --log_name=log --round=1 --gpu=0
# !python main.py --lamda 0.01 --dataset=ml-1m_clean --data_path=../datasets/ml-1m_clean/ --batch_size=400  --emb_size=32 --noise_scale 0.001 --mean_type=eps --steps=5 --noise_min=0.005 --noise_max=0.02 --sampling_steps=0 --reweight=1 --log_name=log --round=1 --cuda --gpu=0
# res at epoch 140= [Test]: Precision: 0.0516-0.0437-0.0315-0.0228 Recall: 0.1007-0.1675-0.2817-0.3799 NDCG: 0.0829-0.1048-0.1412-0.1694 MRR: 0.1398-0.1517-0.1584-0.16

# !python main.py --lamda 0.01 --dataset=ml-1m_clean --data_path=../datasets/ml-1m_clean/ --batch_size=400  --emb_size=10 --noise_scale 0.001 --mean_type=eps --steps=40 --noise_min=0.005 --noise_max=0.02 --sampling_steps=0 --reweight=1 --log_name=log --round=1 --cuda --gpu=0
# recall = 
# python main.py --lamda 0.03 --dataset=ml-1m_clean --data_path=../datasets/ml-1m_clean/ --batch_size=400  --emb_size=32 --noise_scale 0.001 --mean_type=eps --steps=40 --noise_min=0.005 --noise_max=0.02 --sampling_steps=0 --reweight=1 --log_name=log --round=1 --cuda --gpu=0

# python main.py --tst_w_val --dataset=ml-1m_clean --data_path=../datasets/ml-1m_clean/ --batch_size=400  --emb_size=10 --noise_scale=0.005 --mean_type=eps --steps=40 --noise_min=0.005 --noise_max=0.02 --sampling_steps=0 --reweight=1 --log_name=log --round=1 --gpu=0 --lr1=0.001 --lamda=0.03 --cuda
# best model