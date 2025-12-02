lambda_val=200
nb_perturbations=6

python src/train.py \
--out_path "./results/pd_01/" \
--gpu_id '0' \
--epochs 300 \
--learning_rate 0.0005 \
--eps 0.0001 \
--nb_rewightings 1 \
--save_freq 20 \
--nb_perturbations $nb_perturbations \
--lambda_val $lambda_val