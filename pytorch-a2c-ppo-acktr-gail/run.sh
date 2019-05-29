python main.py --env-name "Driving-v2" --algo ppo --use-gae\
 --log-interval 1 --num-steps 1000 --num-processes 16 --lr 3e-4 --entropy-coef 0\
 --value-loss-coef 0.5 --ppo-epoch 15 --num-mini-batch 4 --gamma 0.99\
 --gae-lambda 0.95 --num-env-steps 16000000 --use-linear-lr-decay\
 --use-proper-time-limits
