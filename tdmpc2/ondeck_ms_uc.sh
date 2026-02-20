for i in {9..12}; do
    echo "**************Run $i***************"
    python train.py task=pick-cube model_size=5 wandb_project=tdmpc2 wandb_entity=jambotime uncertainty_weight=true seed=$i &
    sleep 1
    python train.py task=pick-cube model_size=5 wandb_project=tdmpc2 wandb_entity=jambotime seed=$((i*10))
    sleep 60m
    python train.py task=pick-cube model_size=5 wandb_project=tdmpc2 wandb_entity=jambotime uncertainty_weight=true seed=$((i*100)) obs=rgb &
    sleep 1
    python train.py task=pick-cube model_size=5 wandb_project=tdmpc2 wandb_entity=jambotime seed=$((i*1000)) obs=rgb
done