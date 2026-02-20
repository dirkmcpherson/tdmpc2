for i in {3..8}; do
    echo "**************Run $i***************"
    python train.py task=pick-cube model_size=5 wandb_project=tdmpc2 wandb_entity=jambotime demo_dir=demonstrations/motionplanning/ seed=$i &
    sleep 1
    python train.py task=pick-cube model_size=5 wandb_project=tdmpc2 wandb_entity=jambotime demo_dir=demonstrations/motionplanning/ seed=$((i*10)) obs=rgb
done