for i in {1..5}; do
    echo "**************Run $i***************"
    python train.py task=pick-cube model_size=5 wandb_project=tdmpc2 wandb_entity=jambotime obs=rgb second_cam=rgb seed=$(shuf -i 0-1073741824 -n 1) demo_dir=demonstrations/maniskill/rgb_rgb uncertainty_weight=true &
    sleep 1
    python train.py task=pick-cube model_size=5 wandb_project=tdmpc2 wandb_entity=jambotime obs=rgb second_cam=rgb seed=$(shuf -i 0-1073741824 -n 1) demo_dir=demonstrations/maniskill/rgb_rgb
done