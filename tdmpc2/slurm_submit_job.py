import subprocess
import argparse
import os


def submit_job(run_cmd, job_name, gpu_type, double_run=False):
    # Determine constraint based on GPU type
    if "a100" in gpu_type.lower():
        constraint = '"a100-80G"'
    else:
        constraint = gpu_type.lower()

    if double_run:
        run_cmd = f"{run_cmd} & \n {run_cmd} & \nwait"
    print(f"Prepared command for {job_name}: {run_cmd}")

    slurm_template = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH --time=02-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:{gpu_type}:1
#SBATCH --constraint={constraint}
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64g
#SBATCH --output={job_name}.%j.%N.out
#SBATCH --error={job_name}.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=james.staley625703@tufts.edu

module load anaconda/2025.06.0
conda activate rl
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

{run_cmd}
"""

    temp_script = f"temp_submit_{job_name}.sh"
    with open(temp_script, "w") as f:
        f.write(slurm_template)

    try:
        subprocess.run(["sbatch", temp_script], check=True)
        print(f"Successfully submitted {job_name} on {gpu_type}")
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)


if __name__ == "__main__":
    # Common args shared across all unreliable region experiments
    common = (
        "task=pick-cube model_size=5 obs=state "
        "wandb_project=tdmpc2_unreliable wandb_entity=jambotime "
        "uncertainty_weight=false unreliable_region=true "
        "unreliable_axis=1 unreliable_threshold=0.0 unreliable_side=positive "
        "ee_obs_idx=18 reliability_mode=mask"
    )

    commands = {
        "ur_baseline":  f"python train.py {common} exp_name=baseline seed=$(shuf -i 0-1073741824 -n 1)",
        "ur_avoidance": f"python train.py {common} oracle_avoidance=true exp_name=avoidance seed=$(shuf -i 0-1073741824 -n 1)",
        "ur_paired":    (
            f"python train.py {common} oracle_avoidance=true exp_name=avoidance seed=$SEED & \n"
            f"sleep 1 \n"
            f"python train.py {common} exp_name=baseline seed=$SEED & \n"
            f"wait"
        ),
    }

    parser = argparse.ArgumentParser(description="Submit TDMPC2 unreliable region SLURM jobs.")
    parser.add_argument("--name", required=True, choices=commands.keys(),
                        help="Job name (ur_baseline, ur_avoidance, ur_paired)")
    parser.add_argument("--gpu", default="l40s", choices=["l40s", "h200", "a100"],
                        help="GPU type")
    parser.add_argument("--double_run", action="store_true",
                        help="Run the command twice in parallel")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fixed seed (overrides random seed in command)")
    args = parser.parse_args()

    run_cmd = commands[args.name]
    if args.seed is not None:
        # Replace the shuf expression or $SEED with the fixed seed
        run_cmd = run_cmd.replace("$(shuf -i 0-1073741824 -n 1)", str(args.seed))
        run_cmd = run_cmd.replace("$SEED", str(args.seed))
    else:
        # For paired mode, generate a shared seed
        if "SEED" in run_cmd:
            run_cmd = f"SEED=$(shuf -i 0-1073741824 -n 1)\n{run_cmd}"

    submit_job(run_cmd, args.name, args.gpu, double_run=args.double_run)
