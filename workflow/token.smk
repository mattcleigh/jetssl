import os
from pathlib import Path

# Run using
# pip install snakemake-executor-plugin-slurm==0.4.1 snakemake==8.4.1
# snakemake --snakefile workflow/token.smk --workflow-profile workflow/exp --configfile workflow/config.yaml
# -e dryrun --dag | dot -Tpng > dag.png


# This tells snakemake to check if the variables exist before running
envvars:
    "WANDB_API_KEY",


# Set the container to use for all rules
container: config["container_path"]
wdir = config["workdir"]


########################################

# Define important paths
project_name = "token"
output_dir = "/srv/beegfs/scratch/groups/rodem/jetssl/" + project_name + "/"

# Define the settings for the models
models = ["kmeans", "vae", "none"]

########################################

rule all:
    input:
        expand(f"{output_dir}{{m_name}}/outputs/test_set.h5", m_name=models)

rule export:
    input:
        f"{output_dir}{{m_name}}/train_finished.txt",
    output:
        f"{output_dir}{{m_name}}/outputs/test_set.h5",
    params:
        "scripts/export.py",
        "network_name={m_name}",
        f"project_name={project_name}",
    threads: 8
    resources:
        slurm_partition="shared-gpu,private-dpnc-gpu",
        runtime=60,  # minutes
        slurm_extra="--gres=gpu:ampere:1",
        mem_mb=20000,
    wrapper:
        "file:hydra_cli"

rule train:
    output:
        f"{output_dir}{{m_name}}/train_finished.txt",
    params:
        "scripts/train.py",
        "model=token_class",
        "experiment=train_classifier",
        "network_name={m_name}",
        f"project_name={project_name}",
        "model.token_type={m_name}",
        #
        "trainer.max_steps=200_000",
        "model.scheduler.warmup_steps=10_000",
        "datamodule.batch_size=500",
        "n_jets=100_000_000",
        "+trainer.val_check_interval=20_000",
        "~callbacks.backbone_finetune"
    threads: 12
    resources:
        slurm_partition="shared-gpu,private-dpnc-gpu",
        runtime=12 * 60,  # minutes
        slurm_extra="--gres=gpu:ampere:1",
        mem_mb=20000,
    wrapper:
        "file:hydra_cli"
