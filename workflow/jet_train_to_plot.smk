import os
from pathlib import Path

# Run using
# pip install snakemake-executor-plugin-slurm snakemake
# snakemake --snakefile workflow/jet_train_to_plot.smk --workflow-profile workflow/exp --configfile workflow/config.yaml
# -e dryrun --dag | dot -Tpng > dag.png


# This tells snakemake to check if the variables exist before running
envvars:
    "WANDB_API_KEY",


# Set the container to use for all rules
container: config["container_path"]


########################################

# Define important paths
project_name = "new_mae"
output_dir = "/srv/beegfs/scratch/groups/rodem/flowbert/"
wdir = config["workdir"]
proj = str(Path(output_dir, project_name)) + "/"
plot_dir = str(Path(wdir, "plots", project_name)) + "/"

# Define the models to train. Key=network_name, Value=List of hydra parameters
model_list = {
    "onlyid": ["model=mpmreg", "model._target_=src.models.mpm_base.MPMBase"],
    "kmeans": ["model=mpmtoken"],
    "flow": ["model=mpmflow"],
    "regression": ["model=mpmreg"],
    "dino": ["model=jetdino"],
}

# Define the finetuning tasks
downstream_tasks = {
    "jetclass": ["experiment=train_jetclassifier", "datamodule=jetclass"],
    "shlomi": ["experiment=train_jetclassifier", "datamodule=shlomi"],
    # "vtx": ["experiment=train_jetvertex", "datamodule=shlomi"],
}

########################################

# Flatten the parameters with spaces for easier injection
model_names = list(model_list.keys())
dt_names = list(downstream_tasks.keys())
model_args = {k: " ".join(v) for k, v in model_list.items()}
dt_args = {k: " ".join(v) for k, v in downstream_tasks.items()}

# All the model names including the reinit
all_model_names = model_names + ["reinit"]


# Final rule to form the endpoint of the DAG
rule all:
    input:
        expand(f"{plot_dir}{{dt}}.pdf", dt=dt_names),


# For each downstream task make a rule to plot, export and finetune
for dt in dt_names:

    rule:
        name:
            f"plot_{dt}"
        input:
            expand(f"{proj}{{m}}_{dt}/outputs/test_set.h5", m=all_model_names),
        output:
            f"{plot_dir}{dt}.pdf",
        params:
            "plotting/roc.py",
            "outfile=roc.pdf",
            *(f"+models.{m}={m}_{dt}" for m in all_model_names),
            f"plot_dir={plot_dir}",
            f"path={proj}",
        threads: 1
        resources:
            slurm_partition="shared-cpu,private-dpnc-cpu,public-short-cpu",
            runtime=60,  # minutes
            mem_mb=4000,
        wrapper:
            "file:hydra_cli"

    # Each model must be exported
    for m in all_model_names:
        rule:
            name:
                f"export_{m}_{dt}"
            input:
                f"{proj}{m}_{dt}/train_finished.txt",
            output:
                f"{proj}{m}_{dt}/outputs/test_set.h5",
            params:
                "scripts/export.py",
                f"network_name={m}_{dt}",
                f"project_name={project_name}",
            threads: 8
            resources:
                slurm_partition="shared-gpu,private-dpnc-gpu",
                slurm_extra="--gres=gpu:1",
                runtime=60,  # minutes
            wrapper:
                "file:hydra_cli"

    # Each model must be finetuned (not reinit)
    for m in model_names:
        rule:
            name:
                f"finetune_{m}_{dt}"
            input:
                f"{proj}{m}/train_finished.txt",
            output:
                f"{proj}{m}_{dt}/train_finished.txt",
            params:
                "scripts/train.py",
                "experiment=train_jetfinetune",
                f"network_name={m}_{dt}",
                f"project_name={project_name}",
                f"model.backbone_path={proj}{m}/backbone.pkl",
                dt_args[dt],
            threads: 8
            resources:
                slurm_partition="shared-gpu,private-dpnc-gpu",
                runtime=3 * 60,  # minutes
                slurm_extra="--gres=gpu:ampere:1",
                mem_mb=20000,
            wrapper:
                "file:hydra_cli"

    # Reinit trained
    rule:
        name:
            f"reinit_{dt}"
        input:
            f"{proj}{model_names[-1]}/train_finished.txt",
        output:
            f"{proj}reinit_{dt}/train_finished.txt",
        params:
            "scripts/train.py",
            "experiment=train_jetfinetune",
            "model.reinstantiate=True",
            f"network_name=reinit_{dt}",
            f"project_name={project_name}",
            f"model.backbone_path={proj}{model_names[-1]}/backbone.pkl",
            dt_args[dt],
        threads: 8
        resources:
            slurm_partition="shared-gpu,private-dpnc-gpu",
            slurm_extra="--gres=gpu:ampere:1",
            mem_mb=20000,
            runtime=12 * 60,  # minutes
        wrapper:
            "file:hydra_cli"

# For each model make a rule to pretrain
for m in model_names:
    rule:
        name:
            f"pretrain_{m}"
        output:
            f"{proj}{m}/train_finished.txt",
        params:
            "scripts/train.py",
            "experiment=train_mpm",
            f"network_name={m}_L",
            f"project_name={project_name}",
            model_args[m],
        threads: 6
        resources:
            slurm_partition="private-dpnc-gpu,shared-gpu",
            slurm_extra="--gres=gpu:ampere:1,VramPerGpu:20G",
            runtime=12 * 60,  # minutes
            mem_mb=40000,
        wrapper:
            "file:hydra_cli"
