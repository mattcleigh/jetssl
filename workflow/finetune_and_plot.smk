import os
from pathlib import Path

# Run using
# pip install snakemake-executor-plugin-slurm==0.4.1 snakemake==8.4.1
# snakemake --snakefile workflow/finetune_and_plot.smk --workflow-profile workflow/exp --configfile workflow/config.yaml
# -e dryrun --dag | dot -Tpng > dag.png


# This tells snakemake to check if the variables exist before running
envvars:
    "WANDB_API_KEY",


# Set the container to use for all rules
container: config["container_path"]


########################################

# Define important paths
project_name = "jetssl_finetune2"
output_dir = "/srv/beegfs/scratch/groups/rodem/jetssl/"
backbones = "/srv/beegfs/scratch/groups/rodem/jetssl/jetssl2/backbones/"
wdir = config["workdir"]
proj = str(Path(output_dir, project_name)) + "/"
plot_dir = str(Path(wdir, "plots", project_name)) + "/"
seeds = [0]

# Define the model backbones to finetune
model_names = ["kmeans", "diff", "flow", "untrained"]

# Define the finetuning tasks
downstream_tasks = {
    # "jetclass": ["experiment=train_classifier", "datamodule=jetclass"],
    # "shlomi": ["experiment=train_classifier", "datamodule=shlomi"],
    # "vtx": ["experiment=train_vertexer"],
    "cwola": ["experiment=train_cwola"],
}

# Define the number of jets to finetune on
n_jets = [1e3, 1e4, 1e5, 1e6]

########################################

# Flatten the parameters with spaces for easier injection
dt_names = list(downstream_tasks.keys())
dt_args = {k: " ".join(v) for k, v in downstream_tasks.items()}

# Final rule to form the endpoint of the DAG
rule all:
    input:
        expand(f"{plot_dir}{{dt}}.pdf", dt=dt_names),


# For each downstream task make a rule to plot, export and finetune
for dt in dt_names:

    # For the cwola task, we need much less jets!
    if dt == "cwola":
        n_jets = [5e2, 1e3, 5e3, 1e4, 1e5]
    # For the vertexing task, we use all jets only
    elif dt == "vtx":
        n_jets = [1e6]
    n_jets = [int(j) for j in n_jets]

    # Work out whick plotting script to run
    if dt == "vtx":
        plot_script = "plotting/vtx_perf.py"
    elif dt == "cwola":
        plot_script = "plotting/sic_vs_njets.py"
    else:
        plot_script = "plotting/acc_vs_njets.py"

    # Plotting rule
    # Takes in: Exported scores from each model per n_jets
    # Produces: A plot of the accuracy vs n_jets
    rule:
        name:
            f"plot_{dt}"
        input:
            expand(f"{proj}{dt}_{{m}}_{{nj}}_{{s}}/outputs/test_set.h5", m=model_names, nj=n_jets, s=seeds),
        output:
            f"{plot_dir}{dt}.pdf",
        params:
            plot_script,
            f"outfile={dt}",
            *(f"+models.{m}={dt}_{m}" for m in model_names), # Will search for the njets and seeds
            f"plot_dir={plot_dir}",
            f"path={proj}",
        threads: 1
        resources:
            slurm_partition="shared-cpu,private-dpnc-cpu,public-short-cpu",
            runtime=60,  # minutes
            mem_mb=4000,
        wrapper:
            "file:hydra_cli"

    # Next steps must be done for each model, n_jets, fixed or pt, seed
    for nj in n_jets:
        for m in model_names:
            for s in seeds:

                # Combine the model and n_jets into a unique identifier
                model_id = f"{dt}_{m}_{nj}_{s}"

                # Export each model
                # Takes in: A finetuned model with a successful training txt file
                # Produces: A test_set.h5 file with the model predictions
                rule:
                    name:
                        f"export_{model_id}"
                    input:
                        f"{proj}{model_id}/train_finished.txt",
                    output:
                        f"{proj}{model_id}/outputs/test_set.h5",
                    params:
                        "scripts/export.py",
                        f"network_name={model_id}",
                        f"project_name={project_name}",
                    threads: 8
                    resources:
                        slurm_partition="shared-gpu,private-dpnc-gpu",
                        runtime=60,  # minutes
                        slurm_extra="--gres=gpu:ampere:1",
                        mem_mb=20000,
                    wrapper:
                        "file:hydra_cli"

                # Fine tune each model
                # Takes in: A pre-trained model checkpoint
                # Produces: A finetuned model with a successful training txt file
                rule:
                    name:
                        f"finetune_{model_id}"
                    output:
                        f"{proj}{model_id}/train_finished.txt",
                    params:
                        "scripts/train.py",
                        "experiment=train_jetfinetune",
                        f"network_name={model_id}",
                        f"project_name={project_name}",
                        f"model.backbone_path={backbones}{m}.pkl",
                        f"n_jets={nj}",
                        f"seed={s}",
                        dt_args[dt],
                    threads: 8
                    resources:
                        slurm_partition="shared-gpu,private-dpnc-gpu",
                        runtime=3 * 60,  # minutes
                        slurm_extra="--gres=gpu:ampere:1",
                        mem_mb=20000,
                    wrapper:
                        "file:hydra_cli"

