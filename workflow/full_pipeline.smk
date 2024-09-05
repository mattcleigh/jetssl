import os
from pathlib import Path

# Run using
# pip install snakemake-executor-plugin-slurm==0.4.1 snakemake==8.4.1
# snakemake --snakefile workflow/full_pipeline.smk --workflow-profile workflow/exp --configfile workflow/config.yaml
# -e dryrun --dag | dot -Tpng > dag.png


# This tells snakemake to check if the variables exist before running
envvars:
    "WANDB_API_KEY",


# Set the container to use for all rules
container: config["container_path"]


########################################

# Define important paths
debug_mode = False
fix_backbone = True
project_name = "jetssl_fixed_frozen" if fix_backbone else "jetssl_fixed_finetune"
output_dir = "/srv/beegfs/scratch/groups/rodem/jetssl/"
backbones = "/srv/beegfs/scratch/groups/rodem/jetssl/jetssl_fixed/"
wdir = config["workdir"]
proj = str(Path(output_dir, project_name)) + "/"
plot_dir = str(Path(wdir, "plots", project_name)) + "/"
seeds = [0] #, 1, 2, 3, 4]

# Define the model backbones to finetune
model_names = [
    # "kmeans",
    # "reg",
    # "diff",
    # "flow",
    # "vae",
    # "mdm",
    "sdm",
    # "untrained",
    # "nobackbone",
]


# Define the finetuning tasks
downstream_tasks = [
    "jetclass",
    # "cwola",
    # "btag",
    # "vtx",
    # "trk",
]

########################################

# If fixed -> use nobackbone as baseline
# if fix_backbone:
#     model_names.remove("untrained")
# If fine-tuning -> use untrained as baseline
# else:
#     model_names.remove("nobackbone")

# Final rule to form the endpoint of the DAG
rule all:
    input:
        expand(f"{plot_dir}{{dt}}.pdf", dt=downstream_tasks),


# For each downstream task make a rule to finetune -> export -> plot
for dt in downstream_tasks:

    # Standard classification we want the full dataset
    if dt == "jetclass":
        n_jets = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    # btag doesnt have alot of samples
    elif dt == "btag":
        n_jets = [2e3, 2e4, 2e5, 2_023_331]
    # For the cwola task, we need much less jets!
    if dt == "cwola":
        n_jets = [5e2, 1e3, 5e3, 1e4, 1e5]
    # For the vertexing task or tracking, we use all jets only
    elif dt in {"vtx", "trk"}:
        n_jets = [2_023_331]
    n_jets = [int(j) for j in n_jets]

    # Which plotting script to run
    if dt == "vtx":
        plot_script = "plotting/vtx_perf.py"
    elif dt == "cwola":
        plot_script = "plotting/sic_vs_njets.py"
    elif dt == "trk":
        plot_script = "plotting/trk_perf.py"
    else:
        plot_script = "plotting/acc_vs_njets.py"

    # Trim everything if we are in debug mode
    if debug_mode:
        n_jets = n_jets[:1]
        seeds = seeds[:1]
        model_names = model_names[:1]

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
    for s in seeds:
        for nj in n_jets:
            for m in model_names:

                # Combine the model and n_jets into a unique identifier
                model_id = f"{dt}_{m}_{nj}_{s}"

                # Set all of the rules for finetuning
                ft_rules = ""

                # Select the appropriate experiment and data module
                if dt == "jetclass":
                    ft_rules += "experiment=train_classifier "
                    ft_rules += "datamodule=jetclass "
                elif dt == "btag":
                    ft_rules += "experiment=train_classifier "
                    ft_rules += "datamodule=btag "
                elif dt == "vtx":
                    ft_rules += "experiment=train_vertexer "
                elif dt == "cwola":
                    ft_rules += "experiment=train_cwola "
                elif dt == "trk":
                    ft_rules += "experiment=train_tracker "

                # Setting the patience parameter for the classifier tasks
                if dt in {"jetclass", "btag"}:
                    if nj < 1e4:
                        ft_rules += "trainer.check_val_every_n_epoch=10 "
                        ft_rules += "callbacks.early_stopping.patience=5 "
                    elif nj <= 1e4:
                        ft_rules += "callbacks.early_stopping.patience=25 "
                    elif nj <= 1e6:
                        ft_rules += "callbacks.early_stopping.patience=10 "
                    else:
                        ft_rules += "callbacks.early_stopping.patience=5 "

                # Changine the warmup period to be longer for huge datasets
                if not fix_backbone:
                    if nj >= 1e7:
                        ft_rules += "model.scheduler.warmup_steps=40000 " # 5K

                    # Setting the learning rate to be higher for jetclass
                    if dt == "jetclass" and (nj == 1e8 or m in {"untrained", "nobackbone"}):
                        ft_rules += "model.optimizer.lr=0.001 " # Default is 1e-4

                    # Deciding on when to unfreeze the backbone
                    if m in {"untrained", "nobackbone"}:
                        ft_rules += "callbacks.backbone_finetune.unfreeze_at_step=1 "
                        ft_rules += "callbacks.backbone_finetune.catchup_steps=1 "
                    elif nj >= 1e7:
                        ft_rules += "callbacks.backbone_finetune.unfreeze_at_step=20000 "
                        ft_rules += "callbacks.backbone_finetune.catchup_steps=20000 "
                    elif nj <= 1e3:
                        ft_rules += "callbacks.backbone_finetune.unfreeze_at_step=100 "

                else: # Fix the backbone
                    # Never fix the nobackbone model
                    if m == "nobackbone":
                        ft_rules += "callbacks.backbone_finetune.unfreeze_at_step=1 "
                        ft_rules += "callbacks.backbone_finetune.catchup_steps=1 "
                    else:
                        ft_rules += "callbacks.backbone_finetune.unfreeze_at_step=9999999999999 "

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
                        f"network_name={model_id}",
                        f"project_name={project_name}",
                        f"model.backbone_path={backbones}{m}/backbone.pkl",
                        f"n_jets={nj}",
                        f"seed={s}",
                        "full_resume=True",
                        ft_rules,
                    threads: 8
                    resources:
                        slurm_partition="shared-gpu,private-dpnc-gpu",
                        runtime=12 * 60,  # minutes
                        slurm_extra="--gres=gpu:ampere:1",
                        mem_mb=20000,
                    wrapper:
                        "file:hydra_cli"
