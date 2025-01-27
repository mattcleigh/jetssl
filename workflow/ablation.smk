import os
from pathlib import Path

# Run using
# pip install snakemake-executor-plugin-slurm==0.4.1 snakemake==8.4.1
# snakemake --snakefile workflow/ablation.smk --workflow-profile workflow/exp --configfile workflow/config.yaml
# -e dryrun --dag | dot -Tpng > dag.png


# This tells snakemake to check if the variables exist before running
envvars:
    "WANDB_API_KEY",


# Set the container to use for all rules
container: config["container_path"]
wdir = config["workdir"]


########################################

# Define important paths
project_name = "ablation"
output_dir = "/srv/beegfs/scratch/groups/rodem/jetssl/" + project_name + "/"

# Define the settings for the models
# style_sweep = [
#     "mpm-reg-3-noID-BERT-0reg       False [reg,probe]       3  bert  0",
#     "mpm-kmeans-3-noID-BERT-0reg    False [kmeans,probe]    3  bert  0",
#     "mpm-reg-7-noID-BERT-0reg       False [reg,probe]       7  bert  0",
#     "mpm-kmeans-7-noID-BERT-0reg    False [kmeans,probe]    7  bert  0",
#     "mpm-reg-7-yesID-BERT-0reg      True  [reg,id,probe]    7  bert  0",
#     "mpm-kmeans-7-yesID-BERT-0reg   True  [kmeans,id,probe] 7  bert  0",
#     "mpm-reg-7-yesID-MAE-0reg       True  [reg,id,probe]    7  mae   0",
#     "mpm-kmeans-7-yesID-MAE-0reg    True  [kmeans,id,probe] 7  mae   0",
#     "mpm-reg-7-yesID-MAE-8reg       True  [reg,id,probe]    7  mae   8",
#     "mpm-kmeans-7-yesID-MAE-8reg    True  [kmeans,id,probe] 7  mae   8",
# ]
# depth_sweep = [f"mae-kmeans-depth{d}  {d}" for d in range(1, 5)]
# mask_sweep = [f"mae-kmeans-mask{m*10:.0f}  {m}" for m in map(lambda x: x/10, range(2, 10))]
# weight_sweep = [f"mae-{method}-weight {method}" for method in ["kmeans", "reg"]]
joint_sweep = [
    "jepa x",
    "dijepa x",
]

# Get the list of models in all the sweeps
model_names = [m.split()[0] for s in [joint_sweep] for m in s]

# Parameters used for ALL rules
standard_params = (
    f"project_name={project_name}",
    "trainer.max_steps=200_000",
    "model.scheduler.warmup_steps=10_000",
    "trainer.val_check_interval=20_000",
    "datamodule.batch_size=500",
    "n_jets=100_000_000", # All 100M jets
    "full_resume=True",
)

########################################

rule all:
    input:
        expand(f"{output_dir}jetclass_{{m_name}}/train_finished.txt", m_name=model_names)

rule finetune:
    input:
        f"{output_dir}{{m_name}}/backbone.pkl"
    output:
        f"{output_dir}jetclass_{{m_name}}/train_finished.txt",
    params:
        "scripts/train.py",
        "experiment=train_classifier",
        "network_name=jetclass_{m_name}",
        f"model.backbone_path={output_dir}{{m_name}}/backbone.pkl",
        "callbacks.backbone_finetune.unfreeze_at_step=9999999999",
        lambda w : f"csts_dim={3 if '-3-noID-BERT-0reg' in w.m_name else 7}",
        *standard_params
    threads: 12
    resources:
        slurm_partition="shared-gpu,private-dpnc-gpu",
        runtime=12 * 60,  # minutes
        slurm_extra="--gres=gpu:ampere:1",
        mem_mb=20000,
    wrapper:
        "file:hydra_cli"

for model in joint_sweep:
    m_name, _ = model.split()

    rule:
        name:
            f"pretrain_{m_name}"
        output:
            protected(f"{output_dir}{m_name}/backbone.pkl")
        params:
            "scripts/train.py",
            "experiment=jepa",
            f"network_name={m_name}",
            *standard_params
        threads: 12
        resources:
            slurm_partition="shared-gpu,private-dpnc-gpu",
            runtime=12 * 60,  # minutes
            slurm_extra="--gres=gpu:ampere:1",
            mem_mb=20000,
        wrapper:
            "file:hydra_cli"

# for model in style_sweep:
#     m_name, use_id, tasks, csts_dim, objective, num_registers = model.split()

#     rule:
#         name:
#             f"pretrain_{m_name}"
#         output:
#             protected(f"{output_dir}{m_name}/backbone.pkl")
#         params:
#             "scripts/train.py",
#             "experiment=mpmv1",
#             f"network_name={m_name}",
#             f"+model/tasks={tasks}",
#             f"csts_dim={csts_dim}",
#             f"model.objective={objective}",
#             f"model.use_id={use_id}",
#             f"model.encoder_config.num_registers={num_registers}",
#             *standard_params
#         threads: 12
#         resources:
#             slurm_partition="shared-gpu,private-dpnc-gpu",
#             runtime=12 * 60,  # minutes
#             slurm_extra="--gres=gpu:ampere:1",
#             mem_mb=20000,
#         wrapper:
#             "file:hydra_cli"

# for model in depth_sweep:
#     m_name, depth = model.split()

#     rule:
#         name:
#             f"pretrain_{m_name}"
#         output:
#             protected(f"{output_dir}{m_name}/backbone.pkl")
#         params:
#             "scripts/train.py",
#             "experiment=pretrain",
#             f"network_name={m_name}",
#             f"model.decoder_config.num_layers={depth}",
#             "+model/tasks=[kmeans,id,probe]",
#             *standard_params
#         threads: 12
#         resources:
#             slurm_partition="shared-gpu,private-dpnc-gpu",
#             runtime=12 * 60,  # minutes
#             slurm_extra="--gres=gpu:ampere:1",
#             mem_mb=20000,
#         wrapper:
#             "file:hydra_cli"

# for model in mask_sweep:
#     m_name, mask_fraction = model.split()

#     rule:
#         name:
#             f"pretrain_{m_name}"
#         output:
#             protected(f"{output_dir}{m_name}/backbone.pkl")
#         params:
#             "scripts/train.py",
#             "experiment=pretrain",
#             f"network_name={m_name}",
#             f"mask_fraction={mask_fraction}",
#             "+model/tasks=[kmeans,id,probe]",
#             *standard_params
#         threads: 12
#         resources:
#             slurm_partition="shared-gpu,private-dpnc-gpu",
#             runtime=12 * 60,  # minutes
#             slurm_extra="--gres=gpu:ampere:1",
#             mem_mb=20000,
#         wrapper:
#             "file:hydra_cli"

# for model in weight_sweep:
#     m_name, method = model.split()

#     rule:
#         name:
#             f"pretrain_{m_name}"
#         output:
#             protected(f"{output_dir}{m_name}/backbone.pkl")
#         params:
#             "scripts/train.py",
#             "experiment=pretrain",
#             f"network_name={m_name}",
#             f"+model/tasks=[{method},id,probe]",
#             "use_weights=True",
#             *standard_params
#         threads: 12
#         resources:
#             slurm_partition="shared-gpu,private-dpnc-gpu",
#             runtime=12 * 60,  # minutes
#             slurm_extra="--gres=gpu:ampere:1",
#             mem_mb=40000,
#         wrapper:
#             "file:hydra_cli"
