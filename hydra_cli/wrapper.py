"""A simple wrapper for executing python scripts via snakemake.

It is used to allow the use of hydra's config system.
"""

from pathlib import Path

from snakemake.shell import shell

if globals().get("snakemake") is None:
    snakemake = None  # prevent linter problems
    raise Exception("This script was run without snakemake")

if len(snakemake.params.keys()) > 0:
    raise Exception(
        "Keyword arguments are not allowed with this wrapper: ",
        list(snakemake.params.keys()),
    )

script = snakemake.params.pop(0)
if not Path(script).exists():
    raise Exception("First parameter is not a valid path to a script file: ", script)

params_context = snakemake.__dict__.copy()
args = []
for opt in snakemake.params:
    option = opt.replace("{hs:", "{")
    exec(f"arg = f'{option}'", {}, params_context)
    args.append(params_context["arg"])
args = " ".join(args)

if snakemake.log:
    shell(f"python {script} {args} > {snakemake.log} 2>&1")
else:
    shell(f"python {script} {args}")
