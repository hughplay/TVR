import wandb

api = wandb.Api()
entity = api.default_entity

# get runs from the project
filters = {"tags": {"$in": ["view"]}}

filters = {
    "$and": [
        {"tags": {"$in": ["view"]}},
        {"tags": {"$in": ["base"]}},
    ],
}
runs = api.runs(f"{entity}/tvr", filters=filters)
runs = [run for run in runs if "val/acc" in run.summary]


commands = []
for run in runs:
    command = (
        f"python test.py /log/exp/tvr/{run.id} --ckpt last --update_wandb\n\n"
    )
    commands.append(command)

with open("scripts/testing/retest.sh", "w") as f:
    f.write("".join(commands))
