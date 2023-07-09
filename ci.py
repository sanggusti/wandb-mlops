import wandb


print(f'The version of wandb is: {wandb.__version__}')
assert wandb.__version__ == '0.15.5', f'Expected version 0.15.5, but got {wandb.__version__}'
