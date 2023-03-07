# BDD Semantic Segmentation using WandB

This is semantic segmentation project on BDD cityscapes-like dataset that are tracked via wandb. You can see the run on [my wandb](https://wandb.ai/gustiwinata/mlops-course-001).

## How to set up the project

The setups are pretty basic, just go

```bash
> pip install virtualenv
> virtualenv venv
> source venv/bin/activate
> pip install requirements.txt
> wandb login
```

Then run the code sequentially
```bash
> python data_loader.py
> python split.py
> python baseline.py
> python eval.py
```

> You only need to run `data_loader.py` and `split.py` once since this data is static, but you could run `baseline.py` and `eval.py` multiple times since it is what the experiments about.

You can set configs of hyperparameters to experiment in the `baseline.py` file, try tweak some of the hyperparameters. The `eval.py` is to check the model that are produced on `baseline.py` executions on test holdout set.

## Reports on Wandb

You can check my reports on this repository executions on these pages
- [Dataset Exploration](https://api.wandb.ai/links/gustiwinata/etuh4k5c)
- [Hyperparameter Sweep](https://api.wandb.ai/links/gustiwinata/x2vn7bk9)
- [Model Evaluation](https://api.wandb.ai/links/gustiwinata/8rw8l59g)
