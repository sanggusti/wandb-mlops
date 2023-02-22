import wandb
import pandas as pd
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

import config
from utils import get_predictions, create_iou_table, MIOU, BackgroundIOU, \
                  RoadIOU, TrafficLightIOU, TrafficSignIOU, PersonIOU, VehicleIOU, BicycleIOU

train_config = SimpleNamespace(
    framework="fastai",
    img_size=(180, 320),
    batch_size=8,
    augment=True, # use data augmentation
    epochs=10, 
    lr=2e-3,
    pretrained=True,  # whether to use pretrained encoder
    seed=42,
)

def label_func(fname):
    return (fname.parent.parent/"labels")/f"{fname.stem}_mask.png"

def get_data(df, bs=4, img_size=(180, 320), augment=True):
    block = DataBlock(blocks=(ImageBlock, MaskBlock(codes=config.BDD_CLASSES)),
                  get_x=ColReader("image_fname"),
                  get_y=ColReader("label_fname"),
                  splitter=ColSplitter(),
                  item_tfms=Resize(img_size),
                  batch_tfms=aug_transforms() if augment else None,
                 )
    return block.dataloaders(df, bs=bs)


if __name__ == "__main__":
    set_seed(train_config.seed, reproducible=True)
    # Set Wandb
    run = wandb.init(project=config.WANDB_PROJECT, entity=config.ENTITY, job_type="training", config=train_config)
    # Use data that are uploaded to wandb
    processed_data_at = run.use_artifact(f'{config.PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')

    df = df[df.Stage != 'test'].reset_index(drop=True)
    df['is_valid'] = df.Stage == 'valid'

    # assign paths
    df["image_fname"] = [processed_dataset_dir/f'images/{f}' for f in df.File_Name.values]
    df["label_fname"] = [label_func(f) for f in df.image_fname.values]

    # Track Hyperparameters
    training_config = wandb.config
    dls = get_data(df, bs=training_config.batch_size, img_size=training_config.img_size, augment=training_config.augment)
    metrics = [MIOU(), BackgroundIOU(), RoadIOU(), TrafficLightIOU(), \
           TrafficSignIOU(), PersonIOU(), VehicleIOU(), BicycleIOU()]

    learn = unet_learner(dls, arch=resnet18, pretrained=training_config.pretrained, metrics=metrics)

    callbacks = [
        SaveModelCallback(monitor='miou'),
        WandbCallback(log_preds=False, log_model=True)
    ]
    # Training model
    learn.fit_one_cycle(training_config.epochs, training_config.lr, cbs=callbacks)

    # Log to tables
    samples, outputs, predictions = get_predictions(learn)
    table = create_iou_table(samples, outputs, predictions, config.BDD_CLASSES)
    wandb.log({"pred_table":table})

    # Save loss and metrics via wandb.summary
    scores = learn.validate()
    metric_names = ['final_loss'] + [f'final_{x.name}' for x in metrics]
    final_results = {metric_names[i] : scores[i] for i in range(len(scores))}
    for k,v in final_results.items(): 
        wandb.summary[k] = v
    wandb.finish()
