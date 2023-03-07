import wandb
import pandas as pd
import torchvision.models as tvmodels
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
import argparse

import config
from utils import get_predictions, create_iou_table, MIOU, BackgroundIOU, \
                  RoadIOU, TrafficLightIOU, TrafficSignIOU, PersonIOU, VehicleIOU, BicycleIOU, t_or_f

default_config = SimpleNamespace(
    framework="fastai",
    img_size=180, #(180, 320) in 16:9 proportions,
    batch_size=8, #8 keep small in Colab to be manageable
    augment=True, # use data augmentation
    epochs=10, # for brevity, increase for better results :)
    lr=2e-3,
    pretrained=True,  # whether to use pretrained encoder,
    mixed_precision=True, # use automatic mixed precision
    arch="resnet18",
    seed=42,
    log_preds=False,
)

def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--img_size', type=int, default=default_config.img_size, help='image size')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--arch', type=str, default=default_config.arch, help='timm backbone architecture')
    argparser.add_argument('--augment', type=t_or_f, default=default_config.augment, help='Use image augmentation')
    argparser.add_argument('--seed', type=int, default=default_config.seed, help='random seed')
    argparser.add_argument('--log_preds', type=t_or_f, default=default_config.log_preds, help='log model predictions')
    argparser.add_argument('--pretrained', type=t_or_f, default=default_config.pretrained, help='Use pretrained model')
    argparser.add_argument('--mixed_precision', type=t_or_f, default=default_config.mixed_precision, help='use fp16')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

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

def download_data():
    processed_data_at = wandb.use_artifact(f'{config.PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir

def get_df(processed_dataset_dir, is_test=False):
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')

    if not is_test:
        df = df[df.Stage != 'test'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
    else:
        df = df[df.Stage != 'test'].reset_index(drop=True)

    # assign paths
    df["image_fname"] = [processed_dataset_dir/f'images/{f}' for f in df.File_Name.values]
    df["label_fname"] = [label_func(f) for f in df.image_fname.values]
    return df

def log_predictions(learn):
    # Log to tables
    samples, outputs, predictions = get_predictions(learn)
    table = create_iou_table(samples, outputs, predictions, config.BDD_CLASSES)
    wandb.log({"pred_table":table})

def log_final_training(learn):
    # Save loss and metrics via wandb.summary
    scores = learn.validate()
    metric_names = ['final_loss'] + [f'final_{x.name}' for x in learn.metrics]
    final_results = {metric_names[i] : scores[i] for i in range(len(scores))}
    for k,v in final_results.items(): 
        wandb.summary[k] = v

def train(train_config):
    set_seed(train_config.seed, reproducible=True)
    # Set Wandb
    run = wandb.init(
        project=config.WANDB_PROJECT, 
        entity=config.ENTITY, 
        job_type="training", 
        config=train_config
    )
    # Track Hyperparameters
    training_config = wandb.config

    processed_dataset_dir = download_data()
    df = get_df(processed_dataset_dir)
    dls = get_data(
        df, 
        bs=training_config.batch_size, 
        img_size=training_config.img_size, 
        augment=training_config.augment
    )
    
    metrics = [MIOU(), BackgroundIOU(), RoadIOU(), TrafficLightIOU(), \
           TrafficSignIOU(), PersonIOU(), VehicleIOU(), BicycleIOU()]

    learn = unet_learner(
        dls, 
        arch=getattr(tvmodels, training_config.arch), 
        pretrained=training_config.pretrained, 
        metrics=metrics
    )

    callbacks = [
        SaveModelCallback(monitor='miou'),
        WandbCallback(log_preds=False, log_model=True)
    ]
    # Training model
    learn.fit_one_cycle(
        training_config.epochs, 
        training_config.lr, 
        cbs=callbacks
    )

    if training_config.log_preds:
        log_predictions(learn)

    log_final_training(learn)
    wandb.finish()


if __name__ == "__main__":
    parse_args()
    train(default_config)
    
