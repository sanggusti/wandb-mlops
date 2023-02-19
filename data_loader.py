import config
import wandb
from fastai.vision.all import *


def label_func(fname):
    return(fname.parent.parent/"labels")/f"{fname.stem}_mask.png"

def get_classes_per_image(mask_data, class_labels):
    unique = list(np.unique(mask_data))
    result_dict = {}
    for _class in class_labels.keys():
        result_dict[class_labels[_class]] = int(_class in unique)
    return result_dict

def _create_table(image_files, class_labels):
    "Create a table with the dataset"
    labels = [str(class_labels[_lab]) for _lab in list(class_labels)]
    table = wandb.Table(columns=["File_Name", "Images", "Split"] + labels)
    
    for i, image_file in progress_bar(enumerate(image_files), total=len(image_files)):
        image = Image.open(image_file)
        mask_data = np.array(Image.open(label_func(image_file)))
        class_in_image = get_classes_per_image(mask_data, class_labels)
        table.add_data(
            str(image_file.name),
            wandb.Image(
                    image,
                    masks={
                        "predictions": {
                            "mask_data": mask_data,
                            "class_labels": class_labels,
                        }
                    }
            ),
            "None", # we don't have a dataset split yet
            *[class_in_image[_lab] for _lab in labels]
        )
    
    return table 


if __name__ == "__main__":
    path = Path(untar_data(config.URL, force_download=True))
    print(path.ls)

    run = wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.ENTITY,
        job_type="upload"
    )

    raw_data_at = wandb.Artifact(config.RAW_DATA_AT, type="raw_data")

    raw_data_at.add_file(path/'LICENSE.txt', name='LICENSE.txt')

    # Add the images and label masks
    raw_data_at.add_dir(path/'images', name='images')
    raw_data_at.add_dir(path/'labels', name='labels')

    image_files = get_image_files(path/'images', recurse=False)

    table = _create_table(image_files, config.BDD_CLASSES)

    raw_data_at.add(table, "eda_table")

    run.log_artifact(raw_data_at)
    run.finish()
