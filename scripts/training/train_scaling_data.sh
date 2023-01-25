# basic

python train.py \
    experiment=basic_resnet_subtract \
    logging.wandb.tags="[basic, scale_data]" \
    dataset.dataset_cfg.train.n_samples=10000

python train.py \
    experiment=basic_resnet_subtract \
    logging.wandb.tags="[basic, scale_data]" \
    dataset.dataset_cfg.train.n_samples=50000


# event

python train.py \
    experiment=event_resnet_subtract_former \
    logging.wandb.tags="[event, scale_data]" \
    dataset.dataset_cfg.train.n_samples=10000

python train.py \
    experiment=event_resnet_subtract_former \
    logging.wandb.tags="[event, scale_data]" \
    dataset.dataset_cfg.train.n_samples=50000

python train.py \
    experiment=event_resnet_subtract_former \
    logging.wandb.tags="[event, scale_data]" \
    dataset.dataset_cfg.train.n_samples=100000

python train.py \
    experiment=event_resnet_subtract_former \
    logging.wandb.tags="[event, scale_data]" \
    dataset.dataset_cfg.train.n_samples=200000

python train.py \
    experiment=event_resnet_subtract_former \
    logging.wandb.tags="[event, scale_data]" \
    dataset.dataset_cfg.train.n_samples=300000

python train.py \
    experiment=event_resnet_subtract_former \
    logging.wandb.tags="[event, scale_data]" \
    dataset.dataset_cfg.train.n_samples=400000


# view

python train.py \
    experiment=view_resnet_subtract_former \
    logging.wandb.tags="[view, scale_data]" \
    dataset.dataset_cfg.train.n_samples=10000

python train.py \
    experiment=view_resnet_subtract_former \
    logging.wandb.tags="[view, scale_data]" \
    dataset.dataset_cfg.train.n_samples=50000

python train.py \
    experiment=view_resnet_subtract_former \
    logging.wandb.tags="[view, scale_data]" \
    dataset.dataset_cfg.train.n_samples=100000

python train.py \
    experiment=view_resnet_subtract_former \
    logging.wandb.tags="[view, scale_data]" \
    dataset.dataset_cfg.train.n_samples=200000

python train.py \
    experiment=view_resnet_subtract_former \
    logging.wandb.tags="[view, scale_data]" \
    dataset.dataset_cfg.train.n_samples=300000

python train.py \
    experiment=view_resnet_subtract_former \
    logging.wandb.tags="[view, scale_data]" \
    dataset.dataset_cfg.train.n_samples=400000
