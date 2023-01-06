# Basic half precision

python train.py experiment=basic_cnn_concat logging.wandb.tags="[basic, half]" pl_trainer.precision=16

python train.py experiment=basic_cnn_subtract logging.wandb.tags="[basic, half]" pl_trainer.precision=16

python train.py experiment=basic_bcnn logging.wandb.tags="[basic, half]" pl_trainer.precision=16

python train.py experiment=basic_duda logging.wandb.tags="[basic, half]" pl_trainer.precision=16

python train.py experiment=basic_resnet_concat logging.wandb.tags="[basic, half]" pl_trainer.precision=16

python train.py experiment=basic_resnet_subtract logging.wandb.tags="[basic, half]" pl_trainer.precision=16


# Event half precision

python train.py experiment=event_cnn_concat logging.wandb.tags="[event, half]" pl_trainer.precision=16

python train.py experiment=event_cnn_subtract logging.wandb.tags="[event, half]" pl_trainer.precision=16

python train.py experiment=event_bcnn logging.wandb.tags="[event, half]" pl_trainer.precision=16

python train.py experiment=event_duda logging.wandb.tags="[event, half]" pl_trainer.precision=16

python train.py experiment=event_resnet_concat logging.wandb.tags="[event, half]" pl_trainer.precision=16

python train.py experiment=event_resnet_subtract logging.wandb.tags="[event, half]" pl_trainer.precision=16


# View half precision

python train.py experiment=view_cnn_concat logging.wandb.tags="[view, half]" pl_trainer.precision=16

python train.py experiment=view_cnn_subtract logging.wandb.tags="[view, half]" pl_trainer.precision=16

python train.py experiment=view_bcnn logging.wandb.tags="[view, half]" pl_trainer.precision=16

python train.py experiment=view_duda logging.wandb.tags="[view, half]" pl_trainer.precision=16

python train.py experiment=view_resnet_concat logging.wandb.tags="[view, half]" pl_trainer.precision=16

python train.py experiment=view_resnet_subtract logging.wandb.tags="[view, half]" pl_trainer.precision=16
