# ckpt of best epoch

python train.py \
    experiment=event_resnet_concat_reinforce_test \
    criterion.loss.reward_type="acc" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_concat_best_r_acc" \
    logging.wandb.tags="[event, reinforce, adamw]" \
    optim.lr=0.001

python train.py \
    experiment=event_resnet_concat_reinforce_test \
    criterion.loss.reward_type="dist" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_concat_best_r_dist" \
    logging.wandb.tags="[event, reinforce, adamw]" \
    optim.lr=0.001

python train.py \
    experiment=event_resnet_concat_reinforce_test \
    criterion.loss.reward_type="acc_dist" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_concat_best_r_acc_dist" \
    logging.wandb.tags="[event, reinforce, adamw]" \
    optim.lr=0.001

python train.py \
    experiment=event_resnet_concat_reinforce_test \
    criterion.loss.reward_type="acc" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_concat_best_r_acc" \
    logging.wandb.tags="[event, reinforce, adamw]" \
    optim.lr=0.0005

python train.py \
    experiment=event_resnet_concat_reinforce_test \
    criterion.loss.reward_type="dist" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_concat_best_r_dist" \
    logging.wandb.tags="[event, reinforce, adamw]" \
    optim.lr=0.0005

python train.py \
    experiment=event_resnet_concat_reinforce_test \
    criterion.loss.reward_type="acc_dist" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_concat_best_r_acc_dist" \
    logging.wandb.tags="[event, reinforce, adamw]" \
    optim.lr=0.0005

python train.py \
    experiment=event_resnet_concat_reinforce_test \
    criterion.loss.reward_type="acc" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_concat_best_r_acc" \
    logging.wandb.tags="[event, reinforce, adamw]" \
    optim.lr=0.0001

python train.py \
    experiment=event_resnet_concat_reinforce_test \
    criterion.loss.reward_type="dist" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_concat_best_r_dist" \
    logging.wandb.tags="[event, reinforce, adamw]" \
    optim.lr=0.0001

python train.py \
    experiment=event_resnet_concat_reinforce_test \
    criterion.loss.reward_type="acc_dist" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_concat_best_r_acc_dist" \
    logging.wandb.tags="[event, reinforce, adamw]" \
    optim.lr=0.0001

python train.py \
    experiment=event_resnet_concat_reinforce_test \
    criterion.loss.reward_type="acc" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_concat_best_r_acc" \
    logging.wandb.tags="[event, reinforce, adamw]" \
    optim.lr=0.00005

python train.py \
    experiment=event_resnet_concat_reinforce_test \
    criterion.loss.reward_type="dist" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_concat_best_r_dist" \
    logging.wandb.tags="[event, reinforce, adamw]" \
    optim.lr=0.00005

python train.py \
    experiment=event_resnet_concat_reinforce_test \
    criterion.loss.reward_type="acc_dist" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_concat_best_r_acc_dist" \
    logging.wandb.tags="[event, reinforce, adamw]" \
    optim.lr=0.00005
