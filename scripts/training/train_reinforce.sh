python train.py \
    experiment=event_resnet_subtract_former_reinforce \
    criterion.loss.reward_type="acc" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.SubtractResNetFormer.ReasonCriterion.2022-12-22_22-38-40/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_subtract_best_r_acc" \
    logging.wandb.tags="[event, reinforce_former]"

python train.py \
    experiment=event_resnet_subtract_former_reinforce \
    criterion.loss.reward_type="dist" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.SubtractResNetFormer.ReasonCriterion.2022-12-22_22-38-40/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_subtract_best_r_dist" \
    logging.wandb.tags="[event, reinforce_former]"

python train.py \
    experiment=event_resnet_subtract_former_reinforce \
    criterion.loss.reward_type="acc_dist" \
    model.pretrained="/log/exp/tvr/TRANCEDataModule.SubtractResNetFormer.ReasonCriterion.2022-12-22_22-38-40/checkpoints/epoch\=025-step\=101582.ckpt" \
    logging.wandb.name="event_resnet_subtract_best_r_acc_dist" \
    logging.wandb.tags="[event, reinforce_former]"
