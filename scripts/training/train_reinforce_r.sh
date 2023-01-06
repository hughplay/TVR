# python train.py \
#     experiment=event_resnet_concat_reinforce_test \
#     criterion.loss.reward_type="acc_dist" \
#     model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
#     logging.wandb.name="event_resnet_concat_best_r_acc_dist" \
#     logging.wandb.tags="[event, reinforce, fix_encoder]" \
#     +model.fix_encoder_weights=true

# python train.py \
#     experiment=event_resnet_concat_reinforce_test \
#     criterion.loss.reward_type="acc_dist" \
#     model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
#     logging.wandb.name="event_resnet_concat_best_r_acc_dist" \
#     logging.wandb.tags="[event, reinforce, rnn_dropout]" \
#     +model.rnn_dropout=0.1

# python train.py \
#     experiment=event_resnet_concat_reinforce_test \
#     criterion.loss.reward_type="acc_dist" \
#     model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
#     logging.wandb.name="event_resnet_concat_best_r_acc_dist" \
#     logging.wandb.tags="[event, reinforce, decoder_dropout]" \
#     +model.decoder_dropout=0.1

# python train.py \
#     experiment=event_resnet_concat_reinforce_test \
#     criterion.loss.reward_type="acc_dist" \
#     model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
#     logging.wandb.name="event_resnet_concat_best_r_acc_dist" \
#     logging.wandb.tags="[event, reinforce, rnn_dropout]" \
#     +model.rnn_dropout=0.3

# python train.py \
#     experiment=event_resnet_concat_reinforce_test \
#     criterion.loss.reward_type="acc_dist" \
#     model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
#     logging.wandb.name="event_resnet_concat_best_r_acc_dist" \
#     logging.wandb.tags="[event, reinforce, decoder_dropout]" \
#     +model.decoder_dropout=0.3

# python train.py \
#     experiment=event_resnet_concat_reinforce_test \
#     criterion.loss.reward_type="acc_dist" \
#     model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
#     logging.wandb.name="event_resnet_concat_best_r_acc_dist" \
#     logging.wandb.tags="[event, reinforce, rnn_dropout]" \
#     +model.rnn_dropout=0.5

# python train.py \
#     experiment=event_resnet_concat_reinforce_test \
#     criterion.loss.reward_type="acc_dist" \
#     model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
#     logging.wandb.name="event_resnet_concat_best_r_acc_dist" \
#     logging.wandb.tags="[event, reinforce, decoder_dropout]" \
#     +model.decoder_dropout=0.5

# python train.py \
#     experiment=event_resnet_concat_reinforce_test \
#     criterion.loss.reward_type="acc_dist" \
#     model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
#     logging.wandb.name="reinforce_best_r_acc_dist" \
#     logging.wandb.tags="[event, reinforce, fix_encoder, rnn_dropout, decoder_dropout]" \
#     +model.fix_encoder_weights=true \
#     +model.rnn_dropout=0.3 \
#     +model.decoder_dropout=0.3

# python train.py \
#     experiment=event_resnet_concat_reinforce_test \
#     criterion.loss.reward_type="acc_dist" \
#     model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
#     logging.wandb.name="reinforce_best_r_acc_dist" \
#     logging.wandb.tags="[event, reinforce, fix_encoder, rnn_dropout, decoder_dropout]" \
#     +model.fix_encoder_weights=true \
#     +model.rnn_dropout=0.3 \
#     +model.decoder_dropout=0.3 \
#     optim.lr=0.001

# python train.py \
#     experiment=event_resnet_concat_reinforce_test \
#     criterion.loss.reward_type="acc_dist" \
#     model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
#     logging.wandb.name="reinforce_best_r_acc_dist" \
#     logging.wandb.tags="[event, reinforce, fix_encoder, rnn_dropout, decoder_dropout]" \
#     +model.fix_encoder_weights=true \
#     +model.rnn_dropout=0.3 \
#     +model.decoder_dropout=0.3 \
#     optim.lr=0.001 \
#     scheduler=step \
#     scheduler.step_size=5

# python train.py \
#     experiment=event_resnet_concat_reinforce_test \
#     criterion.loss.reward_type="acc_dist" \
#     model.pretrained="/log/exp/tvr/TRANCEDataModule.ConcatResNet.ReasonCriterion.2022-12-17_00-30-36/checkpoints/epoch\=025-step\=101582.ckpt" \
#     logging.wandb.name="reinforce_best_r_acc_dist" \
#     logging.wandb.tags="[event, reinforce, fix_encoder, rnn_dropout, decoder_dropout]" \
#     +model.fix_encoder_weights=true \
#     +model.rnn_dropout=0.3 \
#     +model.decoder_dropout=0.3 \
#     optim.lr=0.0001
