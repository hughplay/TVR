from collections import OrderedDict
import json
from pathlib import Path
import random
import shutil
import sys

import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

import evaluation
import dataset, model
import utils
from evaluation import recorder


class BaseRunner:

    def __init__(self, config):
        self.config = config

    def init_as_trainer(self, restart=False):
        self.restart = restart
        self.training = True

        self.epoch = 0
        self.step = 0

        self._prepare_path()
        self._prepare_logger()
        self._prepare_recorder()
        self._prepare_tensorboard()
        self._set_seed()
        self._prepare_device()
        self._prepare_train_data()
        self._prepare_model()
        # self._build_model_graph()
        self._prepare_evaluator()
        self._prepare_optimizer()

    def init_as_tester(self):
        self.restart = False
        self.training = False

        self._prepare_path()
        self._prepare_logger()
        self._prepare_recorder()
        self._prepare_device()
        self._prepare_test_data()
        self._prepare_model()
        self._prepare_evaluator()

    @property
    def total_steps(self):
        return self.epoch * len(self.train_data_loader) + self.step

    @property
    def prefix(self):
        return '{}, {}; '.format(self.epoch, self.step)

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    @property
    def data(self):
        if self.training:
            return self.train_dataset
        else:
            return self.test_dataset

    def info(self, message):
        self.logger.info(self.prefix + message)

    def warning(self, message):
        self.logger.warning(self.prefix + message)

    def error(self, message):
        self.logger.error(self.prefix + message, exc_info=True)

    def _prepare_path(self):
        self.output_root = Path(self.config.output_root).expanduser()
        self.output_dir = self.output_root / self.config.data / (
            '{}.{}'.format(self.config.date, self.config.version))
        self.log_dir = self.output_dir / 'log'
        self.ckpt_dir = self.output_dir / 'ckpt'
        self.tb_dir = self.output_dir / 'tb'
        self.result_dir = self.output_dir / 'result'
        self.result_path = self.result_dir / 'result.json'

        self.config.log_dir = str(self.log_dir)
        self.config.ckpt_dir = str(self.ckpt_dir)
        self.config.tb_dir = str(self.tb_dir)
        self.config.result_path = str(self.result_path)
        self.config.save()

        for directory in [
                self.log_dir, self.ckpt_dir, self.tb_dir, self.result_dir]:
            if self.restart:
                shutil.rmtree(directory, True)
            directory.mkdir(parents=True, exist_ok=True)

    def _prepare_logger(self):
        self.logger = utils.get_logger(
            __name__, log_dir=self.config.log_dir, use_tqdm=True)
        self.logger.info(
            '[Path & Logging] Output dir: {}'.format(self.output_dir))

    def _prepare_recorder(self):
        Recorder = getattr(recorder, self.config.recorder)
        if self.training:
            self.r_train = Recorder()
            self.r_val = Recorder()
        else:
            self.r_test = Recorder()

    def _prepare_tensorboard(self):
        self.tb = SummaryWriter(logdir=str(self.tb_dir))
        self.logger.info('[Tensorboard] prepared.')

    def _prepare_device(self):
        self.device = torch.device(self.config.device)
        torch.cuda.set_device(self.device)

        self.logger.info('[Device] {}.'.format(self.device))

    def _prepare_train_data(self):
        self.train_dataset, self.train_data_loader = self._get_data(
            self.config.train_data, self.config.train_data_args,
            self.config.train_data_loader_args)
        self.val_dataset, self.val_data_loader = self._get_data(
            self.config.val_data, self.config.val_data_args,
            self.config.val_data_loader_args)
        self.logger.info('[Data] {}: {} (train), {} (val).'.format(
            self.config.train_data, len(self.train_dataset),
            len(self.val_dataset)))

    def _prepare_test_data(self):
        self.test_dataset, self.test_data_loader = self._get_data(
            self.config.test_data, self.config.test_data_args,
            self.config.test_data_loader_args)
        self.logger.info('[Data] Total test samples: {}.'.format(
            len(self.test_dataset)))

    def _get_data(self, class_name, dataset_args, data_loader_args):
        Dataset = getattr(dataset, class_name)
        data = Dataset(**dataset_args)
        data_loader = DataLoader(data, **data_loader_args)
        return data, data_loader

    def _parse_data(self, sample_batched):
        # return tuple(inputs), tuple(targets)
        raise NotImplementedError

    def _prepare_model(self):
        raise NotImplementedError

    def _build_model_graph(self):
        self.logger.info('Writing model graph to tensorboard...')
        try:
            inputs, _ = self._parse_data(next(iter(self.train_data_loader)))
            self.tb.add_graph(self.model, inputs)
        except Exception as e:
            self.warning('Failed to build model graph.')
            self.warning(str(e))

    def _prepare_evaluator(self):
        Evaluator = getattr(evaluation, self.config.evaluator)
        self.evaluator = Evaluator(**self.config.evaluator_args)
        self.logger.info(
            '[Evaluator] {} has been prepared.'.format(self.config.evaluator))

    def _prepare_optimizer(self):
        Optimizer = getattr(optim, self.config.optimizer)
        self.optimizer = Optimizer(
            self.model.parameters(), **self.config.optimizer_args)
        self.logger.info(
            '[Optimizer] {} has been prepared.'.format(self.config.optimizer))

        if self.config.lr_scheduler:
            Scheduler = getattr(optim.lr_scheduler, self.config.lr_scheduler)
            self.scheduler = Scheduler(
                self.optimizer, **self.config.lr_scheduler_args)
            self.logger.info('[Scheduler] {} has been prepared.'.format(
                self.config.lr_scheduler))
        else:
            self.scheduler = None

    def get_ckpt_path(self, name):
        if Path(name).exists():
            return Path(name)
        else:
            if name.endswith('.pth') or name.endswith('.pt'):
                return self.ckpt_dir / '{}'.format(name)
            else:
                return self.ckpt_dir / '{}.pth'.format(name)

    @property
    def latest_ckpt(self):
        return self.get_ckpt_path('latest')

    @property
    def best_ckpt(self):
        return self.get_ckpt_path('best')

    def _load_ckpt(self, ckpt='best', pretrain=False):
        ckpt = self.get_ckpt_path(ckpt)
        if ckpt.exists():
            if not pretrain:
                self.logger.info(
                    '[Checkpoint] Loading {}...'.format(ckpt.name))
            else:
                self.logger.info(
                    '[Checkpoint] Loading pretrain model {}...'.format(
                        str(ckpt)))
            ckpt_dict = torch.load(ckpt, map_location='cpu')
            self.logger.info(
                '[Checkpoint] {} epoch'.format(ckpt_dict['epoch']))
            self.model.load_state_dict(ckpt_dict['model_state'])
            if self.training and not pretrain:
                self.optimizer.load_state_dict(ckpt_dict['optimizer_state'])
                self.epoch = ckpt_dict['epoch']
                train_loader = self.train_data_loader
                if ckpt_dict['batch_size'] == train_loader.batch_size and \
                        ckpt_dict['step'] == (len(train_loader) - 1):
                    self.epoch += 1
                if self.scheduler is not None:
                    self.scheduler.last_epoch = self.epoch - 1
            return ckpt_dict
        else:
            self.logger.info(
                '[Checkpoint] not found'.format(ckpt))
            return None

    def get_state_dict(self):
        return {
            'time': utils.get_time(),
            'epoch': self.epoch,
            'step': self.step,
            'batch_size': self.train_data_loader.batch_size,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_history': self.r_train.history,
            'val_history': self.r_val.history
        }

    def train(self, ckpt='latest', pretrain=False):
        ckpt_dict = self._load_ckpt(ckpt, pretrain) \
            if not self.restart else None
        if ckpt_dict and not pretrain:
            self.r_train.load(ckpt_dict['train_history'])
            self.r_val.load(ckpt_dict['val_history'])
        self.logger.info('Start point: {} epoch, 0 step.'.format(self.epoch))
        try:
            for self.epoch in trange(
                    self.epoch, self.config.epochs, initial=self.epoch,
                    total=self.config.epochs, desc='Epochs', unit='epoch',
                    ncols=80):
                self._set_seed()
                self._train_epoch()
                self._validate()
                self._save_model()
                self._update_lr()
        except KeyboardInterrupt:
            self.logger.warning('Keyboard Interrupted.')
            sys.exit(1)
        except Exception as e:
            self.logger.error(str(e), exc_info=True)

    def _set_seed(self):
        seed = self.config.seed + self.epoch
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        self.logger.info('[Seed] Random seed is {}.'.format(seed))

    def _train_epoch(self):
        self.r_train.epoch_start()
        for self.step, batch in enumerate(tqdm(
                self.train_data_loader, desc='Train', unit='batch',
                ncols=80, leave=False)):
            info = self._step(batch, backward=True)
            self.r_train.batch_update(info)
            if (self.step + 1) % self.config.log_steps == 0:
                self.show_state(self.r_train, phase='train')
        improvement = self.r_train.epoch_end()
        self.info('[Train] improvement: {:.6f}'.format(improvement))

    def _validate(self):
        self.r_val.epoch_start()
        for batch in tqdm(
                self.val_data_loader, desc='Val', unit='batch',
                ncols=80, leave=False):
            info = self._step(batch)
            self.r_val.batch_update(info)
        self.show_state(self.r_val, phase='val')
        improvement = self.r_val.epoch_end()
        self.info('[Val] improvement: {:.6f}'.format(improvement))

    def _step(self, batch, backward=False):
        raise NotImplementedError

    def show_state(self, recorder, tensorboard=True, phase='train'):
        self.info('[{}] {}'.format(phase.capitalize(), recorder.brief))
        if tensorboard:
            step = self.epoch if phase != 'train' else self.total_steps
            for key, val in recorder.state.items():
                self.tb.add_scalar('{}/{}'.format(phase, key), val, step)
            if phase == 'train':
                self.tb.add_scalar('{}/{}'.format(phase, 'lr'), self.lr, step)

    def _update_lr(self):
        if self.scheduler is not None:
            old_lr = self.lr
            self.scheduler.step()
            if self.lr != old_lr:
                self.info('Change LR: {} => {}'.format(old_lr, self.lr))

    def _save_model(self):
        torch.save(self.get_state_dict(), self.latest_ckpt)
        self.logger.info('Model saved: {}'.format(self.latest_ckpt))
        if self.r_val.is_latest_best():
            shutil.copyfile(self.latest_ckpt, self.best_ckpt)

    def test(self, ckpt='best'):
        if self._load_ckpt(ckpt=ckpt):
            self.r_test.epoch_start()
            for batch in tqdm(
                    self.test_data_loader, desc='Test', unit='batch',
                    ncols=80, leave=False):
                info = self._step(batch, detail=True)
                self.r_test.batch_update(info)
            self.r_test.epoch_end()
            self.logger.info('[Test] {}'.format(self.r_test.brief))
            self._save_test_result(ckpt)
        else:
            self.warning('Model not found.')

    def _save_test_result(self, ckpt='best'):
        with open(self.result_path, 'w') as f:
            json.dump(self.r_test.detail, f, indent=2)


class BasicRunner(BaseRunner):

    def _prepare_model(self):
        Model = getattr(model, self.config.model)
        t = self.data.vectorizer
        self.model = Model(
            c_obj=t.c_obj, n_pair=t.n_pairs, **self.config.model_args)
        self.model = self.model.cuda()
        self.logger.info(
            '[Model] {} has been prepared.'.format(self.config.model))

    def _parse_data(self, batch, training=False):
        init = batch['init'].to(self.device).float().div(255)
        fin = batch['fin'].to(self.device).float().div(255)
        init_desc = batch['init_desc'].to(self.device).float()
        obj_target, pair_target = [x.to(self.device) for x in batch['target']]
        obj_target_vec = batch['obj_target_vec'].to(self.device)
        options = batch['options'].to(self.device)
        if training:
            inputs = (init, fin, init_desc, obj_target_vec)
        else:
            inputs = (init, fin, init_desc)
        targets = (obj_target, pair_target, options)
        return inputs, targets

    def _step(self, batch, backward=False, detail=False):
        if backward:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        inputs, targets = self._parse_data(batch, training=backward)
        obj_choice, pair_choice = self.model(*inputs)
        loss, info = self.evaluator.evaluate(
            obj_choice, pair_choice, *targets, detail=detail)
        if backward:
            loss.backward()
            self.optimizer.step()
        return info

    def _save_test_result(self, ckpt='best'):
        with open(self.result_path, 'w') as f:
            json.dump(self.r_test.detail, f, indent=2)

        result = OrderedDict()
        result['version'] = self.config.version
        result['ckpt'] = ckpt
        result['model_size'] = utils.model_size(self.model)
        result['time'] = utils.get_time()
        for key, val in self.r_test.state.items():
            result[key] = val
        self.config['result'] = dict(result)
        self.config.save()


class EventRunner(BasicRunner):

    def _parse_data(self, batch, training=False):
        init = batch['init'].to(self.device).float().div(255)
        fin = batch['fin'].to(self.device).float().div(255)
        init_desc = batch['init_desc'].to(self.device).float()
        fin_desc = batch['fin_desc'].to(self.device).float()
        obj_target, pair_target = [x.to(self.device) for x in batch['target']]
        obj_target_vec = batch['obj_target_vec'].to(self.device)
        if training:
            inputs = (init, fin, init_desc, obj_target_vec, pair_target)
        else:
            inputs = (init, fin, init_desc)
        targets = (init_desc, fin_desc, obj_target, pair_target)
        return inputs, targets


    def _step(self, batch, backward=False, detail=False):
        if backward:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        inputs, targets = self._parse_data(batch, training=backward)
        obj_choice, pair_choice = self.model(*inputs)
        loss, info = self.evaluator.evaluate(
            obj_choice, pair_choice, *targets, detail=detail)
        if backward:
            loss.backward()
            self.optimizer.step()
        return info

    def test(self, ckpt='best'):
        self.r_test.enable_step_result = True
        super().test(ckpt)


class ViewRunner(EventRunner):
    def _step(self, batch, backward=False, detail=False):
        if backward:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        inputs, targets = self._parse_data(batch, training=backward)
        obj_choice, pair_choice = self.model(*inputs)
        loss, info = self.evaluator.evaluate(
            obj_choice, pair_choice, *targets, batch['view'],
            final_views=self.data.final_views, detail=detail)
        if backward:
            loss.backward()
            self.optimizer.step()
        return info


class PretrainEventRunner(EventRunner):

    def _load_ckpt(self, ckpt='best', pretrain=False):
        ckpt = self.get_ckpt_path(ckpt)
        if ckpt.exists():
            if not pretrain:
                self.logger.info(
                    '[Checkpoint] Loading {}...'.format(ckpt.name))
            else:
                self.logger.info(
                    '[Checkpoint] Loading pretrain model {}...'.format(
                        str(ckpt)))
            ckpt_dict = torch.load(ckpt, map_location='cpu')
            if pretrain:
                self.model.encoder.load_state_dict(ckpt_dict['encoder_state'])
            else:
                self.logger.info(
                    '[Checkpoint] {} epoch'.format(ckpt_dict['epoch']))
                self.model.load_state_dict(ckpt_dict['model_state'])
            if self.training and not pretrain:
                self.optimizer.load_state_dict(ckpt_dict['optimizer_state'])
                self.epoch = ckpt_dict['epoch']
                train_loader = self.train_data_loader
                if ckpt_dict['batch_size'] == train_loader.batch_size and \
                        ckpt_dict['step'] == (len(train_loader) - 1):
                    self.epoch += 1
                if self.scheduler is not None:
                    self.scheduler.last_epoch = self.epoch - 1
            return ckpt_dict
        else:
            self.logger.info('[Checkpoint] not found'.format(ckpt))
            return None


class PretrainFixedEventRunner(PretrainEventRunner):

    def _prepare_model(self):
        Model = getattr(model, self.config.model)
        t = self.data.vectorizer
        self.model = Model(
            c_obj=t.c_obj, n_pair=t.n_pairs, **self.config.model_args)
        self.model = self.model.cuda()
        self.logger.info(
            '[Model] {} has been prepared.'.format(self.config.model))
        for param in self.model.encoder.parameters():
            param.requires_grad = False