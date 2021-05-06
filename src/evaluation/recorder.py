from collections import defaultdict, OrderedDict
import time

import numpy as np


class BasicRecorder:

    def __init__(self):
        self.history = defaultdict(list)
        self.t_total = 0

    @property
    def state(self):
        return {
            'loss': self.get_average('loss'),
            'loss_obj': self.get_average('loss_obj'),
            'loss_pair': self.get_average('loss_pair'),
            'acc': self.get_average('acc'),
            'acc_obj': self.get_average('acc_obj'),
            'acc_pair': self.get_average('acc_pair'),
            'acc_attr': self.get_average('acc_attr'),
        }

    @property
    def brief(self):
        return (
            'loss: {loss:.4f} ({loss_obj:.4f}; {loss_pair:.4f}),'
            ' acc: {acc:.4f} ({acc_obj:.4f}, {acc_attr:.4f}, '
            '{acc_pair:.4f})').format(**self.state)

    @property
    def total_time(self):
        return sum(self.history['time'])

    @property
    def best_result(self):
        if len(self.history['acc']) > 0:
            return max(self.history['acc'])
        else:
            return 0.

    def get_average(self, key):
        return sum(self.metrics[key]) / sum(self.metrics['n_sample'])

    def load(self, history):
        self.history.update(history)

    def epoch_start(self):
        self.t_start = time.time()
        self.metrics = defaultdict(list)
        self.detail = defaultdict(list)

    def epoch_end(self):
        last_best_acc = self.best_result
        for key, val in self.state.items():
            self.history[key].append(val)
        self.history['time'].append(time.time() - self.t_start)
        self.history['n_sample'].append(sum(self.metrics['n_sample']))
        improvement = self.history['acc'][-1] - last_best_acc
        return improvement

    def batch_update(self, info):
        for key, val in info.items():
            if key == 'detail':
                for k, v in val.items():
                    self.detail[k].extend(v)
            else:
                self.metrics[key].append(val)

    def is_latest_best(self):
        return self.history['acc'][-1] == self.best_result


class ViewRecorder(BasicRecorder):

    @property
    def state(self):
        state = OrderedDict({
            'loss': self.get_average('loss'),
            'loss_obj': self.get_average('loss_obj'),
            'loss_pair': self.get_average('loss_pair'),
            'acc': self.get_average('acc'),
            'acc_obj': self.get_average('acc_obj'),
            'acc_pair': self.get_average('acc_pair'),
            'acc_attr': self.get_average('acc_attr'),
        })
        for view in self.metrics['final_views']:
            for key in ['acc', 'acc_obj', 'acc_pair', 'acc_attr']:
                view_metric = '{}_{}'.format(view, key)
                state[view_metric] = sum(self.metrics[view_metric]) / sum(
                    self.metrics['n_{}_sample'.format(view)])
        return dict(state)

    def batch_update(self, info):
        for key, val in info.items():
            if key == 'detail':
                for k, v in val.items():
                    self.detail[k].extend(v)
            elif key == 'final_views':
                self.metrics[key] = val
            else:
                self.metrics[key].append(val)

    @property
    def brief(self):
        view_acc_str = ' '.join([
            '{}: {{{}_acc:.4f}}'.format(view[0], view)
            for view in self.metrics['final_views']])
        return (
            'loss: {loss:.4f}, acc: {acc:.4f} ({acc_obj:.4f}, '
            '{acc_attr:.4f}, {acc_pair:.4f}) ' + view_acc_str).format(
                **self.state)

    def epoch_end(self):
        last_best_acc = self.best_result
        for key, val in self.state.items():
            self.history[key].append(val)
        self.history['time'].append(time.time() - self.t_start)
        self.history['n_sample'].append(sum(self.metrics['n_sample']))
        self.history['final_views'] = self.metrics['final_views']
        self.detail['final_views'] = self.metrics['final_views']
        improvement = self.history['acc'][-1] - last_best_acc
        return improvement


class EventRecorder(BasicRecorder):

    def __init__(self):
        super().__init__()
        self.enable_step_result = False

    @property
    def state(self):
        state = OrderedDict({
            'loss': self.get_average('loss'),
            'loss_obj': self.get_average('loss_obj'),
            'loss_pair': self.get_average('loss_pair'),
            'acc': self.get_average('acc'),
            'loose_acc': self.get_average('loose_acc'),
            'avg_dist': self.get_average('avg_dist'),
            'avg_norm_dist': self.get_average('avg_norm_dist'),
            'avg_step_diff': self.get_average('avg_step_diff')
        })
        if self.enable_step_result:
            steps = np.unique(np.array(self.detail['target_step'])).tolist()
            step_result = defaultdict(dict)
            for step in steps:
                idx = (np.array(self.detail['target_step']) == step)
                step_result[step]['acc'] = float(np.mean(np.array(
                    self.detail['correct'])[idx]))
                step_result[step]['loose_acc'] = float(np.mean(np.array(
                    self.detail['loose_correct'])[idx]))
                step_result[step]['avg_dist'] = float(np.mean(np.array(
                    self.detail['dist'])[idx]))
                step_result[step]['avg_norm_dist'] = float(np.mean(np.array(
                    self.detail['norm_dist'])[idx]))
            state['step_result'] = dict(step_result)
        return dict(state)

    @property
    def brief(self):
        return (
            'loss: {loss:.4f}, acc: {acc:.4f} ({loose_acc:.4f}), '
            'AD: {avg_dist:.4f}, AND: {avg_norm_dist:.4f}, '
            'step_diff: {avg_step_diff:.2f}'.format(**self.state))

    def epoch_end(self):
        last_best_acc = self.best_result
        for key, val in self.state.items():
            self.history[key].append(val)
        self.history['time'].append(time.time() - self.t_start)
        self.history['n_sample'].append(sum(self.metrics['n_sample']))
        improvement = self.history['acc'][-1] - last_best_acc
        return improvement


class MultiViewRecorder(EventRecorder):

    @property
    def state(self):
        state = OrderedDict({
            'loss': self.get_average('loss'),
            'loss_obj': self.get_average('loss_obj'),
            'loss_pair': self.get_average('loss_pair'),
            'acc': self.get_average('acc'),
            'loose_acc': self.get_average('loose_acc'),
            'avg_dist': self.get_average('avg_dist'),
            'avg_norm_dist': self.get_average('avg_norm_dist'),
            'avg_step_diff': self.get_average('avg_step_diff')
        })
        for view in self.metrics['final_views']:
            for key in [
                    'acc', 'loose_acc', 'avg_dist', 'avg_norm_dist',
                    'avg_step_diff']:
                view_metric = '{}_{}'.format(view, key)
                state[view_metric] = sum(self.metrics[view_metric]) / sum(
                    self.metrics['n_{}_sample'.format(view)])
        if self.enable_step_result:
            steps = np.unique(np.array(self.detail['target_step'])).tolist()
            step_result = defaultdict(dict)
            for step in steps:
                idx = (np.array(self.detail['target_step']) == step)
                step_result[step]['acc'] = float(np.mean(np.array(
                    self.detail['correct'])[idx]))
                step_result[step]['loose_acc'] = float(np.mean(np.array(
                    self.detail['loose_correct'])[idx]))
                step_result[step]['avg_dist'] = float(np.mean(np.array(
                    self.detail['dist'])[idx]))
                step_result[step]['avg_norm_dist'] = float(np.mean(np.array(
                    self.detail['norm_dist'])[idx]))
                step_result[step]['avg_step_diff'] = float(np.mean(
                    np.array(self.detail['pred_step'])[idx]
                    - np.array(self.detail['target_step'])[idx]))

                step_result[step]['view'] = defaultdict(OrderedDict)
                for i, view in enumerate(self.metrics['final_views']):
                    v_idx = idx * (np.array(self.detail['view']) == i)
                    step_result[step]['view'][view]['acc'] = float(
                        np.mean(np.array(self.detail['correct'])[v_idx]))
                    step_result[step]['view'][view]['loose_acc'] = float(
                        np.mean(np.array(self.detail['loose_correct'])[v_idx]))
                    step_result[step]['view'][view]['avg_dist'] = float(
                        np.mean(np.array(self.detail['dist'])[v_idx]))
                    step_result[step]['view'][view]['avg_norm_dist'] = float(
                        np.mean(np.array(self.detail['norm_dist'])[v_idx]))
                    step_result[step]['view'][view]['avg_step_diff'] = float(
                        np.mean(np.array(self.detail['pred_step'])[v_idx]
                            - np.array(self.detail['target_step'])[v_idx]))
                    step_result[step]['view'][view] = dict(
                        step_result[step]['view'][view])
                step_result[step]['view'] = dict(step_result[step]['view'])
            state['step_result'] = dict(step_result)
        return dict(state)

    def batch_update(self, info):
        for key, val in info.items():
            if key == 'detail':
                for k, v in val.items():
                    self.detail[k].extend(v)
            elif key == 'final_views':
                self.metrics[key] = val
            else:
                self.metrics[key].append(val)

    @property
    def brief(self):
        return (
            'loss: {loss:.4f}, acc: {acc:.4f} ({loose_acc:.4f}), '
            'AD: {avg_dist:.4f}, AND: {avg_norm_dist:.4f}, '
            'step_diff: {avg_step_diff:.2f}'.format(**self.state))

    def epoch_end(self):
        last_best_acc = self.best_result
        for key, val in self.state.items():
            self.history[key].append(val)
        self.history['time'].append(time.time() - self.t_start)
        self.history['n_sample'].append(sum(self.metrics['n_sample']))
        self.history['final_views'] = self.metrics['final_views']
        self.detail['final_views'] = self.metrics['final_views']
        improvement = self.history['acc'][-1] - last_best_acc
        return improvement


class ReinforceEventRecorder(BasicRecorder):

    @property
    def state(self):
        state = OrderedDict({
            'loss': self.get_average('loss'),
            'loss_obj': self.get_average('loss_obj'),
            'loss_pair': self.get_average('loss_pair'),
            'reward': self.get_average('reward'),
            'acc': self.get_average('acc'),
            'loose_acc': self.get_average('loose_acc'),
            'avg_dist': self.get_average('avg_dist'),
            'avg_norm_dist': self.get_average('avg_norm_dist'),
            'avg_step_diff': self.get_average('avg_step_diff')
        })
        return dict(state)

    @property
    def brief(self):
        return (
            'loss: {loss:.4f}, reward: {reward:.4f}, acc: {acc:.4f} '
            '({loose_acc:.4f}), AD: {avg_dist:.4f}, AND: '
            '{avg_norm_dist:.4f}'.format(**self.state))

    def epoch_end(self):
        last_best_acc = self.best_result
        for key, val in self.state.items():
            self.history[key].append(val)
        self.history['time'].append(time.time() - self.t_start)
        self.history['n_sample'].append(sum(self.metrics['n_sample']))
        improvement = self.history['acc'][-1] - last_best_acc
        return improvement