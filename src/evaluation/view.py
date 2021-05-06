import torch
import torch.nn.functional as F

from dataset import Vectorizer
from .event import EventEvaluator


class ViewEvaluator(EventEvaluator):

    def evaluate(
            self, obj_choice, pair_choice, init_desc, fin_desc, 
            obj_target, pair_target, view_idx, final_views=[], detail=False):

        loss, loss_obj, loss_pair = self.compute_loss(
            obj_choice, pair_choice, obj_target, pair_target)

        preds = (obj_choice.argmax(dim=2), pair_choice.argmax(dim=2))
        targets = (obj_target, pair_target)

        res = self.eval_tensor_results(
            preds, init_desc, fin_desc, targets, keep_tensor=True)

        n_sample = obj_choice.shape[0]
        info ={
            'n_sample': n_sample,
            'final_views': [],
            'loss': loss.item() * n_sample,
            'loss_obj': loss_obj.item() * n_sample,
            'loss_pair': loss_pair.item() * n_sample,
            'acc': res['correct'].sum().item(),
            'loose_acc': res['loose_correct'].sum().item(),
            'avg_dist': res['dist'].sum().item(),
            'avg_norm_dist': res['norm_dist'].sum().item(),
            'avg_step_diff': (
                res['pred_step'] - res['target_step']).sum().item()
        }

        n_view = len(final_views)
        idx = torch.arange(n_sample).to(obj_target)
        for i, view in enumerate(final_views):
            view = view.split('_')[-1].lower()
            info['final_views'].append(view)
            selected = (view_idx == i)
            info['{}_acc'.format(view)] = res['correct'][selected].sum().item()
            info['{}_loose_acc'.format(view)] = \
                res['loose_correct'][selected].sum().item()
            info['{}_avg_dist'.format(view)] = \
                res['dist'][selected].sum().item()
            info['{}_avg_norm_dist'.format(view)] = \
                res['norm_dist'][selected].sum().item()
            info['{}_avg_step_diff'.format(view)] = (
                res['pred_step'][selected] - res[
                    'target_step'][selected]).sum().item()
            info['n_{}_sample'.format(view)] = selected.sum().item()

        if detail:
            res = {
                k : v.squeeze().tolist() if type(v) is torch.Tensor else v
                for k, v in res.items()}
            info['detail'] = res
            info['detail']['view'] = view_idx.squeeze().tolist()


        return loss, info