import torch
import torch.nn.functional as F

from dataset import Vectorizer

class BasicEvaluator:
    def __init__(
            self, values_json='../data/gen_src/resource/values.json',
            properties_json='../data/gen_src/resource/properties.json'):

        self.t = Vectorizer(
            values_json=values_json, properties_json=properties_json)
        self.pair2attr = torch.tensor([
            self.t.attrs.index(pair.split(self.t.SPLIT)[0])
            for pair in self.t.pairs])

    def evaluate(
            self, obj_choice, pair_choice, obj_target, pair_target, options,
            detail=False):
        loss_obj = F.nll_loss(obj_choice, obj_target)
        loss_pair = F.nll_loss(pair_choice, pair_target)
        loss = loss_obj + loss_pair

        obj_pred_idx = obj_choice.argmax(dim=1, keepdim=True)
        obj_target_idx = obj_target.view_as(obj_pred_idx)
        obj_correct = obj_pred_idx.eq(obj_target_idx)

        pair_pred_vec, pair_pred_idx = pair_choice.max(dim=1, keepdim=True)
        pair_pred_vec = pair_choice.eq(pair_pred_vec)
        pair_target_idx = pair_target.view_as(pair_pred_idx)
        pair_correct = (options * pair_pred_vec).sum(dim=1, keepdim=True) >= 1

        attr_pred_idx = self.pair2attr.to(pair_pred_idx)[pair_pred_idx]
        attr_target_idx = self.pair2attr.to(pair_target_idx)[pair_target_idx]
        attr_correct = attr_pred_idx.eq(attr_target_idx)

        correct = obj_correct * pair_correct

        n_sample = obj_choice.shape[0]
        info = {
            'n_sample': n_sample,
            'loss': loss.item() * n_sample,
            'loss_obj': loss_obj.item() * n_sample,
            'loss_pair': loss_pair.item() * n_sample,
            'acc': correct.sum().item(),
            'acc_obj': obj_correct.sum().item(),
            'acc_attr': attr_correct.sum().item(),
            'acc_pair': pair_correct.sum().item(),
        }
        if detail:
            info['detail'] = {
                'correct': correct.squeeze().tolist(),
                'obj_correct': obj_correct.squeeze().tolist(),
                'pair_correct': pair_correct.squeeze().tolist(),
                'obj_pred': obj_pred_idx.squeeze().tolist(),
                'obj_target': obj_target_idx.squeeze().tolist(),
                'pair_pred': pair_pred_idx.squeeze().tolist(),
                'pair_target': pair_target_idx.squeeze().tolist(),
                'attr_pred': attr_pred_idx.squeeze().tolist(),
                'attr_target': attr_target_idx.squeeze().tolist(),
            }

        return loss, info