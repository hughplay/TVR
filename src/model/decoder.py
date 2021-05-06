import torch
from torch import nn
import torch.nn.functional as F

from .block import FCBlocks


class ObjectMatcher(nn.Module):

    def __init__(
            self, c_in, c_obj=33, c_img_en=[1024, 256], act='relu',
            dropout=None, log_softmax=True):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_img_en + [c_obj]
        self.act = act
        self.dropout = dropout
        self.log_softmax = log_softmax

        self.predictor = FCBlocks(
            self.c_in, self.c_out, act=self.act, dropout=self.dropout)

    def forward(self, objects, img_feat):
        obj_vec = self.predictor(img_feat)
        similarity = (objects.bmm(obj_vec.unsqueeze(dim=2))).squeeze()
        if self.log_softmax:
            choice = F.log_softmax(similarity, dim=1)
        else:
            choice = F.softmax(similarity, dim=1)
        idx = torch.argmax(similarity, dim=1)
        selected_obj_vec = objects[torch.arange(objects.size(0)), idx]
        return choice, selected_obj_vec


class ActionClassifier(nn.Module):

    def __init__(
            self, c_in, c_obj, n_pair, c_img_en=[1024, 256],
            c_fusion=[128, 64], act='relu', dropout=None, log_softmax=True):
        super().__init__()

        self.c_img_en_in = c_in
        self.c_img_en = c_img_en
        self.c_fusion_in = c_img_en[-1] + c_obj
        self.c_fusion = c_fusion + [n_pair]
        self.act = act
        self.dropout = dropout
        self.log_softmax = log_softmax

        self.img_en = FCBlocks(
            self.c_img_en_in, self.c_img_en, act=self.act,
            dropout=self.dropout)
        self.fusion = FCBlocks(
            self.c_fusion_in, self.c_fusion, act=self.act,
            dropout=self.dropout)

    def forward(self, img_feat, pred_obj):
        out = self.img_en(img_feat)
        out = torch.cat([out, pred_obj], dim=1)
        out = self.fusion(out)
        if self.log_softmax:
            out = F.log_softmax(out, dim=1)
        else:
            out = F.softmax(out, dim=1)
        return out


class TransformationDecoder(nn.Module):

    def __init__(
            self, c_in, c_obj, n_pair, c_img_en, c_fusion,
            act='relu', dropout=None, log_softmax=True):
        super().__init__()
        self.matcher = ObjectMatcher(
            c_in, c_obj, c_img_en, act, dropout, log_softmax)
        self.classifier = ActionClassifier(
            c_in, c_obj, n_pair, c_img_en, c_fusion, act, dropout, log_softmax)

    def forward(
            self, objects, img_feat, output_obj_vec=False,
            obj_target_vec=None):
        obj_choice, selected_obj_vec = self.matcher(objects, img_feat)
        if type(obj_target_vec) is torch.Tensor:
            selected_obj_vec = obj_target_vec
        pair_choice = self.classifier(img_feat, selected_obj_vec)
        if output_obj_vec:
            return obj_choice, selected_obj_vec, pair_choice
        else:
            return obj_choice, pair_choice


class JointTransformationDecoder(nn.Module):
    def __init__(
            self, c_in, c_obj, n_pair, n_obj, c_img_en, c_fusion, c_extra=0,
            act='relu', dropout=None, log_softmax=True):
        super().__init__()
        self.log_softmax = log_softmax
        self.en = FCBlocks(c_in + c_obj * n_obj, c_img_en)
        c_out = c_fusion + [n_obj * n_pair + c_extra]
        self.classifier = FCBlocks(c_img_en[-1], c_out)

    def forward(self, objects, img_feat, output_obj_vec=False):
        out = self.en(img_feat)
        objects = torch.flatten(objects, start_dim=1)
        out = torch.cat([out, objects], dim=1)
        out = self.classifier(out)
        if self.log_softmax:
            out = F.log_softmax(out, dim=1)
        else:
            out = F.softmax(out, dim=1)
        return out
