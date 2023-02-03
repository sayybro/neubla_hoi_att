
from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from util.fed import load_class_freq, get_fed_loss_inds                       
from .layers.roi_align import ROIAlign
from typing import List
import numpy as np

class DETRHOI(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_classes, num_queries, aux_loss=False, args=None):
        super().__init__()

        
        self.num_queries = num_queries
        self.transformer = transformer        
        hidden_dim = transformer.d_model        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)        
        self.mtl = args.mtl
        
        #multi task
        if args.mtl:
            if 'hico' in args.mtl_data or 'vcoco' in args.mtl_data:
                self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
                if 'vcoco' in args.mtl_data:
                    self.vcoco_verb_class_embed = nn.Linear(hidden_dim, num_classes['vcoco'])
                if 'hico' in args.mtl_data:

                    self.hico_verb_class_embed = nn.Linear(hidden_dim, num_classes['hico']) 
            if 'vaw' in args.mtl_data:
                
                self.att_class_embed = nn.Linear(hidden_dim, num_classes['att'])
  
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1) 
        
        #single-task
        else:
            if args.hoi:
                self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
                self.verb_class_embed = nn.Linear(hidden_dim, num_classes['hoi'])

            elif args.att_det:
                self.att_class_embed = nn.Linear(hidden_dim, num_classes['att'])
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)   
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, args.num_att_classes)


    def forward(self, samples: NestedTensor, targets, dtype: str='', dataset:str=''):
        if not isinstance(samples, NestedTensor): 
            samples = nested_tensor_from_tensor_list(samples)   

        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        if dtype == 'att':

            object_boxes = [torch.Tensor([int(i)]+self.convert_bbox(box.tolist())) for i, target in enumerate(targets) for box in target['boxes']]
            box_tensors = torch.stack(object_boxes,0) #[K,5] , K: box annotation length in mini-batch
            encoder_src = self.input_proj(src)
            B,C,H,W = encoder_src.shape
            encoder_src = encoder_src.flatten(2).permute(2, 0, 1)
            pos_embed = pos[-1].flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)
            memory = self.transformer.encoder(encoder_src, src_key_padding_mask=mask, pos=pos_embed)
            encoder_output = memory.permute(1, 2, 0) 
            encoder_output = encoder_output.view([B,C,H,W]) 
            box_roi_align = ROIAlign(output_size=(7,7), spatial_scale=1.0, sampling_ratio=-1, aligned=True)         
            feature_H, feature_W = encoder_output.shape[2], encoder_output.shape[3]

            #unnormalize box size
            box_tensors[...,1], box_tensors[...,3] = feature_W*box_tensors[...,1], feature_W*box_tensors[...,3] 
            box_tensors[...,2], box_tensors[...,4] = feature_H*box_tensors[...,2], feature_H*box_tensors[...,4] 
            
            pooled_feature = box_roi_align(input = encoder_output, rois = box_tensors.cuda()) 
            x = self.conv(pooled_feature) 
            x = self.avgpool(x) 
            x = torch.flatten(x, 1)
            outputs_class = self.fc(x) 

        if self.mtl:    
            if dtype=='hoi':
                if dataset == 'hico': 
                    outputs_class = self.hico_verb_class_embed(hs) 
                elif dataset =='vcoco':
                    outputs_class = self.vcoco_verb_class_embed(hs) 
            outputs_obj_class = self.obj_class_embed(hs)
            outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()

            if dtype =='att':
                out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_logits': outputs_class,
                        'pred_obj_boxes': outputs_obj_coord[-1],'type': dtype,'dataset':dataset}
                
            else:
                out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_logits': outputs_class[-1],
                        'pred_obj_boxes': outputs_obj_coord[-1],'type': dtype,'dataset':dataset}
            

            if dtype=='hoi':
                outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
                out.update({'pred_sub_boxes': outputs_sub_coord[-1]})
            
            #auxiliary output
            if self.aux_loss:
                if dtype=='hoi':
                    out['aux_outputs'] = self._set_aux_loss_hoi(outputs_obj_class, outputs_class, 
                                                        outputs_sub_coord, outputs_obj_coord)
                elif dtype=='att':
                    out['aux_outputs'] = self._set_aux_loss_att(outputs_obj_class, outputs_class, 
                                                        outputs_obj_coord)
            
        else:

            outputs_obj_class = self.obj_class_embed(hs)
            outputs_verb_class = self.verb_class_embed(hs)
            outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
            outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
            out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_logits': outputs_verb_class[-1],
                'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss_hoi(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord)

        return out

    def convert_bbox(self,bbox:List): #annotation bbox (c_x,c_y,w,h)-> (x1,y1,x2,y2) for roi align
        c_x, c_y, w,h = bbox[0], bbox[1], bbox[2], bbox[3]
        x1,y1 = c_x-(w/2), c_y-(h/2)
        x2,y2 = c_x+(w/2), c_y+(h/2)  
        return [x1,y1,x2,y2]

    @torch.jit.unused
    def _set_aux_loss_hoi(self, outputs_obj_class, outputs_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_class[:-1], 
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
    @torch.jit.unused
    def _set_aux_loss_att(self, outputs_obj_class, outputs_class, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_logits': b, 'pred_obj_boxes': c}
                for a, b, c in zip(outputs_obj_class[:-1], outputs_class[:-1],
                                       outputs_obj_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, loss_type,args=None):
        super().__init__()

        assert loss_type == 'bce' or loss_type == 'focal'

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.loss_type = loss_type
        if args.att_det or 'vaw' in args.mtl_data:
            self.register_buffer('fed_loss_weight', load_class_freq(freq_weight=0.5))

    def loss_obj_labels(self, outputs, targets, indices, num_att_or_inter, dtype, log=True):
        

        if dtype=='att':
            losses={'loss_obj_ce': outputs['pred_obj_logits'].new_zeros([1],dtype=torch.float32)[0]}
            if log:
                losses.update({'obj_class_error':outputs['pred_obj_logits'].new_zeros([1],dtype=torch.float32)[0]})
            return losses

        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o
        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_att_or_inter, dtype):

        if dtype=='att':
            return {'obj_cardinality_error': outputs['pred_obj_logits'].new_zeros([1],dtype=torch.float32)[0]}

        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}

        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_att_or_inter ,dtype):
        if dtype=='att':
            return {'loss_verb_ce': outputs['pred_logits'].new_zeros([1],dtype=torch.float32)[0]}
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        if self.loss_type == 'bce':
            loss_verb_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        elif self.loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._neg_loss(src_logits, target_classes)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_att_or_inter,dtype):
        
        if dtype=='att':
            return {'loss_sub_bbox': outputs['pred_obj_boxes'].new_zeros([1],dtype=torch.float32)[0],
                    'loss_sub_giou': outputs['pred_obj_boxes'].new_zeros([1],dtype=torch.float32)[0],
                    'loss_obj_bbox': outputs['pred_obj_boxes'].new_zeros([1],dtype=torch.float32)[0],
                    'loss_obj_giou': outputs['pred_obj_boxes'].new_zeros([1],dtype=torch.float32)[0]}
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_att_or_inter
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_att_or_inter
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    #attribute loss
    def loss_att_labels(self, outputs, targets, indices, num_att_or_inter,dtype):        
        if dtype=='hoi':
            return {'loss_att_ce': outputs['pred_logits'].new_zeros([1],dtype=torch.float32)[0]}
        
        src_logits = outputs['pred_logits'] 
        target_classes=torch.cat([t['pos_att_classes'] for t in targets])

        pos_gt_classes = torch.nonzero(target_classes==1)[...,-1]
        neg_classes_o = torch.cat([t['neg_att_classes'] for t in targets])
        neg_gt_classes = torch.nonzero(neg_classes_o==1)[...,-1]

        inds = get_fed_loss_inds(
            gt_classes=torch.cat([pos_gt_classes,neg_gt_classes]),
            num_sample_cats=50,
            weight=self.fed_loss_weight,
            C=src_logits.shape[1])
        if self.loss_type == 'bce':
            loss_att_ce = F.binary_cross_entropy_with_logits(src_logits[...,inds], target_classes[...,inds])
        elif self.loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_att_ce = self._neg_loss(src_logits[...,inds], target_classes[...,inds])
        losses = {'loss_att_ce': loss_att_ce}        
        return losses
    
    # def postprocess_att(self, targets):

    #     pos,neg = [], []
    #     for target in targets:
    #         #box label : attribute label = 1 : 1
    #         if (len(target['boxes']) > 0) and (len(target['pos_att_classes']) == (len(target['boxes']))):
    #             pos.append(target['pos_att_classes'])
    #             neg.append(target['neg_att_classes'])

    #         #box label : attribute label = 1 : N
    #         #torch : scatter
    #         if (len(target['boxes']) == 1) and (len(target['pos_att_classes']) > (len(target['boxes']))): 
    #             pos.append(sum(target['pos_att_classes']).unsqueeze(0))
    #             neg.append(sum(target['pos_att_classes']).unsqueeze(0))


    #         if (len(target['boxes']) > 1) and len(target['boxes']) != len(target['pos_att_classes']): 
    #             tmp = torch.from_numpy(np.tile(np.array(-1),(target['boxes'].shape[0],620))).cuda()
    #             pos.append(tmp)
    #             neg.append(tmp)
                
    #     return torch.cat(pos), torch.cat(neg)


    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, dtype,**kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'att_labels': self.loss_att_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, dtype, **kwargs)




    def forward(self, outputs, targets):

        dtype = outputs['type']
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        if dtype == 'att':
            indices = None
        else:
            indices = self.matcher(outputs_without_aux, targets, dtype) #<class 'list'>, 

        num_att_or_inter = sum(len(t['obj_labels']) for t in targets) if outputs['type'] =='hoi' else sum(len(t['labels']) for t in targets)
        num_att_or_inter = torch.as_tensor([num_att_or_inter], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_att_or_inter)
        num_att_or_inter = torch.clamp(num_att_or_inter / get_world_size(), min=1).item()


        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_att_or_inter,dtype))
                


        return losses


class PostProcessHOI_ATT(nn.Module):

    def __init__(self, subject_category_id):
        super().__init__()
        self.subject_category_id = subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        #import pdb; pdb.set_trace()
        if outputs['type'] == 'hoi':
            out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
                                                                            outputs['pred_logits'], \
                                                                            outputs['pred_sub_boxes'], \
                                                                            outputs['pred_obj_boxes']

            # assert len(out_obj_logits) == len(target_sizes)
            # assert target_sizes.shape[1] == 2
            if outputs['dataset']=='vcoco':
                obj_prob = F.softmax(out_obj_logits, -1)
                obj_scores, obj_labels = obj_prob[..., :-1].max(-1)
            elif outputs['dataset']=='hico':
                obj_prob = F.softmax(torch.cat([out_obj_logits[...,:-2],out_obj_logits[...,-1:]],-1), -1)
                obj_scores, obj_labels = obj_prob[..., :-1].max(-1)
            verb_scores = out_verb_logits.sigmoid()
            
            #import pdb; pdb.set_trace()
            img_h, img_w = target_sizes.unbind(1)
        
            #img_h, img_w = target_sizes[1], target_sizes[0]
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
            sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
            sub_boxes = sub_boxes * scale_fct[:, None, :]
            obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
            obj_boxes = obj_boxes * scale_fct[:, None, :]

            results = []
            #import pdb; pdb.set_trace()
            for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
                sl = torch.full_like(ol, self.subject_category_id)
                l = torch.cat((sl, ol))
                b = torch.cat((sb, ob))
                results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

                vs = vs * os.unsqueeze(1)

                ids = torch.arange(b.shape[0])

                results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                    'obj_ids': ids[ids.shape[0] // 2:]})
        elif outputs['type'] == 'att':
            out_obj_logits, out_att_logits,out_obj_boxes = outputs['pred_obj_logits'], \
                                                                            outputs['pred_logits'], \
                                                                            outputs['pred_obj_boxes']


            assert len(out_obj_logits) == len(target_sizes)
            assert target_sizes.shape[1] == 2

            # obj_prob = F.softmax(out_obj_logits, -1)
            obj_prob = F.softmax(torch.cat([out_obj_logits[...,:-2],out_obj_logits[...,-1:]],-1), -1)
            obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

            attr_scores = out_att_logits.sigmoid()

            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(attr_scores.device)
            obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
            obj_boxes = obj_boxes * scale_fct[:, None, :]
            
            results = []
            for ol, ats, ob in zip(obj_labels, attr_scores, obj_boxes):
                # sl = torch.full_like(ol, 0) # self.subject_category_id = 0 in HICO-DET
                # l = torch.cat((sl, ol))
                # b = torch.cat((sb, ob))
                
                
                results.append({'labels': ol.to('cpu'), 'boxes': ob.to('cpu')})
                
                ids = torch.arange(ob.shape[0])
                res_dict = {
                    'attr_scores': ats.to('cpu'),
                    'obj_ids': ids,
                    
                }
                results[-1].update(res_dict)

        return results

    

class PostProcessHOI_orig(nn.Module):
    def __init__(self, subject_category_id):
        super().__init__()
        self.subject_category_id = subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
                                                                        outputs['pred_verb_logits'], \
                                                                        outputs['pred_sub_boxes'], \
                                                                        outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})

        return results