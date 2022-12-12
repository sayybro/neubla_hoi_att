import argparse
import cv2
import os
from models import build_model
from main import get_args_parser
import torch
import numpy as np
from PIL import Image
import itertools
import util.misc as utils
from datasets.vcoco import make_vcoco_transforms
import pandas as pd
from index2cat import vcoco_index_2_cat, hico_index_2_cat, vaw_index_2_cat
import json


def hoi_att_transforms(image_set):
    transforms = make_vcoco_transforms(image_set)
    return transforms


def inference_for_vid(model, frame, args=None):

    img = Image.fromarray(frame)
    transform = hoi_att_transforms('val')
    sample = img.copy()
    sample, _ = transform(sample, None)
    dataset = args.inf_type #hico or vcoco or hoi or vaw
    if dataset == 'vaw':
        dtype = 'att'
    else: 
        dtype = 'hoi'
    
    if 'hico' in dataset and 'vcoco' in dataset:
        output_hico = model(sample.unsqueeze(0), dtype, 'hico')
        output_vcoco = model(sample.unsqueeze(0), dtype, 'vcoco')
        return output_hico, output_vcoco
    else:
        output = model(sample.unsqueeze(0), dtype, dataset)
        return output


def valid_att_idxs(anno_file):
    #import pdb; pdb.set_trace()
    with open(anno_file, 'r') as f:
        annotations = json.load(f)
    num_attributes = 620
    valid_masks = np.zeros(num_attributes)

    for i in annotations:
        if i['instance_count'] > 0:
            valid_masks[i['id']]=1
    return valid_masks


def change_format(results, args):

    if args.inf_type == 'vcoco' or args.inf_type == 'hico' or ('hico' in args.inf_type and 'vcoco' in args.inf_type):
        boxes,labels,pair_scores = list(map(lambda x: x.cpu().numpy(), [results['boxes'], results['labels'], results['verb_scores']]))
        output_i={}
        output_i['box_predictions']=[]
        output_i['hoi_predictions']=[]
        
        h_boxes = boxes[:100]
        h_labels = labels[:100]
        o_boxes = boxes[100:]
        o_labels = labels[100:]

        for h_box, o_box in zip(h_boxes, o_boxes):
            output_i['box_predictions'].append({'h_bbox':h_box.tolist(), 'o_bbox': o_box.tolist()})

        for h_label, o_label, pair_score in zip(h_labels, o_labels, pair_scores):
            output_i['hoi_predictions'].append({'subject_id':h_label,'object_id':o_label,'max_score':max(pair_score),'pair_score':pair_score})
    
    else:
        #import pdb; pdb.set_trace()
        boxes,labels,attr_scores = list(map(lambda x: x.cpu().numpy(), [results['boxes'], results['labels'], results['attr_scores']]))
        output_i={}
        output_i['predictions']=[]
        attribute_freq = 'data/vaw/annotations/vaw_coco_train_cat_info.json'
        valid_masks = valid_att_idxs(attribute_freq)
        for box, label,attr_score in zip(boxes,labels,attr_scores):
            attr_score *= valid_masks
            output_i['predictions'].append({'bbox':box.tolist(), 'object_id': label, 'max_score':max(attr_score), 'pair_score':attr_score})
    return output_i


def index_2_cat(index,inf_type):
    if inf_type =='vcoco':
        return vcoco_index_2_cat(index)
    elif inf_type =='hico':
        return hico_index_2_cat(index)
    elif inf_type =='vaw':
        return vaw_index_2_cat(index)

def make_color_dict(class_num):
    color_dict = {i: list(np.random.random(size=3) * 256) for i in range(class_num)}
    return color_dict 


def draw_img(img, output_i, top_k, threshold, color_dict, inf_type):
    #import pdb; pdb.set_trace()
    vis_img = img.copy()

    #for attribute inference
    if inf_type == 'vaw':
        list_predictions = []
        for predict in output_i['predictions']:

            #prediction threshold
            if predict['max_score'] < threshold:
                continue
            object_id = predict['object_id']
            object_box = predict['bbox']
            attr_score = predict['pair_score']
            max_attr_score = predict['max_score']
            single_out = {'object_box':np.array(object_box), 'object_id':np.array(object_id), 'attr_score':np.array(attr_score), 'max_attr_score':np.array(max_attr_score)}
            list_predictions.append(single_out)
        if list_predictions:
            df = pd.DataFrame(list_predictions)
            df = df.sort_values(by=['max_attr_score'],ascending=False).iloc[:top_k]
            list_predictions = df.to_dict('records')
            #import pdb; pdb.set_trace()
            for i, prediction in enumerate(list_predictions):
                attributes = np.where(prediction['attr_score'] > threshold)
                o_bbox = prediction['object_box']
                o_class = prediction['object_id']                
                vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                print(f'drawing object box')
                
                
                #import pdb; pdb.set_trace()
                text_size, BaseLine=cv2.getTextSize(index_2_cat(attributes[0][0],args.inf_type),cv2.FONT_HERSHEY_SIMPLEX,1,2)

                #text height for multiple attributes
                text_size_y = text_size[1] 
                cnt = 1
                for attr in attributes[0]:
                    text = index_2_cat(attr,args.inf_type)
                    text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                    if o_bbox[1]-cnt*text_size_y < 0 or o_bbox[0] < 0:
                        break
                    text_box = [o_bbox[0], o_bbox[1]-cnt*text_size_y, o_bbox[0]+text_size[0],o_bbox[1]-(cnt-1)*text_size_y]

                    #draw text
                    vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                    vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                    print(f'drawing attribute box : {text}, {cnt}')
                    cnt += 1 
    
    #for hoi inference
    elif ('vcoco' in inf_type) and ('hico' in inf_type):
        actions = {'hico':[],'vcoco':[]}
        for output in output_i:
            for box, hoi in zip(output['box_predictions'], output['hoi_predictions']):
                #prediction threshold
                if hoi['max_score'] < threshold:
                    continue
                
                subject_id = hoi['subject_id']
                subject_box = box['h_bbox']
                object_id = hoi['object_id']
                object_box = box['o_bbox']
                verb_score = hoi['pair_score']
                max_verb_score = hoi['max_score']
                single_out = {'subject_box':np.array(subject_box), 'object_box':np.array(object_box), 'subject_id':np.array(subject_id), 'object_id':np.array(object_id), 'verb_score':np.array(verb_score), 'max_verb_score':np.array(max_verb_score)}
                if single_out['verb_score'].shape == (117,): #hico
                    actions['hico'].append(single_out)
                elif single_out['verb_score'].shape == (29,): #vcoco
                    actions['vcoco'].append(single_out)
                
            if len(actions['hico']) > 0 and len(actions['vcoco']) > 0:
                #import pdb; pdb.set_trace()
                hico,vcoco = pd.DataFrame(actions['hico']), pd.DataFrame(actions['vcoco'])
                hico,vcoco = hico.sort_values(by=['max_verb_score'],ascending=False).iloc[:top_k], vcoco.sort_values(by=['max_verb_score'],ascending=False).iloc[:top_k] 
                hico_actions = hico.to_dict('records')
                vcoco_actions = vcoco.to_dict('records')
                #import pdb; pdb.set_trace()
                
                for i, action in enumerate(hico_actions):
                    verbs = np.where(action['verb_score'] > threshold)
                    s_bbox = action['subject_box']
                    o_bbox = action['object_box']
                    o_class = action['object_id']
                    #import pdb; pdb.set_trace()
                    vis_img = cv2.rectangle(vis_img, (int(s_bbox[0]),int(s_bbox[1])), (int(s_bbox[2]),int(s_bbox[3])), color_dict[0], 3)
                    print('drawing subject box')
                    
                    vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                    print('drawing object box')
                    #print(int(o_class))
                    text_size, BaseLine=cv2.getTextSize(index_2_cat(verbs[0][0],'hico'),cv2.FONT_HERSHEY_SIMPLEX,1,2)
                    
                    #text height for multiple verbs (interactions)
                    text_size_y = text_size[1] 
                    cnt = 1
                    for verb in verbs[0]:
                        
                        text = index_2_cat(verb,'hico')
                        text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                        if s_bbox[1]-cnt*text_size_y < 0 or s_bbox[0] < 0:
                            break
                        text_box = [s_bbox[0], s_bbox[1]-cnt*text_size_y, s_bbox[0]+text_size[0],s_bbox[1]-(cnt-1)*text_size_y]

                        #draw text
                        vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                        vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                        print(f'drawing hico action box : {text}, {cnt}')
                        cnt += 1 
                    
                for i, action in enumerate(vcoco_actions):
                    verbs = np.where(action['verb_score'] > threshold)
                    for verb in verbs[0]:
                        
                        text = index_2_cat(verb,'vcoco')
                        text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                        if s_bbox[1]-cnt*text_size_y < 0 or s_bbox[0] < 0:
                            break
                        text_box = [s_bbox[0], s_bbox[1]-cnt*text_size_y, s_bbox[0]+text_size[0],s_bbox[1]-(cnt-1)*text_size_y]

                        #draw vcoco verb text
                        vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                        vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                        print(f'drawing vcoco action box : {text}, {cnt}')
                        cnt += 1 

    
    else:


        list_actions = []
        for box, hoi in zip(output_i['box_predictions'], output_i['hoi_predictions']):

            #prediction threshold
            if hoi['max_score'] < threshold:
                continue
            
            subject_id = hoi['subject_id']
            subject_box = box['h_bbox']
            object_id = hoi['object_id']
            object_box = box['o_bbox']
            verb_score = hoi['pair_score']
            max_verb_score = hoi['max_score']
            single_out = {'subject_box':np.array(subject_box), 'object_box':np.array(object_box), 'subject_id':np.array(subject_id), 'object_id':np.array(object_id), 'verb_score':np.array(verb_score), 'max_verb_score':np.array(max_verb_score)}
            list_actions.append(single_out)
        
        if list_actions:
            df = pd.DataFrame(list_actions)
            df = df.sort_values(by=['max_verb_score'],ascending=False).iloc[:top_k]
            list_actions = df.to_dict('records')
            for i, action in enumerate(list_actions):
                verbs = np.where(action['verb_score'] > threshold)
                s_bbox = action['subject_box']
                o_bbox = action['object_box']
                o_class = action['object_id']
                #import pdb; pdb.set_trace()
                vis_img = cv2.rectangle(vis_img, (int(s_bbox[0]),int(s_bbox[1])), (int(s_bbox[2]),int(s_bbox[3])), color_dict[0], 3)
                print('drawing subject box')
                
                vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                print('drawing object box')
                #print(int(o_class))
                text_size, BaseLine=cv2.getTextSize(index_2_cat(verbs[0][0],args.inf_type),cv2.FONT_HERSHEY_SIMPLEX,1,2)
                
                #text height for multiple verbs (interactions)
                text_size_y = text_size[1] 
                cnt = 1
                for verb in verbs[0]:
                    
                    text = index_2_cat(verb,args.inf_type)
                    text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                    if s_bbox[1]-cnt*text_size_y < 0 or s_bbox[0] < 0:
                        break
                    text_box = [s_bbox[0], s_bbox[1]-cnt*text_size_y, s_bbox[0]+text_size[0],s_bbox[1]-(cnt-1)*text_size_y]

                    #draw text
                    vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                    vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                    print(f'drawing action box : {text}, {cnt}')
                    cnt += 1 
    return vis_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser('video inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    video_path = args.video_file
    assert os.path.isfile(video_path), "check video file is in video path"
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))	
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    orig_size = torch.as_tensor([frame_height,frame_width]).unsqueeze(0).to('cuda')
    fps = 30.0
    args.output_dir = 'output_video/'+video_path.split('/')[-1]
    output_file = cv2.VideoWriter(args.output_dir, fourcc, fps, frame_size)
    model, _, postprocessors = build_model(args) #model, criterion, postprocessors
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'],strict=False)
    frame_num = 0
    color_dict = make_color_dict(args.num_obj_classes)
    while(True):
        retval, frame = cap.read() 
        print(f'frame_num : {frame_num}')
        if not retval:
            break
        
        if 'hico' in args.inf_type and 'vcoco' in args.inf_type:
            outputs_hico, outputs_vcoco = inference_for_vid(model, frame, args)
            preds = []
            results_hico = postprocessors(outputs_hico, orig_size)
            results_vcoco = postprocessors(outputs_vcoco, orig_size)
            
            preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_hico))))
            preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_vcoco))))
            output_hico = change_format(preds[0], args)
            output_vcoco = change_format(preds[1], args)
        
            output_i = []
            output_i.append(output_hico)
            output_i.append(output_vcoco)
            #import pdb; pdb.set_trace()
            vis_img = draw_img(frame,output_i,top_k=args.top_k,threshold=args.threshold,color_dict=color_dict,inf_type=args.inf_type)
            output_file.write(vis_img)
            frame_num += 1

        else:
            outputs = inference_for_vid(model, frame, args)
            preds = []
            results = postprocessors(outputs, orig_size)
            preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
            output_i = change_format(preds[0], args)
            vis_img = draw_img(frame,output_i,top_k=args.top_k,threshold=args.threshold,color_dict=color_dict,inf_type=args.inf_type)
            output_file.write(vis_img)
            frame_num += 1

    
    cap.release()
    output_file.release()
    cv2.destroyAllWindows()
