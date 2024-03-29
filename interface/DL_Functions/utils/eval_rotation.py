#coding:utf-8
import os,glob,pickle
import os.path as osp
from typing import cast
import numpy as np
# from nms import nms
import matplotlib.pyplot as plt 
GTCLASS=['ship'] #only one class


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def txt2rect(txt_path):
    '''
    rect annot format: x1,y1,x2,y2,cls
    '''
    rect_list = list()
    txt_file=open(txt_path,encoding='utf-8')

    for line in txt_file:
        #import pdb;pdb.set_trace()
        line=line.strip()
        row_list=line.split()
        rect=[int(float(x)) for x in row_list[:4]]
        annot_name=row_list[-1]
        rect_list.append((annot_name,rect))

    txt_file.close()
    return rect_list

def DotaTxt2polygons(txt_path):
    '''
    dota annot format:x1,y1,x2,y2,x3,y3,x4,y4,cls,diffculty
    '''
    polygon_list = list()
    txt_file=open(txt_path,encoding='utf-8')

    for line in txt_file:
        #import pdb;pdb.set_trace()
        line=line.strip()
        row_list=line.split()
        polygon=[int(float(x)) for x in row_list[:8]]
        annot_name=row_list[-2]
        polygon_list.append((annot_name,polygon))
    txt_file.close()
    return polygon_list

def polygons2rect(polygon_list):
    '''
    turn rotation labels to rectangle
    get the (minx miny),(maxx,maxy) to be lefttop and rightbottom
    '''
    rect_list=[]
    for polygon in polygon_list:
        annot_name=polygon[0]
        rect=polygon[1:]
        rect_array=np.array(rect).reshape((-1,2))
       # import pdb;pdb.set_trace()
        xmin=rect_array[:,0].min()
        xmax=rect_array[:,0].max()
        ymin=rect_array[:,1].min()
        ymax=rect_array[:,1].max()

        rect_list.append((annot_name,(xmin,ymin,xmax,ymax)))

    return rect_list

def LoadTxtGt_polygon(annot_dir,txttype='polygon'):
    '''
    annot_dir: txt labels dir
    return: 
    gt_dict:{basename:[[annot_name,[xmin,ymin,xmax,ymax]],
                       [annot_name,[xmin,ymin,xmax,ymax]],...],
             basename:[...],
             ...
            }
    '''
    annot_list = glob.glob(osp.join(annot_dir,'*.txt'))
    gt_dict={}
    for annot_path in annot_list:
        basename=osp.splitext(osp.basename(annot_path))[0]
        # import pdb;pdb.set_trace()
        polygon_list=DotaTxt2polygons(annot_path)
        gt_dict[basename]=polygon_list
    #assert len(gt_dict.keys())!=0,'ground truth object is 0,please check your annotation directory!'
    return gt_dict
def LoadTxtGt_rect(annot_dir):
    '''
    annot is rect : x1,y1,x2,y2,cls
    '''

def LoadDetfile(det_path):
    '''
    load .pkl format detection results
    detfile:{basename:[[x1,y1,x2,y2,score,cls],[x1,y1,x2,y2,score,cls]...],
             bashename:[...] 
            }
    return: detfile

    '''
    #import pdb;pdb.set_trace()
    detfile=open(det_path,'rb')
    det_dict = pickle.load(detfile)
    

    return det_dict

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def casia_eval(annot_dir,annot_type,det_path,imglist,classname,conf_thre=0.3,ovthresh=0.5,nms_thre=0.5,use_07_metric=False):
    '''
    this is single class ap caculate code,for 
    annot_dir:txt format annot_dir,txt format is:
    annot_type: 
        for polygon,annot txt format is:
            line1:[x1,y1,x2,y2,x3,y3,x4,y4]
            line2：,...,[....]]
        for rect,annot txt format is:
            line1:[x1,y1,x2,y2,cls]
            ...
    annot_path: polygon or rect
    det_path: detections.pkl
              detections.pkl saves a dict,its format is:
              {imagename:[[x1,y1,x2,y2,confidece,cls],[...],[...]],
               imagename:[....]
               ....}  
    imagesetfile: images nedd to evalutate
    classname: classname in txt file

    '''
    # import pdb;pdb.set_trace()
    gt_dict=LoadTxtGt_polygon(annot_dir,annot_type)
    det_dict=LoadDetfile(det_path)
    # import pdb;pdb.set_trace()
    imagenames = [osp.splitext(osp.basename(x))[0] for x in imglist]  
    cls_det=[] 
    cls_gt={} 
    #select groud truth of this cls 
    GtNmus=0
    #import pdb;pdb.set_trace()
    for imagename in imagenames:
        if not imagename in gt_dict.keys():
            cls_gt[imagename]=[]
            continue
        gts=gt_dict[imagename]
        cls_bboxes=[]
        for gt in gts:
            #import pdb;pdb.set_trace()
            if gt[0]==classname:
                detflag=0
                GtNmus+=1
                gtbbox=gt[1]
                cls_bboxes.append([gtbbox,detflag])
        #make sure image has the cls object
        if len(cls_bboxes)>0:
            cls_gt[imagename]=cls_bboxes
    
    #select detections of this cls
    for imagename in imagenames:
        if not imagename in det_dict.keys():
            continue
        dets=det_dict[imagename]
        # dets=nms(dets,nms_thre,conf_thre)
        for det in dets:
            #import pdb;pdb.set_trace()
            if det[-1]==GTCLASS.index(classname):
                cls_det.append(([imagename]+det[:]))
    if len(cls_det)>1:
        #get detction confidence and bbox
        '''
        cls_det:[[imagename,x1,y1,x2,y2,cof,cls],...]
        '''

        imageids=np.array([x[0] for x in cls_det])
        confidence=np.array([float(x[-2]) for x in cls_det])
        BBox=np.array([[float(z) for z in x[1:-2]] for x in cls_det])
       
        # rm confidence below confidence threshhold
        select_mask=confidence>conf_thre
        confidence=confidence[select_mask]
        BBox=BBox[select_mask]
        imageids=imageids[select_mask]

        #sort by confidence
        sorted_ind=np.argsort(-confidence)
        sorted_scores=np.sort(confidence)
        BBox=BBox[sorted_ind,:]
        imageids=imageids[sorted_ind]   

        #mark TPs and FPs
        nd = len(BBox)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        print('cofidence thre: {}\ntotal predicted bbox : {}'.format(conf_thre,len(BBox)))
        for d in range(nd):
            #import pdb;pdb.set_trace()
            '''
            cls_gtbboxs:[[(x1,y1,x2,y2),detflag],...]
            '''
            imgname=imageids[d]
            #import pdb;pdb.set_trace()
            cls_gtbboxs=cls_gt[imgname]
            #image has no object, so the bbox is false positive
            if len(cls_gtbboxs)==0:
                fp[d]=1
                continue
            import pdb;pdb.set_trace()

            BBGT=np.array([bbox[0] for bbox in cls_gtbboxs])
            bb=BBox[d,:].astype(float)
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])

            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not cls_gt[imgname][jmax][-1]:
                    tp[d] = 1.
                    cls_gt[imgname][jmax][-1] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.
        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        rec = tp / float(GtNmus)
        ap=voc_ap(rec, prec, use_07_metric)
        #FP=np.sum(fp)
        #print('false positive object nums {}'.format(FP))

  
        #import pdb;pdb.set_trace()
        print('*'*15)
        #print('save path : {} ')
        #print('weights path :{}\n'.format(args.trained_model))
        print('iou overthre:{}\nConfidence thre:{}\nAP:{}\nMaxRecall:{} \nMinPrecision: {}'\
            .format(ovthresh,conf_thre,ap,rec[-1],prec[-1]))
        save_txt=osp.splitext(det_path)[0]+'AP.txt'
        # with open(save_txt,'w') as f:
        #     f.write(('iou overthre:{} \
        #               Confidence thre:{} \
        #               AP:{} \
        #               MaxRecall:{}\
        #               MinPrecision: {}'.format(ovthresh,conf_thre,ap,rec[-1],prec[-1])))

        return rec,prec,ap

def test_samples():
    annot_dir='/data/03_Datasets/CasiaDatasets/ship/labels_dota'
    det_path='/data/02_code_implement/ssd.pytorch/MixShip/MixShip_iter40000/detections.pkl'
    imagesetfile='/data/02_code_implement/ssd.pytorch/MixShip/MixShip_iter40000/infer.imgnames'
    annot_type='polygon'
    overthre=0.5
    conf_thre=0.1
    clss=GTCLASS[0] #label is '0'
    #Recall,Precision=calculat_Precision(annot_dir,det_path,imagesetfile,overthre,conf_thre)
    rec,prec,ap=casia_eval(annot_dir,annot_type,det_path,imagesetfile,clss,overthre,conf_thre)
    print('*'*15+\
         '\niou overthre:{}\nConfidence thre:{}\nAP:{}\nMaxRecall:{} \nMinPrecision: {}'\
        .format(overthre,conf_thre,ap,rec[-1],prec[-1]))

if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description='evaluation object detecions performance,caculate aps')

    parser.add_argument('--annot_dir', default='/data/03_Datasets/CasiaDatasets/ship/Cutyoloship/ori_labels/val',
                        help='Location of txt format annotation directory')
    parser.add_argument('--image_dir', default='/data/03_Datasets/CasiaDatasets/ship/Cutyoloship/ori_images/val',
                        help='test imagenames')
    parser.add_argument('--annot_type', default='rect',choices=['rect','polygon']
                        ,help='for cutimages is rect,for dota annot is polygon')
    parser.add_argument('--det_path', default='MixShip/MixShip_iter40000/detections.pkl',
                        help='detection results path')
    parser.add_argument('--clss', default='ship', type=str,
                        help='annote class in txt file')
    parser.add_argument('--iou_thre', default=0.3, type=float,
                        help='evalution iou thre ')
    parser.add_argument('--conf_thre', default=0.3, type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--nms_thre', default=0.5, type=float,
                         help='evalution iou thre ')
    

    
    args = parser.parse_args()
    imglist=glob.glob(osp.join(args.image_dir,'*.jpg'))
    rec,prec,ap=casia_eval(args.annot_dir,args.annot_type,args.det_path,  
                    imglist,args.clss,args.iou_thre,args.conf_thre)
    # print('*'*15+\
    #      '\niou overthre:{}\nConfidence thre:{}\nAP:{}\nMaxRecall:{} \nMinPrecision: {}'\
    #     .format(args.iou_thre,args.conf_thre,ap,rec[-1],prec[-1]))
   


