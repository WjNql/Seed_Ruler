import argparse
import os
import shutil
import time
from pathlib import Path
import numpy
import xlwt
import cv2
import torch
import torch.backends.cudnn as cudnn
import xlrd
from xlutils.copy import copy
import os.path as osp
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadImages_files
from utils.general import (
    check_img_size, non_max_suppression_rotation, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging, increment_path)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.remote_utils import crop_xyxy2ori_xyxy, nms, draw_clsdet, draw_clsdet_rotation, rboxes2points, draw_one_box
from detectron2.layers import nms_rotated
from zjf_eage_1 import count_pixel


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz, overlap = \
        increment_path(
            Path(opt.save_dir) / 'exp'), opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.overlap
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages_files(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = numpy.zeros((len(names), 3))
    for indexx in range(len(names)):
        colors[indexx][indexx] = 255

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    save_dir = str(Path(out) / 'images')
    svae_txt_dir = str(Path(out) / 'labelTxt')
    save_excel_dir = str(Path(out) / 'excel_files')

    if os.path.exists(save_dir):  # output dir
        shutil.rmtree(save_dir)  # delete dir
    os.makedirs(save_dir)  # make new dir
    if os.path.exists(svae_txt_dir):  # output dir
        shutil.rmtree(svae_txt_dir)  # delete dir
    os.makedirs(svae_txt_dir)  # make new dir
    if os.path.exists(save_excel_dir):  # output dir
        shutil.rmtree(save_excel_dir)  # delete dir
    os.makedirs(save_excel_dir)  # make new dir

    xlspath = str(Path(out) / 'DL_rate_data.xlsx')
    xls_name = ['name_index', 'w_means-(mm)', 'h_means-(mm)', 'rice_numbers']
    # 创建excel文件, 如果已有就会覆盖
    workbook = xlwt.Workbook(encoding='utf-8')
    workbook.add_sheet('L0')
    workbook.save(xlspath)
    # 写入数据:
    wb = xlrd.open_workbook(xlspath)
    nrows = wb.sheet_by_index(0).nrows  # 当前行数
    copywb = copy(wb)
    targetsheet = copywb.get_sheet(0)
    ncols = 1  # 写第一行名字
    for celldata in xls_name:
        targetsheet.write(nrows, ncols, celldata)
        ncols += 1
    nrows += 1

    per_xls_name = ['index', 'w-(mm)', 'h-(mm)']

    for path, img, im0s, vid_cap in dataset:
        p, s, im0 = path, '', im0s
        s += '%gx%g ' % im0s.shape[:2]

        H, W, C = im0s.shape
        step_h, step_w = (imgsz - overlap), (imgsz - overlap)
        ori_preds = []
        save_path = str(Path(save_dir) / Path(p).name)
        txt_path = str(Path(svae_txt_dir) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')

        per_xlspath = str(Path(save_excel_dir) / Path(p).stem) + '.xls'
        # 创建excel文件, 如果已有就会覆盖
        per_workbook = xlwt.Workbook(encoding='utf-8')
        per_workbook.add_sheet('L0')
        per_workbook.save(per_xlspath)
        # 写入数据:
        per_wb = xlrd.open_workbook(per_xlspath)
        per_nrows = per_wb.sheet_by_index(0).nrows  # 当前行数
        per_copywb = copy(per_wb)
        per_targetsheet = per_copywb.get_sheet(0)
        per_ncols = 0  # 写第一行名字
        for per_celldata in per_xls_name:
            per_targetsheet.write(per_nrows, per_ncols, per_celldata)
            per_ncols += 1
        per_nrows += 1

        rice_numbers = 0
        w_list = []
        h_list = []

        for x_shift in range(0, W - imgsz, step_w):
            for y_shift in range(0, H - imgsz, step_h):
                img_part = img[:, y_shift:y_shift + imgsz, x_shift:x_shift + imgsz]
                img_part = torch.from_numpy(img_part).to(device)
                img_part = img_part.half() if half else img_part.float()  # uint8 to fp16/32
                img_part /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img_part.ndimension() == 3:
                    img_part = img_part.unsqueeze(0)
                # Inference
                pred = model(img_part, augment=opt.augment)[0]
                # Apply NMS
                pred = non_max_suppression_rotation(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                                    agnostic=opt.agnostic_nms)

                if len(pred[0]) > 0:
                    ori_pred = crop_xyxy2ori_xyxy(pred[0], x_shift, y_shift)  ####
                    ori_preds += ori_pred
        ori_preds = numpy.array(ori_preds)
        ori_preds = torch.from_numpy(ori_preds).to(device)
        if len(ori_preds) > 0:
            boxes, scores = ori_preds[:, :5].clone(), ori_preds[:, 5]
        else:
            continue

        objects_i = nms_rotated(boxes, scores, opt.iou_thres)
        ori_preds = ori_preds[objects_i]
        if ori_preds is not None and len(ori_preds):
            pred_data = [pd.cpu().numpy().tolist() for pd in ori_preds]  ###
            for c in ori_preds[:, -1].unique():
                n = (ori_preds[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string
                rice_numbers = n

            det_index = 0
            for *xywh, theta, conf, cls in reversed(pred_data):
                det_index = det_index + 1

                w_list.append(max(xywh[2], xywh[3]))
                h_list.append(min(xywh[2], xywh[3]))

                if save_txt:  # Write to file
                    line = (cls, *xywh, theta, conf) if opt.save_conf else (cls, *xywh, theta)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line) + '\n') % line)

                if save_img or view_img:  # Add bbox to image
                    label = '%s' % (str(det_index))
                    draw_one_box(xywh, theta, im0, label=label, color=colors[int(cls)], line_thickness=1)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f'{save_path} saved!')

        per_images_pixel = count_pixel(p)
        # 写数据
        ncols = 0
        targetsheet.write(nrows, ncols, nrows)
        ncols += 1
        targetsheet.write(nrows, ncols, str(osp.splitext(osp.split(Path(p))[-1])[0]))
        ncols += 1
        w_array = numpy.array(w_list)
        w_means = round(w_array.mean() * per_images_pixel, 4)
        targetsheet.write(nrows, ncols, w_means)
        ncols += 1
        h_array = numpy.array(h_list)
        h_means = round(h_array.mean() * per_images_pixel, 4)
        targetsheet.write(nrows, ncols, h_means)
        ncols += 1
        total_numbers = rice_numbers.item() if isinstance(rice_numbers, torch.Tensor) else rice_numbers
        targetsheet.write(nrows, ncols, total_numbers)
        ncols += 1
        nrows += 1

        for k, w_item in enumerate(w_list):
            per_ncols = 0
            per_targetsheet.write(per_nrows, per_ncols, per_nrows)
            per_ncols += 1
            w_result_data = round(w_item * per_images_pixel, 3)
            h_result_data = round(h_list[k] * per_images_pixel, 3)
            per_targetsheet.write(per_nrows, per_ncols, w_result_data)
            per_ncols += 1
            per_targetsheet.write(per_nrows, per_ncols, h_result_data)
            per_ncols += 1
            per_nrows += 1
        per_copywb.save(per_xlspath)  # 保存修改
    copywb.save(xlspath)  # 保存修改

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--overlap', type=int, default=200, help='sub image overlap size (pixels)')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
