## modified to only track when the court is in view with high confidence

import argparse
import os
import os.path as osp
import time
import cv2
import torch
import csv
import numpy as np
import math
import sys
import glob

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

#########################################################################
def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    #parser.add_argument("-cc", "--court-confidence", type=str, default=None)
    parser.add_argument("-map", "--court-mapping", type=str, default=None)
    parser.add_argument("-mask", "--court-mask", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

#########################################################################
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

#########################################################################
def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

#########################################################################
class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

#########################################################################
def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

#########################################################################
# get the court points from file
# first 14 are in img coords, second 14 are in orthogonal coords
def getpoints(file):
    points = []
    intpoints = []
    orthopoints = []
    intorthopoints = []
    with open(file, mode = 'r') as file:
        csvFile = csv.reader(file)
        i = 0
        for line in csvFile:
            xval = float(line[0])
            yval = float(line[1])
            xvalint = int(xval)
            yvalint = int(yval)
            if i < 14:
                #print(i,line)
                points.append([xval,yval])
                intpoints.append([xvalint, yvalint])
            if i >= 14:
                orthopoints.append([xval,yval])
                intorthopoints.append([xvalint, yvalint])
            i += 1
        #print(points)
        #print(orthopoints)
    return np.float32(points), \
           intpoints, \
           np.float32(orthopoints)

########################################################################
orthscale = 20*1.5
orthshift = 25*1.5
def draw_birdseye(img, points):
    # zero out the court area.
    xmax = int((1.5+6.1)*orthscale+orthshift)
    ymax = int((1.5+13.41)*orthscale+orthshift)
    sqr = np.array([[0,0],[0,ymax],[xmax,ymax],[xmax,0]], dtype=np.int32)
    cv2.fillPoly(img, [sqr], (0,0,0))
    points = points * orthscale
    points = points + orthshift
    points = points.astype(int)
    imgout = cv2.line(img, points[0], points[1], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[1], points[2], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[2], points[3], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[3], points[0], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[4], points[7], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[5], points[6], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[8], points[10], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[9], points[11], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[12], points[13], [0, 0, 255], 3)
    return imgout
########################################################################
def distance_function(pt, xtl, ytl, xbr, ybr):
    # inside check - if inside, distance is 0
    x = pt[0]
    y = pt[1]
    if x > xtl and x < xbr and y > ytl and y < ybr:
        return 0
    #dr = min(abs(x - xtl), abs(x - xbr), abs(y - ytl), abs(y - ybr))
    # if x is inside x range
    if x > xtl and x < xbr:
        dr = min(abs(y - ytl), abs(y - ybr))
        return dr
    # if y is inside y range
    if y > ytl and y < ybr:
        dr = min(abs(x - xtl), abs(x - xbr))
        return dr
    # else do euclidean to nearest corner
    tlDist = math.sqrt((pt[0]-xtl)**2 + (pt[1]-ytl)**2)
    trDist = math.sqrt((pt[0]-xbr)**2 + (pt[1]-ytl)**2)
    blDist = math.sqrt((pt[0]-xtl)**2 + (pt[1]-ybr)**2)
    brDist = math.sqrt((pt[0]-xbr)**2 + (pt[1]-ybr)**2)
    return min(tlDist, trDist, blDist, brDist)

def calc_sorted_distance_from_point(pts):
    x = 3.05 #center point of court
    y = 6.71
    xtl = 0
    ytl = 0
    xbr = 6.1
    ybr = 13.41
    dists = []
    for pt in pts:        
        #dist = math.sqrt((pt[0]-x)**2 + (pt[1]-y)**2)
        dist = distance_function(pt, xtl, ytl, xbr, ybr)
        dists.append(dist)
    s = np.array(dists)
    sort_index = np.argsort(s)
    return sort_index

def getClosestFour(pts):
    # if we get here there are > 4 points and we need to find closest 4
    # to the X,Y target, which is the center of the court (3.05, 6.71)
    sorted_indexes = calc_sorted_distance_from_point(pts)
    out = []
    for i in range(0,4):
        out.append(pts[sorted_indexes[i]])
    return np.array(out)

    # diffs = []
    # for pt in pts:
    #     diffs.append(abs(pt[0] - xtarg))
    # s = np.array(diffs)
    # sort_index = np.argsort(s)
    # out = []
    # for i in range(0,4):
    #     out.append(pts[sort_index[i]])
    # return np.array(out)
        
########################################################################
def drawPeepPoints(img,tlwhs, mat, points):
    pts = []
    outpt = np.array([])
    if len(tlwhs) > 0:
        for tlwh in tlwhs:
            #print("tlwh",tlwh)
            tlx = tlwh[0]
            tly= tlwh[1]
            width=tlwh[2]
            height=tlwh[3]
            pointOfInterest = [tlx+(width/2), tly+height] # bottom middle of rect
            #print("pointOfInterest",pointOfInterest)
            pts.append(pointOfInterest)
        #print("pts",pts)
        src = np.zeros((len(pts), 1, 2))
        src[:, 0] = pts
        #print("src",src)
        outpt = cv2.perspectiveTransform(src, mat)
        #print("out1",outpt)
        outpt = outpt[:,0,:]
        #print("out2",outpt)
        # filter closest 4 points to center line (before scale and shift)
        if len(outpt) > 4:
            #print("filtering ortho points")
            absD = []
            centerlineX = points[9][0] # point 9 is intersection lower baseline and centre serviceline
            newpt = getClosestFour(outpt)
            #print(outpt)
            #print(newpt)
            outpt = newpt
        #print("out3",outpt)
        outptsc = outpt * orthscale
        outptsc = outptsc + orthshift
        outptsc = np.int32(outptsc)
        #print(outpt)

        for pt in outptsc:
            #print("pt",pt)
            cv2.circle(img, (pt[0],pt[1]), 10, (0,255,0), cv2.FILLED)
    if len(outpt) == 0:
        outpt = np.array([[0,0],[0,0],[0,0],[0,0]])
    elif len(outpt) < 4:
        #print(outpt)
        for i in range(0,4-len(outpt)):
            outpt = np.append(outpt, np.array([[0,0]]),axis=0)
        #print(outpt)
    return img, outpt # return the not-scaled peep points.
########################################################################
def imageflow_demo(predictor, vis_folder, current_time, args):
    datfile = args.court_mapping
    imgpoints, intimgpoints, orthpoints = getpoints(datfile)
    xform = cv2.getPerspectiveTransform(imgpoints[:4], orthpoints[:4])
    # read court confidence file
    #conffile = args.court_confidence
    #csv_file = open(conffile)
    #reader = csv.reader(csv_file)
    #line = next(reader) #skip header
    #print(line)
    maskfile = args.court_mask
    msk = cv2.imread(maskfile) # read in as 3 channel image
    #cv2.imshow('mask',msk)
    #cv2.waitKey(0)
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    vis_folder = "."
    save_folder = "." # osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
        save_path = "byteTrackPbcourt.mp4"
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    print("fps ",fps)
    fpsint = round(fps,0)
    #print("fpsint ",fpsint)
    #tracker = BYTETracker(args, frame_rate=30)
    tracker = BYTETracker(args, frame_rate=fpsint)
    timer = Timer()
    frame_id = 0
    results = []
    peepPoints = []

    while True:
        if frame_id % 500 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            # get next confidence value
            #line = next(reader)
            #conf = float(line[1])
            #print(line, conf)
            # mask frame
            frame1 = cv2.bitwise_and(frame, msk) # using mask didn't work, but this did.
            #cv2.imshow('frame',frame)
            #cv2.waitKey(0)
            outputs, img_info = predictor.inference(frame1, timer)
            frame = draw_birdseye(frame, orthpoints)
            #cv2.imshow('frm',frame)
            #cv2.waitKey(0)
            thisPeep = np.array([[0,0],[0,0],[0,0],[0,0]])
            # try skipping if low confidence court detection
			# dont do that for videos that do subtle zooms
            if outputs[0] is not None:# and conf > 0.85:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                timer.toc()
                    #img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                online_im = plot_tracking(
                    frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
                # draw point onto ortho court
                #print("frame",frame_id)
                online_im, thisPeep = drawPeepPoints(online_im,online_tlwhs,xform, orthpoints)
            else:
                timer.toc()
                online_im = frame #img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)

            #print(thisPeep)
            peepPoints.append(
                f"{frame_id},{thisPeep[0][0]:.2f},{thisPeep[0][1]:.2f},{thisPeep[1][0]:.2f},{thisPeep[1][1]:.2f},{thisPeep[2][0]:.2f},{thisPeep[2][1]:.2f},{thisPeep[3][0]:.2f},{thisPeep[3][1]:.2f}\n"
            )            
            #print(peepPoints)
            #cv2.imshow("asd",online_im)
            #ch = cv2.waitKey(0)
            #if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #    break
        else:
            break
        frame_id += 1
    #print(results[1:5])
    #print(peepPoints[1:5])
    #print(peepPoints[1560:1562])
    if args.save_result:
        #res_file = osp.join(vis_folder, f"{timestamp}.txt")
        res_file = osp.join(vis_folder, f"rawDetectionsIds.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        #res_file2 = osp.join(vis_folder, f"{timestamp}_peeps.txt")
        res_file2 = osp.join(vis_folder, f"detectionsIds_peeps.txt")
        with open(res_file2, 'w') as f:
            f.writelines(peepPoints)
        logger.info(f"save results to {res_file}")
    #csv_file.close() # close the court confidence csv file


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        #os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
