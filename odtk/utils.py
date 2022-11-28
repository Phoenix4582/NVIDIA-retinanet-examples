import os.path
import time
import json
import warnings
import signal
from datetime import datetime
from contextlib import contextmanager
from PIL import Image, ImageDraw
import requests
import numpy as np
import math
import torch
import cv2
from nms import nms
import csv


def order_points(pts):
    pts_reorder = []

    for idx, pt in enumerate(pts):
        idx = torch.argsort(pt[:, 0])
        xSorted = pt[idx, :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[torch.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        D = torch.cdist(tl[np.newaxis], rightMost)[0]
        (br, tr) = rightMost[torch.argsort(D, descending=True), :]
        pts_reorder.append(torch.stack([tl, tr, br, bl]))

    return torch.stack([p for p in pts_reorder])

def rotate_boxes(boxes, points=False):
    '''
    Rotate target bounding boxes

    Input:
        Target boxes (xmin_ymin, width_height, theta)
    Output:
        boxes_axis (xmin_ymin, xmax_ymax, theta)
        boxes_rotated (xy0, xy1, xy2, xy3)
    '''

    u = torch.stack([torch.cos(boxes[:,4]), torch.sin(boxes[:,4])], dim=1)
    l = torch.stack([-torch.sin(boxes[:,4]), torch.cos(boxes[:,4])], dim=1)
    R = torch.stack([u, l], dim=1)

    if points:
        cents = torch.stack([(boxes[:,0]+boxes[:,2])/2, (boxes[:,1]+boxes[:,3])/2],1).transpose(1,0)
        boxes_rotated = torch.stack([boxes[:,0],boxes[:,1],
            boxes[:,2], boxes[:,1],
            boxes[:,2], boxes[:,3],
            boxes[:,0], boxes[:,3],
            boxes[:,-2],
            boxes[:,-1]],1)

    else:
        cents = torch.stack([boxes[:,0]+(boxes[:,2])/2, boxes[:,1]+(boxes[:,3])/2],1).transpose(1,0)
        boxes_rotated = torch.stack([boxes[:,0],boxes[:,1],
            (boxes[:,0]+boxes[:,2]), boxes[:,1],
            (boxes[:,0]+boxes[:,2]), (boxes[:,1]+boxes[:,3]),
            boxes[:,0], (boxes[:,1]+boxes[:,3]),
            boxes[:,-2],
            boxes[:,-1]],1)

    xy0R = torch.matmul(R,boxes_rotated[:,:2].transpose(1,0) - cents) + cents
    xy1R = torch.matmul(R,boxes_rotated[:,2:4].transpose(1,0) - cents) + cents
    xy2R = torch.matmul(R,boxes_rotated[:,4:6].transpose(1,0) - cents) + cents
    xy3R = torch.matmul(R,boxes_rotated[:,6:8].transpose(1,0) - cents) + cents

    xy0R = torch.stack([xy0R[i,:,i] for i in range(xy0R.size(0))])
    xy1R = torch.stack([xy1R[i,:,i] for i in range(xy1R.size(0))])
    xy2R = torch.stack([xy2R[i,:,i] for i in range(xy2R.size(0))])
    xy3R = torch.stack([xy3R[i,:,i] for i in range(xy3R.size(0))])

    boxes_axis = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:4] - 1,
        torch.sin(boxes[:,-1, None]), torch.cos(boxes[:,-1, None])], 1)
    boxes_rotated = order_points(torch.stack([xy0R,xy1R,xy2R,xy3R],dim = 1)).view(-1,8)

    return boxes_axis, boxes_rotated


def rotate_box(bbox):
    xmin, ymin, width, height, theta = bbox

    xy1 = xmin, ymin
    xy2 = xmin, ymin + height - 1
    xy3 = xmin + width - 1, ymin + height - 1
    xy4 = xmin + width - 1, ymin

    cents = np.array([xmin + (width - 1) / 2, ymin + (height - 1) / 2])

    corners = np.stack([xy1, xy2, xy3, xy4])

    u = np.stack([np.cos(theta), -np.sin(theta)])
    l = np.stack([np.sin(theta), np.cos(theta)])
    R = np.vstack([u, l])

    corners = np.matmul(R, (corners - cents).transpose(1, 0)).transpose(1, 0) + cents

    return corners.reshape(-1).tolist()


def show_detections(detections):
    'Show image with drawn detections'

    for image, detections in detections.items():
        im = Image.open(image).convert('RGBA')
        overlay = Image.new('RGBA', im.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        detections.sort(key=lambda d: d['score'])
        for detection in detections:
            box = detection['bbox']
            alpha = int(detection['score'] * 255)
            draw.rectangle(box, outline=(255, 255, 255, alpha))
            draw.text((box[0] + 2, box[1]), '[{}]'.format(detection['class']),
                      fill=(255, 255, 255, alpha))
            draw.text((box[0] + 2, box[1] + 10), '{:.2}'.format(detection['score']),
                      fill=(255, 255, 255, alpha))
        im = Image.alpha_composite(im, overlay)
        im.show()


def save_detections(path, detections):
    print('Writing detections to {}...'.format(os.path.basename(path)))
    with open(path, 'w') as f:
        json.dump(detections, f)


@contextmanager
def ignore_sigint():
    handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, handler)


class Profiler(object):
    def __init__(self, names=['main']):
        self.names = names
        self.lasts = {k: 0 for k in names}
        self.totals = self.lasts.copy()
        self.counts = self.lasts.copy()
        self.means = self.lasts.copy()
        self.reset()

    def reset(self):
        last = time.time()
        for name in self.names:
            self.lasts[name] = last
            self.totals[name] = 0
            self.counts[name] = 0
            self.means[name] = 0

    def start(self, name='main'):
        self.lasts[name] = time.time()

    def stop(self, name='main'):
        self.totals[name] += time.time() - self.lasts[name]
        self.counts[name] += 1
        self.means[name] = self.totals[name] / self.counts[name]

    def bump(self, name='main'):
        self.stop(name)
        self.start(name)


def post_metrics(url, metrics):
    try:
        for k, v in metrics.items():
            requests.post(url,
                          data={'time': int(datetime.now().timestamp() * 1e9),
                                'metric': k, 'value': v})
    except Exception as e:
        warnings.warn('Warning: posting metrics failed: {}'.format(e))


def capture_frames(video_source, dest_dir, freq):
    """
    Main function with the goal of capturing image frames at certain interval.
    input:
    args: arguments from ArgumentParser
    """
    try:
        os.mkdir(dest_dir)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(video_source)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    total = 0
    print (f"Converting video at frequency {freq} frame(s) interval..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            count = count + 1
            continue
        # Write the results back to output location.
        # Capture frames at a frequency of args.freq
        if (count % freq == 0):
            cv2.imwrite(dest_dir + "/%#05d.jpg" % (total+1), frame)
            total = total + 1
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print (f"Done extracting frames to {dest_dir}.\n{total} frames extracted")
            print (f"It took {time_end-time_start} seconds for conversion.")
            break

def nms_mod(rbox_dataset, nms_threshold, debug=True):
    """
    Perform Non-Maximum Suppression(NMS) on inference. Threshold can be varied.
    Input:
    rbox_dataset: json-formatted inference dataset
    nms_threshold: a variable to control the performance of NMS.
    debug: boolean to flip angle direction

    Output: the filtered inference dataset
    """
    result = {}
    rboxes = [rbox['bbox'] for rbox in rbox_dataset]
    scores = [rbox['score'] for rbox in rbox_dataset]
    rbox_centered = []
    for rbox in rboxes:
        x, y, w, h, theta = rbox
        if debug:
            theta = -1 * theta
        cx, cy = x + 0.5 * w, y + 0.5 * h
        rbox_centered.append(((cx, cy), (w, h), theta))

    indices = nms.rboxes(rbox_centered, scores, nms_threshold=nms_threshold)
    filtered_dataset = [rbox_dataset[id] for id in indices]
    return filtered_dataset

def save_video_inference(dest_dir, detections_file, freq, nms_thr, iou_thr, show_box, show_track, save_details):
    frame_ids = sorted([img.strip(".jpg") for img in os.listdir(dest_dir) if img.endswith(".jpg")])
    frames = [cv2.imread(os.path.join(dest_dir, '{}.jpg'.format(id))) for id in frame_ids]
    frame_ids = sorted([int(id) for id in frame_ids])
    # print(*frame_ids)
    assert len(frame_ids) == len(frames)

    height, width, layers = frames[0].shape
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    augmented_frames = []
    dots_dict = {}
    for id, frame in zip(frame_ids, frames):
        info = get_inference(detections_file, id, nms_thr)
        dots_dict = update_dots(info, dots_dict)
        # print(dots_dict)
        new_frame = augment_frame(frame, info, dots_dict, show_box, show_track)
        augmented_frames.append(new_frame)


    current = get_time()
    if show_box and show_track:
        current = current + "_full"
    elif show_box:
        current = current + "_box"
    elif show_track:
        current = current + "_track"

    gen_video_dest = os.path.join(dest_dir, f"captured_video_{current}.mp4")
    if os.path.exists(gen_video_dest):
        os.remove(gen_video_dest)

    out = cv2.VideoWriter(gen_video_dest, fourcc, freq, size)
    for frame in augmented_frames:
        out.write(frame)

    out.release()
    print(f"Successfully generated video at {gen_video_dest}")

    # NEED UPDATE IN THE FUTURE
    if save_details:
        csv_dest = os.path.join(dest_dir, 'tracklet_{}.csv'.format(current))
        init_csv(csv_dest)
        video_name = dest_dir.split("/")[-1]
        write_to_csv(csv_dest, video_name, dots_dict)

def get_inference(detections_file, id, nms_thr):
    with open(detections_file, 'r') as json_source:
        detection_contents = json.load(json_source)
        proposals = [anno for anno in detection_contents['annotations'] if anno['image_id'] == id]
        nms_filtered_proposals = nms_mod(proposals, nms_thr)
    return nms_filtered_proposals

def get_time():
    current = datetime.now()
    current_YMD = "{}{}{}".format(str(current.year).zfill(2), str(current.month).zfill(2), str(current.day).zfill(2))
    current_HMS = "{}{}{}".format(str(current.hour).zfill(2), str(current.minute).zfill(2), str(current.second).zfill(2))
    return f"{current_YMD}_{current_HMS}"

def update_dots(info, dots_dict):
    bboxes = [anno['bbox'] for anno in info]
    center_coords = find_center_coords(bboxes)
    if len(dots_dict) == 0:
        for id, center in enumerate(center_coords):
            dots_dict.update({"{}".format(str(id+1)): [center]})
    else:
        # update_record = {k: False for k in dots_dict.keys()}
        # update_details = []
        # Need debugging for correct pairing of proposals
        dots_dict = pair_centers_with_dict(center_coords, dots_dict)

        # for center in center_coords:
        #     # nearest_id = find_nearest_id(dots_dict, circle_center)
        #     # if nearest_id in dots_dict.keys() and not update_record[nearest_id]:
        #     # if not update_record[nearest_id]:
        #     #     target_contents = dots_dict[nearest_id]
        #     #     update_details.append((nearest_id, target_contents, circle_center))
        #     #     update_record.update(nearest_id: True)
        #     # else:
        #     #     largest_key = max([int(key) for key in dots_dict.keys()])
        #     #     update_details.append((str(largest_key+1), center))
        #
        #     # Check if all previous candidates are updated.
        #     if False in update_record.values():
        #
        #         sorted_kd_pairs = sort_distances(dots_dict, center)
        #         for nearest_id in sorted_kd_pairs:
        #             if not update_record[nearest_id]:
        #                 target_contents = dots_dict[nearest_id]
        #                 update_details.append((nearest_id, target_contents, center))
        #                 update_record.update({nearest_id: True})
        #                 break
        #     else:
        #         largest_key = max([int(key) for key in dots_dict.keys()])
        #         update_details.append((str(largest_key+1), center))


        # # Update the entry via update_details
        # for detail in update_details:
        #     # New entry
        #     if len(detail) == 2:
        #         new_key, point = detail
        #         dots_dict.update({new_key: [point]})
        #
        #     # Update old entry
        #     elif len(detail) == 3:
        #         nearest_id, contents, new_center = detail
        #         contents.append(new_center)
        #         dots_dict.update({nearest_id: contents})

    return dots_dict

# For each new center entry, find nearest previous dot,
# Ordering of dots and centers are randomized, 2D np.array is required, focus on minimum distance in 2D array
def pair_centers_with_dict(centers, dots_dict):
    # update_record = {k: False for k in dots_dict.keys()}
    # update_details = []
    latest_dots = {k: v[-1] for (k, v) in dots_dict.items()}
    remainders = [center for center in centers]
    # step = 1
    while (len(latest_dots.keys()) != 0 and len(remainders) != 0):
        # print(f"Current Step {step}: ")
        dots_centers_distances_table = np.array([[calculate_distance(dot, center) for dot in latest_dots.values()] for center in remainders], dtype=np.float32)
        # print(dots_centers_distances_table)
        min_distance = np.amin(dots_centers_distances_table)
        occurences = np.where(dots_centers_distances_table == min_distance)
        first_occurence = list(zip(occurences[0], occurences[1]))[0]
        # print(f"First Occurence: {first_occurence}")
        min_center_id, min_prev_dot_id = first_occurence
        # min_center_id = np.argmin(dots_centers_distances_table[:, 0])
        # print(f"Min center ID :{min_center_id}")
        # min_prev_dot_id = np.argmin(dots_centers_distances_table[min_center_id, :])
        dict_keys = list(latest_dots.keys())
        # print(f"Min prev dot ID: {dict_keys[min_prev_dot_id]}")
        dots_dict[dict_keys[min_prev_dot_id]].append(remainders[min_center_id])
        del latest_dots[dict_keys[min_prev_dot_id]]
        del remainders[min_center_id]
        # print(dots_dict)
        # step = step + 1

    if len(remainders) != 0:
        for remainder in remainders:
            largest_key = max([int(key) for key in dots_dict.keys()])
            dots_dict.update({str(largest_key+1): [remainder]})

    return dots_dict
    # for dot_id, dot in latest_dots.items():
    #     if len(centers) != 0:
    #         distances = {center_id: calculate_distance(dot, center) for center_id, center in enumerate(centers)}
    #         sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
    #         min_distance_id = sorted_distances.keys()[0]
    #         prev_contents = dots_dict[dot_id]
    #         new_entry = centers[min_distance_id]
    #         dots_dict.update({dot_id: prev_contents.append(new_entry)})
    #         del centers[min_distance_id]


def augment_frame(frame, info, dots_dict, show_box, show_track):
    height, width, layers = frame.shape
    circle_radius = 10
    circle_color = (200, 213, 48) # Torquoise for circle
    circle_thinkness = -1 # Filled Circle

    line_color = (0, 128, 255) # Orange for line
    line_thickness = 5

    if show_box:
        frame = draw_bbox(frame, info, debug=True)

    if show_track:
        frame = fill_prev_markings(frame, dots_dict, circle_radius, circle_color, circle_thinkness, line_color, line_thickness)

    return frame

def fill_prev_markings(frame, dots_dict,
                       circle_radius, circle_color, circle_thinkness,
                       line_color, line_thickness):
    for _, dots in dots_dict.items():
        for id, dot in enumerate(dots):
            frame = cv2.circle(frame, dot, circle_radius, circle_color, circle_thinkness)
            if id < len(dots) - 1:
                frame = cv2.line(frame, dot, dots[id+1], line_color, line_thickness)

    return frame

# For results from OpenCV2(from GT), theta is positive as CLOCKWISE, testing over roLabelimg(GT original) is needed
# roLabelimg is limited within (0, pi), CLOCKWISE as positive
# May not need to look into roLabelimg direction, as involvement of head coordinates will correct the angle
# Only need conversion from clockwise to counterclockwise
# For results from inference(IR), theta is positive as COUNTERCLOCKWISE
def draw_bbox(img, dataset, debug):
    if len(dataset) == 0:
        return img

    color = (226, 181, 0)
    arrow_color = (255, 111, 255)
    for data in dataset:
        label = "IR"
        x, y, w, h, theta = data['bbox']
        if debug:
            theta = -1 * theta
        x_center = int(x + 0.5 * w)
        y_center = int(y + 0.5 * h)
        pointer = max(w, h)
        pt1 = (x_center, y_center)
        pt2 = (int(x_center + 0.5*(pointer+100)*math.cos(theta)), int(y_center + 0.5*(pointer+100)*math.sin(theta)))

        label = label + ":{}%".format(round(data['score'] * 100, 3))

        angle = math.degrees(theta)
        rot_rectangle = ((x_center, y_center), (w, h), angle)
        box = cv2.boxPoints(rot_rectangle)
        box = np.int0(box)

        infused_img = cv2.drawContours(img, [box], 0, color, 2)
        infused_img = cv2.arrowedLine(img, pt1, pt2, arrow_color, 2)
        # infused_img = cv2.polylines(img, [points], True, color, 2)
        infused_img = cv2.putText(infused_img, label, (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

    return infused_img

def find_center_coords(bboxes):
    return [(int(box[0] + 0.5 * box[2]), int(box[1] + 0.5 * box[3])) for box in bboxes]

def find_nearest_id(dots_dict, circle_center):
    """
    Find the id that new center entry is closest to, return a single key from dots_dict
    """
    # keys_min_distances_pairs = {k: calculate_distance(v[-1], circle_center) for (k, v) in dots_dict.items()}
    # sorted_k_min_d_pairs = dict(sorted(keys_min_distances_pairs.items(), key=lambda item: item[1]))
    min_distance = float('inf')
    min_distance_key = -1
    for k, v in dots_dict.items():
        prev_dot = v[-1]
        distance = calculate_distance(prev_dot, circle_center)
        if distance < min_distance:
            min_distance_key = k
            min_distance = distance

    # assert min_distance_key == sorted_k_min_d_pairs.keys[0]
    return min_distance_key

# def sort_distances(dots_dict, circle_center):
#     """
#     Sort keys via distances between circle center and previous dots, in ascending order, returns a list of keys
#     """
#     keys_min_distances_pairs = {k: calculate_distance(v[-1], circle_center) for (k, v) in dots_dict.items()}
#     sorted_k_min_d_pairs = dict(sorted(keys_min_distances_pairs.items(), key=lambda item: item[1]))
#
#     return sorted_k_min_d_pairs

def calculate_distance(dot1, dot2):
    x1, y1 = dot1
    x2, y2 = dot2
    return ((x1-x2)**2 + (y1-y2)**2) ** 0.5

def init_csv(file_name):
    header = ['video_name', 'tracklet_id', 'coordinates']
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

def write_to_csv(file_name, video_name, contents):
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for k, v in contents.items():
            for coordinates in v:
                x, y = coordinates
                output_coord = "({},{})".format(x, y)
                row = [video_name, k, output_coord]
                writer.writerow(row)
