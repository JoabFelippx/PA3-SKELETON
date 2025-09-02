import dateutil.parser as dp

import re

from is_msgs.image_pb2 import Image

import cv2
import numpy as np

def get_topic_id(topic: str) -> str: # type: ignore[return]
    re_topic = re.compile(r"CameraGateway.(\d+).Frame")
    result = re_topic.match(topic)
    if result:
        return result.group(1)
    
def span_duration_ms(span) -> float:
    dt = dp.parse(span.end_time) - dp.parse(span.start_time)
    return dt.total_seconds() * 1000.0


def to_np(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image

def to_image(image, encode_format: str = ".jpeg", compression_level: float = 0.8, ) -> Image:
    if encode_format == ".jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * (100 - 0) + 0)]
    elif encode_format == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * (9 - 0) + 0)]
    else:
        return Image()
    cimage = cv2.imencode(ext=encode_format, img=image, params=params)
    return Image(data=cimage[1].tobytes())


def calib_data(calib_file):
    return np.load(calib_file)

def calib_img_from_file(npzCalib, image):
    undistort_img = cv2.undistort(image, npzCalib['K'], npzCalib['dist'], None, npzCalib['nK'])
    _, _, w, h = npzCalib['roi']
    undistort_img = undistort_img[0:h, 0:w, :]
    return undistort_img

def msg_commtrace(msg: str, timestamp_rcvd: float):
    if msg.metadata != {}:

        msg_to_commtrace = (
            f'{{"timestamp_send": "{int(msg.created_at*1000000)}", '
            f'"timestamp_rcvd": "{int(timestamp_rcvd*1000000)}", '
            f'"x-b3-flags": "{msg.metadata["x-b3-flags"]}", '
            f'"x-b3-parentspanid": "{msg.metadata["x-b3-parentspanid"]}", '
            f'"x-b3-sampled": "{msg.metadata["x-b3-sampled"]}", '
            f'"x-b3-spanid": "{msg.metadata["x-b3-spanid"]}", '
            f'"x-b3-traceid": "{msg.metadata["x-b3-traceid"]}", '
            f'"spanname": "frame"}}'
        )

        bytesToSend = str.encode(msg_to_commtrace)

        return (bytesToSend, msg_to_commtrace)

def draw_bounding_box(image, bbox_xyxy, bbox_color, bbox_thickness, bbox_label, bbox_labelstr):

    image = cv2.rectangle(image, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color, bbox_thickness)

    #image = cv2.putText(image, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])

    return image

def draw_identifier(image, bbox_xyxy, bbox_id, bbox_color, bbox_thickness, bbox_labelstr):

    image = cv2.putText(image, str(bbox_id), (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])

    return image

def draw_skeleton(image, bbox_keypoints, skeleton_map, cam_id):
   
    for skeleton_map in skeleton_map:

        srt_kpt_id = skeleton_map['srt_kpt_id']

        if bbox_keypoints[srt_kpt_id][0] == 0 and bbox_keypoints[srt_kpt_id][1] == 0:
            return image

        srt_kpt_x = bbox_keypoints[srt_kpt_id][0]
        srt_kpt_y = bbox_keypoints[srt_kpt_id][1]


        dst_kpt_id = skeleton_map['dst_kpt_id']

        if bbox_keypoints[dst_kpt_id][0] == 0 and bbox_keypoints[dst_kpt_id][1] == 0:
            return image

        dst_kpt_x = bbox_keypoints[dst_kpt_id][0]
        dst_kpt_y = bbox_keypoints[dst_kpt_id][1]

        # print(cam_id)
        # print(f'srt_kpt_id: {srt_kpt_id}, srt_kpt_x: {srt_kpt_x}, srt_kpt_y: {srt_kpt_y}')
        # print(f'dst_kpt_id: {dst_kpt_id}, dst_kpt_x: {dst_kpt_x}, dst_kpt_y: {dst_kpt_y}')

        skeleton_color = skeleton_map['color']

        skeleton_thickness = skeleton_map['thickness']

        image  = cv2.line(image, (srt_kpt_x, srt_kpt_y),(dst_kpt_x, dst_kpt_y),color=skeleton_color,thickness=skeleton_thickness)

    return image

def draw_keypoints(image, bbox_keypoints, kpt_color_map):

    for kpt_id in kpt_color_map:

        kpt_color = kpt_color_map[kpt_id]['color']
        kpt_radius = kpt_color_map[kpt_id]['radius']

        if bbox_keypoints[kpt_id][0] == 0 and bbox_keypoints[kpt_id][1] == 0:
            continue

        kpt_x = bbox_keypoints[kpt_id][0]
        kpt_y = bbox_keypoints[kpt_id][1]

        #print(f'kpt_id: {kpt_id}, kpt_x: {kpt_x}, kpt_y: {kpt_y}')
        image = cv2.circle(image, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

    return image

