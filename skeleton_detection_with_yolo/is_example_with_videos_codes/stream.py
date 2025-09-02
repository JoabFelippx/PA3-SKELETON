from is_msgs.image_pb2 import Image
from is_wire.core import Logger, Subscription, Message, Tracer

from streamChannel import StreamChannel

from skeletons import SkeletonsDetector 
from utils import get_topic_id, to_np, to_image, draw_skeleton, msg_commtrace, calib_data, calib_img_from_file, draw_identifier, draw_bounding_box, draw_keypoints

import cv2

import socket
import json
import time

import sys  
# Carregar modelo

bbox_thickness = 4    
bbox_color = (150, 0, 0)  

bbox_labelstr = {
    'font_size':1,      
    'font_thickness':2,
    'offset_x':0,       
    'offset_y':-10,     
}

kpt_labelstr = {
    'font_size':4,             
    'font_thickness':2,       
    'offset_x':0,             
    'offset_y':150,            
}

def main():

    # Carregar configurações
    config = json.load(open('etc/conf/options.json'))

    broker_uri = config['broker_uri']
    zipkin_host = config['zipkin_uri']

    camera_id = sys.argv[1]

    cap = cv2.VideoCapture(f'../videos/camera_{camera_id}.avi')
    
    sd = SkeletonsDetector(config)

    service_name = 'Skeleton.Detector'
    log = Logger(name=service_name)
    # channel = StreamChannel()
    log.info(f'Connected to broker {broker_uri}')
    
    # Calibração da câmera
    calib_path = 'calib'
    camCalibs = [calib_data(f'{calib_path}/calib_rt1.npz'), calib_data(f'{calib_path}/calib_rt2.npz'), calib_data(f'{calib_path}/calib_rt3.npz'), calib_data(f'{calib_path}/calib_rt4.npz')]

    img_list = list()

    while True:

        ret, im_np = cap.read()
        if not ret:
            print("End of video or error occurred.")
            break

        im_np = calib_img_from_file(camCalibs[int(camera_id)-1], im_np)
        img_list.append(im_np)

        if len(img_list) >= 1: 

            results = sd.detect(im_np)
            skeleton_msg = Message()
            skeleton_msg.topic = f'SkeletonDetector.{camera_id}.Detection'
            
            # obs_annotations = sd.to_object_annotations(results, im_np.shape)
            # skeleton_msg.pack(obs_annotations)
            # skeleton_msg.created_at = time.time()
            # channel.publish(skeleton_msg)

            num_bbox = len(results.boxes.cls)
            bboxes_xyxy = results.boxes.xyxy.cpu().numpy().astype('uint32') 
            bboxes_keypoints = results.keypoints.data.cpu().numpy().astype('uint32')

            for idx in range(num_bbox): 
                bbox_xyxy = bboxes_xyxy[idx] 
                bbox_label = results.names[0]
                bbox_keypoints = bboxes_keypoints[idx]
                
                try: 
                    identifier = int(results.boxes.id[idx])
                except:
                    continue
                    
                im_np = draw_bounding_box(im_np, bbox_xyxy, bbox_color, bbox_thickness, bbox_label, bbox_labelstr)
                im_np = draw_skeleton(im_np, bbox_keypoints, sd.skeleton_map, camera_id)
                im_np = draw_keypoints(im_np, bbox_keypoints, sd.kpt_color_map)
                im_np = draw_identifier(im_np, bbox_xyxy, identifier, bbox_color, bbox_thickness, bbox_labelstr) 


            cv2.imshow(f'Camera {camera_id}', im_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Encerrando a visualização.")
                break

        
        del img_list[0]
    
    cap.release()
    cv2.destroyAllWindows()  
if __name__ == "__main__":
    main()
