from is_msgs.image_pb2 import Image
from is_wire.core import Logger, Subscription, Message, Tracer

from streamChannel import StreamChannel

from skeletons import SkeletonsDetector 
from utils import get_topic_id, to_np, to_image, create_exporter, draw_skeleton, msg_commtrace, calib_data, calib_img_from_file, draw_identifier, draw_bounding_box, draw_keypoints

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


def send_commtrace_msg(msg:str,timestamp_rcvd:int,serverAddressPort:str,log:Logger,bufferSize=2048):
    if msg.metadata != {}:
        bytesToSend, msg_to_commtrace = msg_commtrace(msg,timestamp_rcvd)

        UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        #log.info("Enviando mensagem para Contrace")
        UDPClientSocket.sendto(bytesToSend, serverAddressPort)
        #log.info("Mensagem para Contrace: {}", msg_to_commtrace)
    else:
        #log.warn("Could not send message to Commtrace because msg.metadata is empty")
        pass


def main():

    # Carregar configurações
    config = json.load(open('etc/conf/options.json'))

    broker_uri = config['broker_uri']
    zipkin_host = config['zipkin_uri']

    camera_id = sys.argv[1]

    sd = SkeletonsDetector(config)

    service_name = 'Skeleton.Detector'
    log = Logger(name=service_name)
    channel = StreamChannel()
    log.info(f'Connected to broker {broker_uri}')

    exporter = create_exporter(service_name, zipkin_host, log)
    subscription = Subscription(channel=channel)
    subscription.subscribe(f'CameraGateway.{camera_id}.Frame')
    
    # Calibração da câmera
    calib_path = 'calib'
    camCalibs = [calib_data(f'{calib_path}/calib_rt1.npz'), calib_data(f'{calib_path}/calib_rt2.npz'), calib_data(f'{calib_path}/calib_rt3.npz'), calib_data(f'{calib_path}/calib_rt4.npz')]

    img_list = list()

    while True:

        msg = channel.consume()

        # camera_id = get_topic_id(msg.topic)

        if type(msg) == bool or camera_id == '5' or camera_id == '6':
            continue

        #camera_id = get_topic_id(msg.topic)
        #timestamp_rcvd = time.time()
        #serverAddressPort = (config['conmtrace_host'], config['conmtrace_port'])
        #send_commtrace_msg(msg,timestamp_rcvd,serverAddressPort,log)

        tracer = Tracer(exporter=exporter, span_context=msg.extract_tracing())
        span = tracer.start_span(name='detection_and_render')

        with tracer.span(name='unpack'):
            img = msg.unpack(Image)
            im_np = to_np(img)
            im_np = calib_img_from_file(camCalibs[int(camera_id)-1], im_np)

        img_list.append(im_np)
        # print(img_list)


        if len(img_list) >= 1: 
            # detection_span = None
            
            with tracer.span(name='detection') as _span:
                results = sd.detect(im_np)

                # detection_span = _span
            with tracer.span(name='pack_and_publish_detections'):

                skeleton_msg = Message()
                skeleton_msg.topic = f'SkeletonDetector.{camera_id}.Detection'
                skeleton_msg.inject_tracing(span)
                
                obs_annotations = sd.to_object_annotations(results, im_np.shape)
                skeleton_msg.pack(obs_annotations)
                skeleton_msg.created_at = time.time()
                channel.publish(skeleton_msg)

            with tracer.span(name='render_pack_publish'):

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


                rendered_msg = Message()
                rendered_msg.topic = f'SkeletonDetector.{camera_id}.Rendered'
                rendered_msg.inject_tracing(span)
                rendered_msg.pack(to_image(im_np))
                rendered_msg.created_at = time.time()  # Certificando-se de empacotar corretamente
                channel.publish(rendered_msg)

            
            del img_list[0]


        tracer.end_span()

if __name__ == "__main__":
    main()
