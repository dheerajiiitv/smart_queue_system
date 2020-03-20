import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin
import os
import cv2
import argparse
import time

class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d

class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self):
        '''
        TODO: This method needs to be completed by you
        '''
        self.network = None
        self.exec_net = None
        self.input_blob  = None
        self.output_blob = None

    def load_model(self, model_path, extensions, device):
        '''
        TODO: This method needs to be completed by you
        '''
        self.network = IENetwork(model=model_path+'.xml', weights=model_path+'.bin')
        self.plugin  =IEPlugin(device=device)

        if extensions:
            self.plugin.add_cpu_extension(extensions)
        supported_layer = self.plugin.get_supported_layers(self.network)
        if len(supported_layer) < len(self.network.layers):
            print("Some layers are not supported please add them")
            exit(1)
        self.input_blob  = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        self.exe_net = self.plugin.load(network=self.network, num_requests=1)
    

        return self.exe_net

    def check_plugin(self, plugin):
        '''
        TODO: This method needs to be completed by you
        '''
        raise NotImplementedError
        
    def predict(self, image,weights, height):
        '''
        TODO: This method needs to be completed by you

        '''
        
        preprocessed_image = self.preprocess_input(image)

        return self.preprocess_outputs(self.exe_net.infer(inputs={self.input_blob:preprocessed_image}), weights, height, image)


    def preprocess_outputs(self, outputs, weights, height,frame, conf_thres=0.5):
        '''
        TODO: This method needs to be completed by you
        '''
        
        output = outputs[self.output_blob]
        boxes = output[0][0]
        person_count = 0
        person_coord_list = []
        for box in boxes:
            conf = box[2]
            # Person class filter
            if conf > conf_thres and box[1] == 1:

                x_min = int(box[3] * weights)
                y_min =  int(box[4] * height)
                x_max = int(box[5] * weights)
                y_max =  int(box[6] * height)
                person_count+=1
    #             print((box[3], box[4]), (box[5], box[6]))
                person_coord_list.append([x_min, y_min, x_max, y_max])
                frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),(0, 0, 255), 1)


#         print("Drawing box done!")
        return person_coord_list, frame

    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        _,_, height, width = self.network.inputs[next(iter(self.network.inputs))].shape
        image = cv2.resize(image, (width, height))
        image = image.transpose(2,0,1)
        image = image.reshape(1, 3, height, width)

        return image

def main(args):
    extensions=args.extensions
    model=args.model
    device=args.device
    visualise=args.visualise

    start=time.time()
    pd=PersonDetect()
    pd.load_model(model_path=model,extensions=extensions, device=device)
    print("Time taken to load the model is:" ,time.time()-start)
    #  # Queue Parameters
    # # For retail
    # queue.add_queue([620, 1, 915, 562])
    # queue.add_queue([1000, 1, 1264, 461])
    # # For manufacturing
    # queue.add_queue([15, 180, 730, 780])
    # queue.add_queue([921, 144, 1424, 704])	
    # # For Transport 
    # queue.add_queue([50, 90, 838, 794])
    # queue.add_queue([852, 74, 1430, 841])


    queue=Queue()
    queue.add_queue([620, 1, 915, 562])
    queue.add_queue([1000, 1, 1264, 461])
    video_file=args.video
    cap=cv2.VideoCapture(video_file)
    width = int(cap.get(3))
    height = int(cap.get(4))
    i = 0
    while cap.isOpened():
        ret, frame=cap.read()
        if not ret:
            continue

       
    
        if visualise:
            coords, image=pd.predict(frame, width, height)
            num_people=queue.check_coords(coords)
#             cv2.imwrite("frame"+str(i)+".jpg", image)
            i+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            _,coords=pd.predict(frame, width, height)
#             print(coords)

        print("Total People in frame = ", len(coords))
        print("Number of people in queue = ",num_people)

    cap.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extensions', default=None)
    
    parser.add_argument('--visualise', action='store_true')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--max_people', default='To be given by you')
    parser.add_argument('--threshold', default=0.5)
    
    args=parser.parse_args()

    main(args)


