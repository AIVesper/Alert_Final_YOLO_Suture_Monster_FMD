import os
import numpy as np
import cv2.cv2 as cv2
from imutils.video import FPS
import argparse
import datetime
import time
import uuid
import random
import pyimgur
from store_image import storeImage
from linebot import LineBotApi
from fire import getName, getDevice, getLink, getAll
from bigdataProxy import injectNotificationDataSet
from linebot.exceptions import LineBotApiError
from linebot.models import TextSendMessage, TemplateSendMessage, ButtonsTemplate, URITemplateAction, MessageAction, URIAction
from camera.app import getHost
#from pyngrok import ngrok
#http_tunnel = ngrok.connect(5051)
# use index 1 for mac camera
if(getLink() == '0'):
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(getLink())
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))
# print(getDevice())
line_bot_api = LineBotApi(
    "5OjwvGGPi4zutObUFkeeOQ5Cf712R7cwPFinDqyNbMFrWi4zTOF4/QXAbM1Vj/Be5LriCleS8HQmjABnGrKWb1WocThH1l6Q5QyQySDQss57hkE5sS76x2hdEKfqOWcW7+PEp5WD/yHXurbCa2fR0gdB04t89/1O/w1cDnyilFU=")
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], ".names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = [[0, 255, 0], [0, 0, 255]]

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolo.weights"])
configPath = os.path.sep.join([args["yolo"], "yolo.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
next_frame_towait = 5  # for sms
fps = FPS().start()
frameId = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # DETECTION WITH YOLO
    # load our input image and grab its spatial dimensions
    (H, W) = frame.shape[:2]
    frameId += 1
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])
    border_size = 100
    border_text_color = [255, 255, 255]

    filtered_classids = np.take(classIDs, idxs)
    mask_count = (filtered_classids == 0).sum()
    nomask_count = (filtered_classids == 1).sum()
    # display count
    text = "NoMaskCount: {}".format(nomask_count)
    cv2.putText(frame, text, (W-170, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, border_text_color, 2)
    text = "MaskCount: {}".format(mask_count)
    cv2.putText(frame, text, (W-170, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, border_text_color, 2)
    text = "AllCount: {}".format(mask_count+nomask_count)
    cv2.putText(frame, text, (W-170, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, border_text_color, 2)
    # display status
    ratio = nomask_count/(mask_count+nomask_count+0.000001)
    out.write(frame)
    if len(idxs) > 0:
    # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
    if ratio != 0 and np.isnan(ratio) != True:
        text = "Warning !"
        cv2.putText(frame, text, (W-170, int(border_size-50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, [13, 23, 227], 2)
        if fps._numFrames >= next_frame_towait:
            print("[INFO] YOLOV3 took {:.6f} seconds to capture the person without the mask.".format(
                end - start))
            all_info = getAll()
            sb_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            sb_url = str(storeImage(frame, str(uuid.uuid5(
                uuid.NAMESPACE_DNS, str(uuid.uuid1()) + str(random.random())))))
            total_count = int(nomask_count)+int(mask_count)
            if(getLink() == '0'):
                line_bot_api.broadcast(TemplateSendMessage(alt_text='偵測到有人未帶口罩，請盡速查看！', template=ButtonsTemplate(title='場域：'+str(getName()), thumbnail_image_url=sb_url, text="場域內總人數："+str(total_count)+"\n警示事件：有 "+str(
                    nomask_count)+" 人沒戴口罩\n"+"擷取時間："+str(sb_time), actions=[URIAction(label='統計報表', uri='https://datastudio.google.com/u/5/reporting/0420b197-cbec-4bbe-84e6-29f95dd1fe08/page/9qmvB')])))
                line_bot_api.broadcast(TextSendMessage(
                    text="辨識裝置："+str(all_info['device'])+"\n辨識時間：{:.6f} seconds".format(end - start)+"\n串流鏈接：http://"+str(getHost())+":5051/video"))
                injectNotificationDataSet(str(all_info['device']), sb_url, str(sb_time), str(all_info['area']), str(
                    all_info['stream_link']), str(nomask_count), str(int(nomask_count)+int(mask_count)))
            else:
                line_bot_api.broadcast(TemplateSendMessage(alt_text='偵測到有人未帶口罩，請盡速查看！', template=ButtonsTemplate(title='場域：'+str(getName()), thumbnail_image_url=sb_url, text="場域內總人數："+str(total_count)+"\n警示事件：有 "+str(
                    nomask_count)+" 人沒戴口罩\n"+"擷取時間："+str(sb_time), actions=[URIAction(label='統計報表', uri='https://datastudio.google.com/u/5/reporting/0420b197-cbec-4bbe-84e6-29f95dd1fe08/page/9qmvB')])))
                line_bot_api.broadcast(TextSendMessage(text="辨識裝置："+str(all_info['device'])+"\n辨識時間：{:.6f} seconds".format(
                    end - start)+"\n串流鏈接：http://"+getLink()))
                injectNotificationDataSet(str(all_info['device']), sb_url, str(sb_time), str(all_info['area']), str(
                    all_info['stream_link']), str(nomask_count), str(int(nomask_count)+int(mask_count)))

            next_frame_towait = fps._numFrames+(5*15)

    else:
        text = "Safe "
        cv2.putText(frame, text, (W-100, int(border_size-50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, [0, 255, 0], 2)



    # Display the resulting frame
    cv2.imshow('frame', frame)
    fps.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
fps.stop()
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
