import numpy as np

import cv2
import dlib
import time

from videocaptureasync import VideoCaptureAsync

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient

from kortex_api.SessionManager import SessionManager

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

from kortex_api.autogen.messages import Session_pb2, Base_pb2

from ultralytics import YOLO

def find_one_face(frame, detector, FACTOR):
    results = detector(frame)
    if len(results) > 0:
        if len(results[0].boxes) > 0:   
            if results[0].boxes[0].conf > 0.5:
                bbx = results[0].boxes[0].xyxy.cpu().numpy()[0]
                return int(bbx[0] / FACTOR), int(bbx[1] / FACTOR), int(bbx[2] / FACTOR), int(bbx[3] / FACTOR)
    return None

def close_gripper(base_client_service):
    # Create the GripperCommand we will send
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.value = 1
    base_client_service.SendGripperCommand(gripper_command)

def open_gripper(base_client_service):
    # Create the GripperCommand we will send
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.value = 0
    base_client_service.SendGripperCommand(gripper_command)

def send_home(base_client_service):
    print('Going Home....')
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base_client_service.ReadAllActions(action_type)
    action_handle = None
    
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    base_client_service.ExecuteActionFromReference(action_handle)
    time.sleep(6)
    print("Done!")

def get_distance(p1, p2):
    return (p2[0]-p1[0], p2[1]-p1[1])

def twist_command(base_client_service, cmd):
    command = Base_pb2.TwistCommand()
    # command.mode = Base_pb2.
    command.duration = 0  # Unlimited time to execute

    x, y, z, tx, ty, tz = cmd

    twist = command.twist
    twist.linear_x = x
    twist.linear_y = y
    twist.linear_z = z
    twist.angular_x = tx
    twist.angular_y = ty
    twist.angular_z = tz
    base_client_service.SendTwistCommand(command)

if __name__ == "__main__":

    model = YOLO("assets/yolov8n-face.pt")

    DEVICE_IP = "192.168.1.10"
    DEVICE_PORT = 10000

    # Setup API
    error_callback = lambda kException: print("_________ callback error _________ {}".format(kException))
    transport = TCPTransport()
    router = RouterClient(transport, error_callback)
    transport.connect(DEVICE_IP, DEVICE_PORT)

    # Create session
    print("Creating session for communication")
    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = 'admin'
    session_info.password = 'admin'
    session_info.session_inactivity_timeout = 60000   # (milliseconds)
    session_info.connection_inactivity_timeout = 2000 # (milliseconds)
    print("Session created")

    session_manager = SessionManager(router)   
    session_manager.CreateSession(session_info)

    # Create required services
    base_client_service = BaseClient(router)

    send_home(base_client_service)

    video_capture = VideoCaptureAsync("rtsp://192.168.1.10/color", use_rtsp=True)
 
    FACTOR = 0.3
    VEL = 100
    movs = {
        "look_up": np.array((0,0,0,-VEL,0,0)),
        "look_left": np.array((0,0,0,0, VEL,0)),
        "stop": np.array((0,0,0,0,0,0))
    }
    
    # look up
    # twist_command(base_client_service, list(movs["look_up"]))
    # time.sleep(0.5)
    # twist_command(base_client_service, list(movs["stop"]))

    video_capture.start()
    while True:
        ret, frame = video_capture.read()

        center_X = int(frame.shape[1]/2)
        center_Y = int(frame.shape[0]/2)
        
        start = time.time()

        res = cv2.resize(frame, None, fx=FACTOR, fy=FACTOR)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        
        cmd = np.zeros(6)
        face = find_one_face(res, model, FACTOR)
        
        if face is not None:
            print("Face found")
            face_center = (int((face[0]+face[2])/2), int((face[1]+face[3])/2))
            
            cv2.rectangle(frame, face[:2], face[2:], (0, 255, 0), 2)
            cv2.circle(frame, (center_X, center_Y), 2, (255, 0, 0), 2)
            
            distance = get_distance(face_center, (center_X, center_Y))
            if distance[0] > 50 or distance[0] < -50:
                cmd += movs["look_left"] * distance[0] * 0.5

            if distance[1] > 50 or distance[1] < -50:
                cmd += movs["look_up"] * distance[1] * 0.5

            cmd = np.clip(cmd, -5, 5)

        twist_command(base_client_service, list(cmd))

        cv2.imshow('Video', frame)

        if cv2.waitKey(1):
            if 0xFF == ord('q'):
                break
            if 0xFF == ord('h'):
                send_home(base_client_service)
    

        end = time.time()

        print(chr(27) + "[2J")  # Clear terminal
        print("Center = {}\nVel Vec = {}\nFPS: {}".format((center_X, center_Y), cmd, 1/(end - start)))

    video_capture.stop()
    cv2.destroyAllWindows()

    send_home(base_client_service)

    print('Closing Session..')
    session_manager.CloseSession()
    router.SetActivationStatus(False)
    transport.disconnect()
    print('Done!')