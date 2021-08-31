import imghdr
import os
import io
import shutil
import ssl
import tempfile
# import urllib.request
from pathlib import Path
import helper as h
import cv2
import numpy as np
# import pandas as pd
import requests
import streamlit as st
from PIL import Image
import json

DARK_BLUE = (139, 0, 0)
URL = os.environ.get('INGRESS_HOST')
URL = 'localhost'
BASE_URL = 'http://' + URL + ':8000'
ENDPOINT = '/predict'
MODEL = 'yolov4'
s = requests.Session()

st.title('Welcome to Room 6')
st.write(" ------ ")
ssl._create_default_https_context = ssl._create_unverified_context

IMAGE_DISPLAY_SIZE = (330, 330)
IMAGE_DIR = 'demo_photo'
TEAM_DIR = 'team'

# MODEL_WEIGHTS = f'{DEFAULT_MODEL_BASE_DIR}/hpe_epoch107_.hdf5'
# MODEL_JSON = f'{DEFAULT_MODEL_BASE_DIR}/hpe_hourglass_stacks_04_.json'

MODEL_WEIGHTS_DEPLOYMENT_URL = 'https://github.com/robertklee/COCO-Human-Pose/releases/download/v0.1-alpha/hpe_epoch107_.hdf5'
MODEL_JSON_DEPLOYMENT_URL = 'https://github.com/robertklee/COCO-Human-Pose/releases/download/v0.1-alpha/hpe_hourglass_stacks_04_.json'

MAXIMO_VISUAL_INSPECTION_API_URL = 'https://mas83.visualinspection.maximo26.innovationcloud.info/api/dlapis/9ffb662b-790a-4fb2-a419-217fdf1ac0ce'

# Constants for sidebar dropdown
SIDEBAR_OPTION_PROJECT_INFO = "Show Project Info"
# SIDEBAR_OPTION_DEMO_IMAGE = "Select a Demo Image"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload an Image"
# SIDEBAR_OPTION_MEET_TEAM = "Meet the Team"

SIDEBAR_OPTIONS = [SIDEBAR_OPTION_PROJECT_INFO, SIDEBAR_OPTION_UPLOAD_IMAGE]

def response_from_server(url, image_file, verbose=True):
    """Makes a POST request to the server and returns the response.

    Args:
        url (str): URL that the request is sent to.
        image_file (_io.BufferedReader): File to upload, should be an image.
        verbose (bool): True if the status of the response should be printed. False otherwise.

    Returns:
        requests.models.Response: Response from the server.
    """
    with open('test.jpg', 'rb') as f:
            # WARNING! verify=False is here to allow an untrusted cert!
            response = requests.post(MAXIMO_VISUAL_INSPECTION_API_URL,
                    files={'files': ('test.jpg', f)},
                    verify=False)
    status_code = response.status_code
    if verbose:
        msg = "Everything went well!" if status_code == 200 else "There was an error when handling the request."
        print(msg)
    return json.loads(response.text)

def get_image_from_response(response):
    """Display image within server's response.

    Args:
        response (requests.models.Response): The response from the server after object detection.
    """
    
    image_stream = io.BytesIO(response.content)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image
    # filename = "image_with_objects.jpeg"
    # cv2.imwrite(f'images_predicted/{filename}', image)
    # display(Image(f'images_predicted/{filename}'))


def getROI(filename,jsonfile):
    cars = 0
    trackers = []
    counters = {
        'left_lane': 0,
        'right_lane': 0,
        'lost_trackers': 0,
        'frames': 0,
    }





    counters['frames'] += 1
    img = cv2.imread(filename)

    boxes, counters,trackers = h.update_trackers(img, counters,trackers)
    cars = 0

    jsonresp = jsonfile
    print(type(jsonresp))
    print(jsonresp['classified'])

    for obj in h.not_tracked(jsonresp['classified'], boxes):
        if obj['confidence']>0.60:
            if h.in_range(obj):
                cars += 1
                h.add_new_object(obj, img, cars,trackers)  # Label and start tracking

    # Draw the running total of cars in the image in the upper-left corner


    cv2.putText(img, 'Dents detected: ' + str(cars), (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, DARK_BLUE, 4, cv2.LINE_AA)


    print("\nDone")
    return img

def run_app(img):

    left_column, right_column = st.columns(2)

    # xb, yb = app_helper.load_and_preprocess_img(img, num_hg_blocks=1)
    # display_image = cv2.resize(np.array(xb[0]), IMAGE_DISPLAY_SIZE,
    #                     interpolation=cv2.INTER_LINEAR)
    display_img = img #np.array(Image.open(img).convert('RGB'))
    # url_with_endpoint_no_params = BASE_URL + ENDPOINT
    # full_url = url_with_endpoint_no_params + "?model=" + MODEL
    print(MAXIMO_VISUAL_INSPECTION_API_URL)
    #image_file = Image.open(display_img)
    display_img=np.array(Image.open(display_img).convert('RGB'))
    cv2.imwrite("original.jpg",display_img)
    cv2.imwrite("result.jpg", display_img)

    rc, jsonresp = h.detect_objects("result.jpg",s,MAXIMO_VISUAL_INSPECTION_API_URL)

    result_img = getROI("result.jpg", jsonresp)
    st.write(jsonresp)
    left_column.image(img, caption = "Selected Input")
    right_column.image(result_img,  caption = "Predicted Keypoints")



def main():

    st.sidebar.warning('\
        Please upload SINGLE-person images. For best results, please also CENTER the person in the image.')
    st.sidebar.write(" ------ ")
    st.sidebar.title("Explore the Following")

    app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)

    if app_mode == SIDEBAR_OPTION_PROJECT_INFO:
        st.sidebar.write(" ------ ")
        st.sidebar.success("Project information showing on the right!")
        # st.write(get_file_content_as_string("Project_Info.md"))

    elif app_mode == SIDEBAR_OPTION_UPLOAD_IMAGE:


        f = st.sidebar.file_uploader("Please Select to Upload an Image", type=['png', 'jpg', 'jpeg', 'tiff', 'gif'])
        if f is not None:
            st.sidebar.write('Please wait for the magic to happen! This may take up to a minute.')
            run_app(f)
        #upload = st.empty()
        #with upload:

    else:
        raise ValueError('Selected sidebar option is not implemented. Please open an issue on Github: https://github.com/robertklee/COCO-Human-Pose')

main()
expander_faq = st.expander("More About Our Project")
expander_faq.write("Hi there! If you have any questions about our project, or simply want to check out the source code, please visit our github repo: https://github.com/robertklee/COCO-Human-Pose")
