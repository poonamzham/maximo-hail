import imghdr
import os
import io
import shutil
import ssl
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
# import pandas as pd
import requests
import streamlit as st
from PIL import Image


URL = os.environ.get('INGRESS_HOST')
URL = 'localhost'
BASE_URL = 'http://' + URL + ':8000'
ENDPOINT = '/predict'
MODEL = 'yolov4'


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
    #files = {'img': ('test.jpeg', image_file, 'image/jpeg')}
    # r = s.post(MAXIMO_VISUAL_INSPECTION_API_URL,
    #                files={‘files’: (filename, f)},
    #                verify=False)

    files = {'file': ('test.jpg', image_file)}
    response = requests.post(url, files=files, verify=False)
    status_code = response.status_code
    if verbose:
        msg = "Everything went well!" if status_code == 200 else "There was an error when handling the request."
        print(msg)
    return response

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


def run_app(img):

    left_column, right_column = st.columns(2)

    # xb, yb = app_helper.load_and_preprocess_img(img, num_hg_blocks=1)
    # display_image = cv2.resize(np.array(xb[0]), IMAGE_DISPLAY_SIZE,
    #                     interpolation=cv2.INTER_LINEAR)
    display_img = img #np.array(Image.open(img).convert('RGB'))
    # url_with_endpoint_no_params = BASE_URL + ENDPOINT
    # full_url = url_with_endpoint_no_params + "?model=" + MODEL
    print(MAXIMO_VISUAL_INSPECTION_API_URL)

    image_file = Image.open(display_img)
    
    image_file.save("test.jpg")

    with open("test.jpg", "rb") as pred_file:
        prediction = response_from_server(MAXIMO_VISUAL_INSPECTION_API_URL, pred_file)

    # prediction = response_from_server(full_url, image_file)

    print(prediction.status_code)
    print(prediction)


    # result_img = get_image_from_response(prediction)
    result_img = image_file
        
    left_column.image(image_file, caption = "Selected Input")

    # handle, session = load_model()


    # scatter = handle.predict_in_memory(img, visualize_scatter=True, visualize_skeleton=False)
    # skeleton = handle.predict_in_memory(img, visualize_scatter=True, visualize_skeleton=True)

    # scatter_img = Image.fromarray(scatter)
    # skeleton_img = Image.fromarray(skeleton)

    right_column.image(result_img,  caption = "Predicted Keypoints")
    # st.image(skeleton_img, caption = 'FINAL: Predicted Pose')

# def demo():
    # left_column, middle_column, right_column = st.beta_columns(3)

    # left_column.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR,'skier.png'), caption = "Demo Image")

    # middle_column.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'skier_output.png'), caption = "Predicted Heatmap")

    # right_column.subheader("Explanation")
    # right_column.write("We predict human poses based on key joints.")

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
expander_faq = st.beta_expander("More About Our Project")
expander_faq.write("Hi there! If you have any questions about our project, or simply want to check out the source code, please visit our github repo: https://github.com/robertklee/COCO-Human-Pose")
