import streamlit as st
from PIL import Image, ImageDraw
#from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from torchvision import models
import torchvision.transforms as T
import cv2
import numpy as np

st.set_page_config(layout="wide")



st.markdown(
    """
    <style>
    .centered-text {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)





COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]




def get_prediction(img, threshold):
    img=img.resize((250, 200))
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
    pred = model([img]) # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


def load_model():
    # load the model for inference 
    model = models.segmentation.fcn_resnet101(weights=True).eval()
    return model

label_colors = np.array([(0, 0, 0),  # 0=background
              # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
              (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
              # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
              (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
              # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
              (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
              # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
              (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

def seg2rgb(preds):
    colors = label_colors
    colors = label_colors.astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    rgb = Image.fromarray(preds.byte().cpu().numpy())#.resize(preds.shape)
    rgb.putpalette(colors)
    return rgb



def get_segmentation(img_file, model):
    input_image = img_file
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available


    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions


selected_button = st.sidebar.radio("Select a Task:", ("Object Detection", "Image Segmentation"))

if selected_button == "Object Detection":
    st.title("Object Detection")
    
    
    # Upload image through Streamlit
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    col1, col2 = st.columns(2)
    
    if uploaded_image is not None:
        # Display the original image
    
    
        # Process the image for object detection
        image = Image.open(uploaded_image)
        try:
            boxes, pred_cls = get_prediction(image, threshold=0.7) # Get predictions
            show = 1
        except Exception:
            st.warning("Upload another image to see the magic")
            show = 0
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        image_array = cv2.resize(image_array, (250,200))
        img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) # Convert to RGB
        with col1:
            st.markdown("<h2 class='centered-text'>Uploaded Image</h2>", unsafe_allow_html=True)
            st.image(img, caption=" ", use_column_width=True)
        
        rect_th = 1 
        text_size = 0.5
        text_th = 1
    
        if show == 1:
            for box, cls in zip(boxes, pred_cls):#range(len(boxes)):
                box0=list(box[0])
                box0=[int(bo) for bo in box0]
                box1=list(box[1])
                box1=[int(b1) for b1 in box1]
                cv2.rectangle(image_array, box0, box1,color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
                cv2.putText(image_array, cls, box0,  cv2.FONT_HERSHEY_COMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
                image_array = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
                #print(img)
    
        # Display the object-detected 
        with col2:
            st.markdown("<h2 class='centered-text'>Image with Detected Objects</h2>", unsafe_allow_html=True)
            st.image(image_array, caption="", use_column_width=True)




if selected_button == "Image Segmentation":
    
    st.title("Image Segmentation")
    
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    col1, col2 = st.columns(2)
    
    if uploaded_image is not None:
        # Display the original image
    
        # Process the image for object detection
        image = Image.open(uploaded_image)
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        image_array = cv2.resize(image_array, (250,200))
        img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) # Convert to RGB
        with col1:
            st.markdown("<h2 class='centered-text'>Uploaded Image</h2>", unsafe_allow_html=True)
            st.image(img, caption="", use_column_width=True)
            model = load_model()
            preds = get_segmentation(image_array, model)
            rgb = seg2rgb(preds)
            
        with col2:
            st.markdown("<h2 class='centered-text'>Segmented Image</h2>", unsafe_allow_html=True)
            st.image(rgb, caption="", use_column_width=True)









