import streamlit as st
import altair as alt
import os
import random
import cv2
import numpy as np
import tempfile
from PIL import Image
from ultralytics import YOLO

# page config
st.set_page_config(
    page_title="Mango Leaf Disease Image Detection and Instance Segmentation Using YOLOv12",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)
alt.theme.enable("quartz")

if 'page_selection' not in st.session_state:
    st.session_state.page_selection="about"

def set_page_selection(page):
    st.session_state.page_selection = page

# sidebar
with st.sidebar:
    st.title("Mango Leaf Disease Image Detection and Instance Segmentation Using YOLOv12 🥭")
    st.subheader("Pages")
    if st.button("ℹ️ About", width='stretch', on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = "about"
    if st.button("🍃 Dataset", width='stretch', on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = "dataset"
    if st.button("✏️ Annotation", width='stretch', on_click=set_page_selection, args=('annotation',)):
        st.session_state.page_selection = "annotation"
    if st.button("⬆️ Augmentation", width='stretch', on_click=set_page_selection, args=('augmentation',)):
        st.session_state.page_selection = "augmentation"
    if st.button("🏋️‍♀️ Training Results", width='stretch', on_click=set_page_selection, args=('training results',)):
        st.session_state.page_selection = "training results"
    if st.button("👁 Detection", width='stretch', on_click=set_page_selection, args=('detection',)):
        st.session_state.page_selection = "detection"

    st.subheader("Contributors")
    st.markdown("* Butial, A.P.\n* Pagdanganan, J.M.\n* Ricafranca, S.\n* Tan, J. R.\n* Dr. Comia, L.")

    st.subheader("Relevant Links")
    st.markdown("🍃 [MangoLeafDB Dataset](https://data.mendeley.com/datasets/hxsnvwty3r/1)")
    st.markdown("🤖 [Roboflow Annotated Dataset](https://universe.roboflow.com/artificial-intelligence-u9ca8/mango-leaf-image-detection-qvw37)")
    st.markdown("📔 [Google Colab Notebook](https://colab.research.google.com/drive/1Brkn02jJj5Y0SqI8GFjameKClZXUH187?usp=sharing)")
    st.markdown("🗄️ [GitHub Repository](https://github.com/VannCodes/mango_streamlit.git)")

# functions
CLASSES = [
    "Anthracnose",
    "Bacterial Canker",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]
def display_dataset_gallery():
    PATH = 'data/MangoLeafBD Dataset'
    SAMPLES = 3

    for cls in CLASSES:
        st.subheader(cls)
        class_dir = os.path.join(PATH, cls)

        images = os.listdir(class_dir)
    
        selected = random.sample(images, min(SAMPLES, len(images)))
        cols = st.columns(len(selected))
        for i, img_name in enumerate(selected):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).resize((640,640))
            with cols[i]:
                st.image(img, caption=img_name, width='stretch')

def display_annotations():
    PATH = 'resource/annotate'
    for cls in CLASSES:
        st.subheader(cls)
        class_dir = os.path.join(PATH, cls)

        images = os.listdir(class_dir)
        cols = st.columns(len(images))
        for i, img_name in enumerate(images):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path)
            with cols[i]:
                st.image(img, caption=f"{cls} Annotation Sample {i+1}", width='stretch')
    
def detection():
    model = YOLO('model/best.pt')
    st.success("✅ YOLOv12 segmentation model loaded successfully!")

    uploaded_file = st.file_uploader("Upload a mango leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            img_path = tmp.name
        results = model(img_path)

        for r in results:
            img = cv2.imread(img_path)

            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                for mask in masks:
                    mask = mask.astype(np.uint8) * 255
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    colored_mask = np.zeros_like(img)
                    colored_mask[:, :, 2] = mask  # red channel
                    img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            names = model.names

            for box, score, cls_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"{names[cls_id]} {score:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                offset = 10
                text_y = y1 + offset
                if text_y < 0:
                    text_y = y1 + 20
                cv2.putText(img, label, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with col2:
                st.subheader("Output")
                st.image(img_rgb, use_container_width=True)

    
# About page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")
    st.markdown("""
    This is a machine project for instance segmentation of mango leaf disease images for detection task using the YOLOv12 model. 
The image samples were sourced from the [MangoLeafBD Dataset](https://data.mendeley.com/datasets/hxsnvwty3r/1), which consists of 
different images of mango leaves diagnosed with and without diseases. Annotation and augmentation was done in Roboflow to enhance 
the model's performance. The trained model achieved an outstanding **99.5% mAP50 and 98.9% mAP50-95 scores** on the validation set, with its best performance being on the 79th epoch, 
indicating how well it performed for the detection task.

This Streamlit website application is for the documentation and demonstration of the outcome of the machine project.
                
### Pages
- 🍃 `Dataset` - Provides a brief description of the MangoLeafBD dataset use in the machine project and random, resized (640x640) image samples from different classes.
- ✏️ `Annotation` - Provides samples of annotated images using Roboflow.
- ⬆️ `Augmentation` - Details the train:validation:test ratio before and after augmentation and enumerates preprocessing steps and filters used in Roboflow.
- 🏋️‍♀️ `Training Results` - Shows the YOLOv12 model's training results in graphical format.
- 👁 `Detection` - Testing playground for users that allows them to upload images of mango leaves to assess the model's performance in unseen data.
""")
    
# Dataset page
if st.session_state.page_selection == "dataset":
    st.header("🍃 Dataset")
    st.markdown("""
    The [MangoLeafDB Dataset](https://data.mendeley.com/datasets/hxsnvwty3r/1) is a dataset of 4000 image samples of mango leaves belonging from eight different classes that 
are equally distributed (500 per class): `Anthracnose`, `Bacterial Canker`, `Cutting Weevil`, `Die Back`, `Gall Midge`, `Healthy`, `Powdery Mildew`, and `Sooty Mould`. The methodology of the project excluded
the Cutting Weevil class due to the lack of suitable annotation tools for instance segmentation with its complex shape. All images indicate symptoms of diseases caused by fungi, bacteria, and pathogens, 
but for images belonging from the Healthy class. The original size of the images is (240x320) but was resized to (640x640) in the web application for presentation purposes.
""")
    display_dataset_gallery()

# Annotation page
if st.session_state.page_selection == "annotation":
    st.header("✏️ Annotation")
    st.markdown("""
    The machine project manually performed annotation using segmentation masks in Roboflow. Loading the dataset in Roboflow for annotation reduced the number of samples from `3500` to `3479` images after the environment's automatic removal of duplicated images. 
The annotated dataset is publicly available in Roboflow listed as [Mango Leaf Image Detection Computer Vision Model](https://universe.roboflow.com/artificial-intelligence-u9ca8/mango-leaf-image-detection-qvw37) for review and testing.
                
Provided below are the screenshots of some of the annotations performed on the seven classes. 
""")
    display_annotations()

# Augmentation page
if st.session_state.page_selection == "augmentation":
    st.header("⬆️ Augmentation")
    st.markdown("""
    Prior to augmentation, the train:validation:test was split into 70:15:15 ratio. With augmentation, the number of samples in the training set tripled, which resulted to a new ratio of 88:6:6. 
The tables below show the distribution of samples in the three sets before and after augmentation.                
""")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            ### Before Augmentation
                
| Train | Validation | Test |
|-------|------------|------|
| 2435 | 522 | 522 |

**Total = 3479**
___

The specification of augmentations applied is listed below, as provided by Roboflow, which used the default arguments.

## Augmentations                
* `Outputs per training example`: 3
* `Flip`: Horizontal, Vertical
* `90° Rotate`: Clockwise, Counter-Clockwise, Upside Down
* `Crop`: 0% Minimum Zoom, 20% Maximum Zoom
* `Rotation`: Between -15° and +15°
* `Grayscale`: Apply to 15% of images
* `Hue`: Between -15° and +15°
* `Saturation`: Between -25% and +25%
* `Brightness`: Between -15% and +15%
* `Blur`: Up to 2.5px
* `Noise`: Up to 0.1% of pixels
        """)
    with col2:
        st.markdown(
            """
            ### After Augmentation
| Train | Validation | Test |
|-------|------------|------|
| 7314 | 522 | 519 |

**Total = 8355**
___

Likewise, the preprocessing steps are listed below, as also provided by Roboflow, which used the default arguments.

## Preprocessing                
* `Auto-Orient`: Applied
* `Resize`: Stretch to 640x640
        """)

# Training Results page
if st.session_state.page_selection == "training results":
    st.header("🏋️‍♀️ Training Results")
    col1, col2 = st.columns(2)
    IMG_WIDTH = 640
    IMG_HEIGHT = 420

    with col1:
        st.subheader("Total Training Loss")
        tt_loss = Image.open("resource/graphs/training_total_loss.png")
        tt_loss = tt_loss.resize((IMG_WIDTH, IMG_HEIGHT))
        st.image(tt_loss, width='stretch')
        st.markdown(
            """
            The Total Training Loss curve illustrates the model's overall optimization performance across 
            all three components: segmentation, classification, and bounding box. After epoch 30, the curve steadily flattens out 
            and settles close to 1.0, indicating that the model has learned consistently and effectively with little overfitting. 
            This consistent convergence is a result of balanced optimization for each task component and strong overall training stability.
        """)

        st.subheader("Bounding Box mAP50 Results")
        box = Image.open("resource/graphs/map50_box_results.png")
        box = box.resize((IMG_WIDTH, IMG_HEIGHT))
        st.image(box, width='stretch')
        st.markdown(
            """
            The model's high detection accuracy during training is demonstrated by the mAP50 (Bounding Box) curve. After a small early fluctuation 
            over the first few epochs, the curve begins to rise significantly and stabilizes around 1.0 after around the tenth epoch. This sharp 
            increase and long-lasting plateau show that the model picked up the ability to locate objects with remarkable accuracy very fast, 
            and it continued to perform consistently during the following epochs.
        """)

        st.subheader("Normalized Confusion Matrix")
        matrix = Image.open("resource/graphs/confusion_matrix_normalized.png")
        matrix = matrix.resize((IMG_WIDTH, IMG_HEIGHT))
        st.image(matrix, width='stretch')
        st.markdown(
            """
            The confusion matrix represents the model’s classification performance across all categories. 
            Each row represents the true class, and each column represents the predicted class. The model achieved excellent overall performance, 
            signifying near-perfect classification for most classes. All classes achieved 100% recall, meaning the model correctly identified all 
            instances of these diseases. The only notable area of weakness was observed in the Background class, where a few instances were 
            misclassified, likely due to visual similarities or inconsistencies during manual annotation.
        """)

    with col2:
        st.subheader("Box, Seg, and Cls Training Losses")
        losses = Image.open("resource/graphs/combined_loss_curves.png")
        losses = losses.resize((IMG_WIDTH, IMG_HEIGHT))
        st.image(losses, width='stretch')
        st.markdown(
            """
            The Bounding Box, Class, and Segmentation Training Loss curves in collectively demonstrate the model’s progressive improvement in 
            spatial localization, class discrimination, and pixel-level mask prediction. All three losses show steep early declines during the 
            first 10–15 epochs, indicating rapid initial learning, followed by smooth convergence and stabilization at low magnitudes after 
            approximately epoch 30.
        """)

        st.subheader("Mask mAP50 Results")
        mask = Image.open("resource/graphs/map50_mask_results.png")
        mask = mask.resize((IMG_WIDTH, IMG_HEIGHT))
        st.image(mask, width='stretch')
        st.markdown(
            """
            The mAP50 (Mask) curve starts at about 0.90, goes through a few small fluctuations in the early epochs, and then swiftly rises to 0.99, 
            where it stays almost constant for the remainder of the training period. The model successfully learnt high-quality mask segmentation, 
            attaining accurate object delineation and constant accuracy across all epochs, as evidenced by this near-perfect performance.
        """)
        
        st.subheader("Model Testing Results")
        testing = Image.open("resource/graphs/test_images.jpg")
        testing = testing.resize((IMG_WIDTH, IMG_HEIGHT))
        st.image(testing, width='stretch')
        st.markdown(
            """
            The performance of the YOLOv12 segmentation model on unseen test images demonstrates how well it detects and applies segmentation masks 
            on mango leaves with high confidence scores. The model is able to perform well even with the differences in image orientation.
        """)
# Detection page
if st.session_state.page_selection == "detection":
    st.header("👁 Detection")
    detection()