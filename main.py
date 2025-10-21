import streamlit as st
import altair as alt
import os
import random
from PIL import Image

# page config
st.set_page_config(
    page_title="Mango Leaf Disease Image Instance Segmentation Using YOLOv12",
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
    st.title("Mango Leaf Disease Image Instance Segmentation Using YOLOv12 🥭")
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
    st.markdown("🤖 [Roboflow](https://universe.roboflow.com/artificial-intelligence-u9ca8/mango-leaf-image-detection-qvw37)")
    st.markdown("📔 [Google Colab Notebook](https://colab.research.google.com/drive/1xwaCdEhWPi_2sUwpqr9Mp2uCdcOyj45_#scrollTo=MQo1i9FqwRfd)")
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
    

# About page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")
    st.markdown("""
    This is a machine project for instance segmentation of mango leaf disease images for detection task using the YOLOv12 model. 
The image samples were sourced from the [MangoLeafBD Dataset](https://data.mendeley.com/datasets/hxsnvwty3r/1), which consists of 
different images of mango leaves diagnosed with and without diseases. Annotation and augmentation was done in Roboflow to enhance 
the model's performance. The trained model achieved a [insert mAP score], indicating how well it performed for the detection task.

This Streamlit website application is for demonstrating the outcome of the machine project.
                
### Pages
- 🍃 `Dataset` - Provides a brief description of the MangoLeafBD dataset use in the machine project and random, resized (640x640) image samples from different classes.
- ✏️ `Annotation` - Provides samples of annotated images using Roboflow.
- ⬆️ `Augmentation` - Details the train:validation:test ratio before and after augmentation and enumerates preprocessing steps and filters used in Roboflow.
- 🏋️‍♀️ `Training Results` - Shows the YOLOv12 model's training results in tabular and graphical format.
- 👁 `Detection` - Testing playground for users that allows them to upload images of mango leaves to assess the model's performance in unseen data.
    
""")
    
# Dataset page
if st.session_state.page_selection == "dataset":
    st.header("🍃 Dataset")
    st.markdown("""
    The [MangoLeafDB Dataset](https://data.mendeley.com/datasets/hxsnvwty3r/1) is a dataset of 4000 image samples of mango leaves belonging from eight different classes that 
are equally distributed: `Anthracnose`, `Bacterial Canker`, `Cutting Weevil`, `Die Back`, `Gall Midge`, `Healthy`, `Powdery Mildew`, and `Sooty Mould`. The methodology of the project excluded
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