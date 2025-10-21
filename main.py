import streamlit as st
import altair as alt

# page config
st.set_page_conf(
    page_title="Mango Leaf Disease Image Instance Segmentation Using YOLOv12"
    page_icon="🥭"
    layout="wide"
    initial_sidebar_state="expanded"
)
alt.themes.enable("light")

if 'page_selection' not in st.session_state:
    st.session_state.page_selection="about"

def set_page_selection(page):
    st.session_state.page_selection = page

# sidebar
with st.sidebar:
    st.title("Mango Leaf Disease Image Instance Segmentation Using YOLOv12")
    st.subheader("Pages")
    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = "about"
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = "dataset"
    if st.button("Annotation", use_container_width=True, on_click=set_page_selection, args=('annotation',)):
        st.session_state.page_selection = "annotation"
    if st.button("Augmentation", use_container_width=True, on_click=set_page_selection, args=('augmentation',)):
        st.session_state.page_selection = "augmentation"
    if st.button("Training Results", use_container_width=True, on_click=set_page_selection, args=('training results',)):
        st.session_state.page_selection = "training results"
    if st.button("Detection", use_container_width=True, on_click=set_page_selection, args=('detection',)):
        st.session_state.page_selection = "detection"

    st.subheader("Contributors")
    st.markdown("* Butial, A.P.\n* Pagdanganan, J.M.\n* Ricafranca, S.\n* Tan, J. R.\n* Dr. Comia, L.")

    st.subheader("Relevant Links")
    st.markdown("📊 [Dataset](https://data.mendeley.com/datasets/hxsnvwty3r/1)")
    st.markdown("📔 [Google Colab Notebook](https://colab.research.google.com/drive/1xwaCdEhWPi_2sUwpqr9Mp2uCdcOyj45_#scrollTo=MQo1i9FqwRfd)")
    st.markdown("🗄️ [GitHub Repository](https://github.com/VannCodes/mango_streamlit.git)")

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
    
    - `Dataset` - Provides a brief description of the MangoLeafBD dataset use in the machine project and image samples from different classes.
    - `Annotation` - Provides samples of annotated images using Roboflow
    - `Augmentation` - Details the train:validation:test ratio before and after augmentation and enumerates preprocessing steps and filters used in Roboflow.
    - `Training Results` - Shows the YOLOv12 model's training results in tabular and graphical format.
    - `Detection` - Testing playground for users that allows them to upload images of mango leaves to assess the model's performance in unseen data.
    
""")