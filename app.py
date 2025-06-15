import os
import cv2
import time
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import streamlit as st

# --- PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title="Handgebaar Detector Pro",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STIJLEN ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1 {
        color: #2a3f5f;
    }
    .stProgress > div > div > div > div {
        background-color: #4a8cff;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALISATIE ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- MODEL LADEN ---
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Fout bij het laden van het model: {str(e)}")
        return YOLO("yolov8n.pt")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "runs", "detect", "train27", "weights", "best.pt")

# --- UI COMPONENTEN ---
def show_upload_section():
    st.subheader("📁 Afbeelding Upload")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Kies een afbeelding", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="uploader"
        )
        
    if uploaded_file:
        with st.spinner("Afbeelding verwerken..."):
            try:
                image = Image.open(uploaded_file).convert("RGB")
                return image
            except Exception as e:
                st.error(f"Fout bij het openen van de afbeelding: {str(e)}")
                return None

def show_camera_section():
    st.subheader("📷 Live Camera")
    if st.button("🎥 Start Camera", key="start_cam"):
        st.session_state.camera_active = True
        
    if st.button("⏹️ Stop Camera", key="stop_cam"):
        st.session_state.camera_active = False
        if "cap" in st.session_state:
            st.session_state.cap.release()
            
    if getattr(st.session_state, "camera_active", False):
        cam_placeholder = st.empty()
        status_text = st.empty()
        
        if "cap" not in st.session_state:
            st.session_state.cap = cv2.VideoCapture(0)
            
        while st.session_state.camera_active:
            ret, frame = st.session_state.cap.read()
            if not ret:
                status_text.error("Kan camerabeeld niet ontvangen")
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(frame, channels="RGB", use_column_width=True)
            
            if st.button("📸 Maak Foto"):
                return Image.fromarray(frame)
                
            time.sleep(0.1)
            
    return None

def process_image(image, model, conf_threshold):
    with st.spinner("Bezig met detectie..."):
        try:
            results = model(image, conf=conf_threshold, verbose=False)
            
            # Plot resultaten met aangepaste grootte
            annotated = results[0].plot()
            annotated_pil = Image.fromarray(annotated[..., ::-1])  # BGR naar RGB
            
            # Schaal afbeelding voor betere weergave
            max_width = 800
            width_percent = (max_width / float(annotated_pil.size[0]))
            height_size = int((float(annotated_pil.size[1]) * float(width_percent)))
            resized = annotated_pil.resize((max_width, height_size), Image.LANCZOS)
            
            return results, resized
            
        except Exception as e:
            st.error(f"Detectie mislukt: {str(e)}")
            return None, None

def show_results(results, image, model):  # Voeg model parameter toe
    if results and image:
        st.subheader("🔍 Detectie Resultaten")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption="Gedetecteerde gebaren", use_column_width=True)
            
            # Download knop
            buf = BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Resultaat",
                data=buf.getvalue(),
                file_name="detectie_resultaat.png",
                mime="image/png"
            )
            
        with col2:
            st.subheader("Statistieken")
            
            if results[0].boxes:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id]  # Nu correct gedefinieerd
                    conf = float(box.conf[0])
                    
                    st.metric(
                        label=f"{label.upper()} Confidence",
                        value=f"{conf:.2%}",
                        help=f"Detectie zekerheid voor {label}"
                    )
                    
                    st.progress(int(conf * 100))
                    
                # Voeg toe aan geschiedenis
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                for box in results[0].boxes:
                    st.session_state.history.append({
                        "label": model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "timestamp": timestamp
                    })
            else:
                st.warning("Geen gebaren gedetecteerd")

def show_history():
    st.subheader("📊 Detectie Geschiedenis")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        # Toon tabel
        st.dataframe(
            df.sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Download knop
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporteer naar CSV",
            data=csv,
            file_name="detectie_geschiedenis.csv",
            mime="text/csv"
        )
        
        # Statistieken
        st.subheader("📈 Samenvatting")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Totaal detecties",
                len(df),
                help="Aantal keren dat gebaren zijn gedetecteerd"
            )
            
        with col2:
            most_common = df['label'].mode()[0] if not df.empty else "Geen"
            st.metric(
                "Meest voorkomend gebaar",
                most_common.upper()
            )
            
        if st.button("🗑️ Wissen Geschiedenis", type="primary"):
            st.session_state.history = []
            st.success("Geschiedenis gewist!")
    else:
        st.info("Nog geen detecties uitgevoerd")

# --- HOOFD APPLICATIE ---
def main():
    # --- SIDEBAR ---
    with st.sidebar:
        # Logo sectie
        logo_path = "/Users/hacakir/Downloads/HogeschoolRotterdam-655x500.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=150)
        else:
            st.image("https://via.placeholder.com/150x50?text=LOGO", width=150)
        
        model_choice = st.radio(
            "Model selectie",
            ["Standaard YOLOv8n", "Aangepast Model"],
            index=1,
            help="Kies tussen het standaard YOLO model of je eigen getrainde model"
        )
        
        conf_threshold = st.slider(
            "Confidence drempel",
            0.1, 1.0, 0.5, 0.05,
            help="Minimale zekerheid voor detectie"
        )
        
        filter_input = st.text_input(
            "Filter gebaren (komma gescheiden)",
            help="Bijvoorbeeld: a,b,c"
        )
        
        allowed_classes = [cls.strip().lower() for cls in filter_input.split(",")] if filter_input else []
        
        st.markdown("---")
        st.markdown("**Applicatie Info**")
        st.markdown("""
        - Versie: 1.0.0
        - Aangemaakt met YOLOv8
        - Streamlit interface
        """)
    
    # --- MODEL LADEN ---
    if model_choice == "Standaard YOLOv8n":
        model = load_model("yolov8n.pt")
        st.sidebar.info("Standaard YOLO model geladen")
    else:
        model = load_model(model_path) if os.path.exists(model_path) else load_model("yolov8n.pt")
        st.sidebar.success("Aangepast model geladen!" if os.path.exists(model_path) else "Aangepast model niet gevonden, standaard model gebruikt")
    
    # --- HOOFD CONTENT ---
    st.title("✋ Handgebaar Detector Pro")
    st.markdown("Detecteer handgebaren in real-time of via geüploade afbeeldingen")
    
    tab1, tab2, tab3 = st.tabs(["🏠 Home", "🔍 Detectie", "📊 Geschiedenis"])
    
    with tab1:
        st.subheader("Welkom bij de Handgebaar Detector!")
        st.image("https://via.placeholder.com/800x300?text=HANDGEBAAR+DETECTIE", use_column_width=True)
        
        st.markdown("""
        ### Hoe werkt het?
        1. Kies een model in de zijbalk
        2. Upload een afbeelding of gebruik je camera
        3. Bekijk de gedetecteerde gebaren
        4. Analyseer de resultaten in het geschiedenisoverzicht
        
        ### Ondersteunde gebaren
        - Gebaar A: ...
        - Gebaar B: ...
        - Gebaar C: ...
        """)
        
        st.markdown("---")
        st.markdown("""
        *Technologieën gebruikt in deze applicatie:*
        - YOLOv8 voor objectdetectie
        - Streamlit voor de gebruikersinterface
        - OpenCV voor beeldverwerking
        """)
    
    with tab2:
        st.header("Gebaar Detectie")
        
        input_method = st.radio(
            "Invoermethode",
            ["Afbeelding Upload", "Live Camera"],
            horizontal=True
        )
        
        image = None
        if input_method == "Afbeelding Upload":
            image = show_upload_section()
        else:
            image = show_camera_section()
            
        if image:
            results, processed_img = process_image(image, model, conf_threshold)
            show_results(results, processed_img, model)  # Vergeet niet de model parameter toe te voegen
    
    with tab3:
        show_history()

if __name__ == "__main__":
    main()

    
##streamlit run app.py
##test for pull