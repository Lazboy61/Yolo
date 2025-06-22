import os
import cv2
import time
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import streamlit as st
import random
import json

# --- PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title="Handgesture Detector Pro",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STIJLEN ---
st.markdown("""
<style>
    /* Verbeterde home page stijlen */
    .home-content {
        line-height: 1.6;
    }
    .home-content h3 {
        margin-top: 1.5rem;
        color: #2a3f5f;
    }
    .home-content ul {
        padding-left: 1.5rem;
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
        st.error(f"Error loading the model: {str(e)}")
        return YOLO("yolov8n.pt")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "runs", "detect", "train27", "weights", "best.pt")

# --- UI COMPONENTEN ---
def show_upload_section():
    st.subheader("Picture Upload")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a picture", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="uploader"
        )
        
    if uploaded_file:
        with st.spinner("Processing image..."):
            try:
                image = Image.open(uploaded_file).convert("RGB")
                return image
            except Exception as e:
                st.error(f"Error opening image: {str(e)}")
                return None

def show_camera_section():
    st.subheader("Live Camera Detection")
    
    # Configuration section
    with st.expander("⚙️ Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            detection_interval = st.slider("Detection interval (seconds)", 1, 10, 5)
            confidence_threshold = st.slider("Minimum confidence", 0.1, 1.0, 0.5, 0.05)
        with col2:
            show_live_feed = st.checkbox("Show live feed", True)
            show_annotations = st.checkbox("Show annotations", True)

    # Start/Stop controls
    start_col, stop_col, _ = st.columns([1, 1, 2])
    with start_col:
        if st.button("🎥 Start Detection", key="start_detection"):
            # Ensure previous camera session is properly closed
            if "cap" in st.session_state:
                st.session_state.cap.release()
                del st.session_state.cap
                
            st.session_state.camera_active = True
            st.session_state.last_detection_time = 0
            st.session_state.detected_gestures = []
            st.session_state.current_detection = None
    
    with stop_col:
        if st.button("⏹Stop Detection", key="stop_detection"):
            st.session_state.camera_active = False
            if "cap" in st.session_state:
                st.session_state.cap.release()
                del st.session_state.cap  # Remove camera reference

    # Results display
    result_placeholder = st.empty()
    history_placeholder = st.container()
    
    # Camera feed and detection
    if getattr(st.session_state, "camera_active", False):
        feed_placeholder = st.empty()
        model = load_model(model_path) if os.path.exists(model_path) else load_model("yolov8n.pt")
        
        # Initialize camera
        if "cap" not in st.session_state:
            st.session_state.cap = cv2.VideoCapture(0)
            if not st.session_state.cap.isOpened():
                st.error("Cannot open camera")
                return None
        
        cap = st.session_state.cap
        
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot receive camera feed")
                break
            
            current_time = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Show live feed if requested
            if show_live_feed:
                display_frame = frame_rgb.copy()
                
                # Add current detection as annotation
                if show_annotations and st.session_state.current_detection:
                    label = st.session_state.current_detection['gesture']
                    confidence = st.session_state.current_detection['confidence']
                    cv2.putText(display_frame, 
                               f"{label} ({confidence:.0%})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                               1, (0, 255, 0), 2)
                
                feed_placeholder.image(display_frame, channels="RGB", use_container_width=True)
            
            # Detection logic
            if current_time - st.session_state.get("last_detection_time", 0) >= detection_interval:
                results = model(frame_rgb, verbose=False)
                
                if results[0].boxes:
                    # Find most confident detection above threshold
                    boxes = results[0].boxes
                    valid_indices = [i for i, conf in enumerate(boxes.conf) if conf > confidence_threshold]
                    
                    if valid_indices:
                        max_conf_idx = max(valid_indices, key=lambda i: boxes.conf[i])
                        class_id = int(boxes.cls[max_conf_idx])
                        label = model.names[class_id]
                        confidence = float(boxes.conf[max_conf_idx])
                        
                        # Update current detection
                        st.session_state.current_detection = {
                            "gesture": label,
                            "confidence": confidence,
                            "time": current_time
                        }
                        
                        # Add to history
                        if "detected_gestures" not in st.session_state:
                            st.session_state.detected_gestures = []
                        st.session_state.detected_gestures.append({
                            "gesture": label,
                            "confidence": confidence,
                            "timestamp": time.strftime("%H:%M:%S")
                        })
                        
                        # Update result display
                        result_placeholder.success(
                            f" {time.strftime('%H:%M:%S')} - "
                            f"Gesture: **{label.upper()}** "
                            f"(confidence: {confidence:.2%})"
                        )
                    else:
                        feedback_msg = """
                        **🔍 No gesture detected - Tips:**
                        - Ensure your hand is clearly visible
                        - Try moving closer to the camera
                        - Ensure good lighting
                        - Hold your hand steady
                        - Try a different gesture
                        """
                        result_placeholder.markdown(feedback_msg)
                else:
                    result_placeholder.info("⏳ No gestures detected")
                
                st.session_state.last_detection_time = current_time
            
            time.sleep(0.1)  # Reduce CPU usage
    
    # Show history when camera inactive
    if "detected_gestures" in st.session_state and st.session_state.detected_gestures:
        with history_placeholder:
            st.subheader("📜 Detection History")
            for i, detection in enumerate(reversed(st.session_state.detected_gestures), 1):
                st.write(
                    f"{i}.  {detection['timestamp']} - "
                    f"**{detection['gesture'].upper()}** "
                    f"(confidence: {detection['confidence']:.2%})"
                )
                
            if st.button("Clear History"):
                st.session_state.detected_gestures = []
                st.session_state.current_detection = None
                st.rerun()

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
        st.subheader("Detectie Resultaten")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption="Gedetecteerde gebaren", use_column_width=True)
            
            # Download knop
            buf = BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="Download Resultaat",
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
    st.subheader("📊 Detection History")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        # Display table
        st.dataframe(
            df.sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export to CSV",
            data=csv,
            file_name="detection_history.csv",
            mime="text/csv"
        )
        
        # Statistics
        st.subheader("📈 Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Total Detections",
                len(df),
                help="Number of times gestures were detected"
            )
            
        with col2:
            most_common = df['label'].mode()[0] if not df.empty else "None"
            st.metric(
                "Most Common Gesture",
                most_common.upper()
            )
            
        if st.button("🗑️ Clear History", type="primary"):
            st.session_state.history = []
            st.success("History cleared!")
    else:
        st.info("No detections performed yet")
# --- HOOFD APPLICATIE ---
def main():
    # --- SIDEBAR ---
    with st.sidebar:
        # Logo section
        logo_path = "/Users/hacakir/Downloads/HogeschoolRotterdam-655x500.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=150)
        else:
            st.image("https://via.placeholder.com/150x50?text=LOGO", width=150)
        
        model_choice = st.radio(
            "Model Selection",
            ["Standard YOLOv8n", "Custom Model"],
            index=1,
            help="Choose between standard YOLO model or your trained model"
        )
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            0.1, 1.0, 0.5, 0.05,
            help="Minimum detection confidence level"
        )
        
        filter_input = st.text_input(
            "Filter gestures (comma separated)",
            help="Example: a,b,c"
        )
        
        allowed_classes = [cls.strip().lower() for cls in filter_input.split(",")] if filter_input else []

        if "detected_gestures" in st.session_state:
            st.markdown("---")
            st.markdown("**Session Stats**")
            total = len(st.session_state.detected_gestures)
            accuracy = np.mean([g['confidence'] for g in st.session_state.detected_gestures])
            
            st.metric("Total Detections", total)
            st.metric("Average Confidence", f"{accuracy:.1%}")
            
            if total > 10:
                st.progress(min(accuracy, 0.99))
                st.caption(f"Pro Tip: Try letters {random.sample(['A','B','L','Y'], 2)} for best results")
            
        st.markdown("---")
        st.markdown("**Application Info**")
        st.markdown("""
        - Version: 1.0.0
        - Built with YOLOv8
        - Streamlit interface
        """)
    # --- MODEL LADEN ---
    if model_choice == "Standaard YOLOv8n":
        model = load_model("yolov8n.pt")
        st.sidebar.info("Standaard YOLO model geladen")
    else:
        model = load_model(model_path) if os.path.exists(model_path) else load_model("yolov8n.pt")
        st.sidebar.success("Aangepast model geladen!" if os.path.exists(model_path) else "Aangepast model niet gevonden, standaard model gebruikt")
        # --- MAIN CONTENT ---
    st.title("✋ Hand Gesture Detector Pro")
    st.markdown("Detect hand gestures in real-time or via uploaded images")
    
    tab1, tab2, tab3 = st.tabs(["🏠 Home", "🔍 Detection", "📊 History"])
    
    with tab1:
        st.markdown("""
        ## ✋ Welcome to Hand Gesture Detector Pro
        
        **Detect and recognize hand gestures in real-time** using advanced YOLOv8 object detection.
        """)
        
        st.markdown("""
        ### 🚀 How It Works
        
        1. **Select your model** - Choose between standard YOLO model or your custom trained model
        2. **Start detection** - Use live camera or upload an image
        3. **View results** - See detected gestures with confidence scores
        4. **Analyze** - Review detection history for patterns
        
        ### 🔍 Key Features
        
        - Real-time hand gesture detection via webcam
        - Image upload for analysis
        - History of previous detections
        - Adjustable confidence thresholds
        - Interval-based detection (every X seconds)
        
        ### 🛠️ Technologies
        
        - **YOLOv8** - For fast and accurate object detection
        - **Streamlit** - For user interface
        - **OpenCV** - For image processing
        - **Python** - Backend logic
        
        ### 📌 Usage Tips
        
        - Ensure good lighting when using camera
        - Start with confidence threshold of 0.5 and adjust as needed
        - Use custom model for best results with your specific gestures
        """)
        
        with st.expander("🚀 Quick Start Guide (60 seconds)"):
            video_path = "/Users/hacakir/Desktop/Schermopname 2025-06-22 om 21.34.42.mov"
            with open(video_path, "rb") as file:
                video_bytes = file.read()

            st.video(video_bytes)
            st.markdown("""
            1. **Camera Setup**  
            - Ensure good lighting 💡  
            - Position hands 30-50cm from webcam  
            2. **Gesture Tips**  
            - Hold each letter for 2 seconds ✋→🅰️  
            - Avoid fast movements 🐢 > 🐇  
            3. **Troubleshooting**  
            - Refresh page if camera freezes ♻️  
            - Lower confidence threshold if needed 📉  
            """)

        
    with tab2:
        st.header("Gesture Detection")
        
        input_method = st.radio(
            "Input Method",
            ["Image Upload", "Live Camera"],
            horizontal=True
        )
        
        image = None
        if input_method == "Image Upload":
            image = show_upload_section()
        else:
            image = show_camera_section()
            
        if image:
            results, processed_img = process_image(image, model, conf_threshold)
            show_results(results, processed_img, model)
    
    with tab3:
        show_history()
        
        # Export section
        st.markdown("---")
        st.subheader("📤 Export Results")
        
        # Fun file name generator
        funny_names = ["MyAwesomeGestures", "HandSignsData", "GestureParty", "AI_Read_My_Hands"]
        default_name = random.choice(funny_names)
        
        export_col1, export_col2 = st.columns([3, 1])
        with export_col1:
            file_name = st.text_input("File name", default_name)
        with export_col2:
            file_format = st.selectbox("Format", ["CSV", "JSON", "TXT"])
        
        # Export button with fun styling
        if st.button("🚀 Blast Off My Data!", help="Export your gesture history"):
            if not st.session_state.history:
                st.warning("Nothing to export yet!")
            else:
                export_data = convert_to_format(st.session_state.history, file_format)
                st.download_button(
                    label="👇 Download Magic",
                    data=export_data,
                    file_name=f"{file_name}.{file_format.lower()}",
                    mime="text/csv" if file_format == "CSV" else "text/plain"
                )

def convert_to_format(data, format):
    """Helper function to convert data to different formats"""
    if format == "CSV":
        return pd.DataFrame(data).to_csv(index=False)
    elif format == "JSON":
        return json.dumps(data, indent=2)
    else:  # TXT
        lines = []
        for i, item in enumerate(data, 1):
            lines.append(f"{i}. {item['timestamp']} - {item['label']} ({item['confidence']:.2%})")
            if i % 5 == 0:  # Add funny comment every 5 items
                lines.append(f"   🤖 Model says: '{random_funny_comment()}'")
        return "\n".join(lines)

def random_funny_comment():
    comments = [
        "I see you're practicing your A-B-Cs!",
        "My robot heart loves these gestures!",
        "Is this sign language or secret messages?",
        "Beep boop - nice hand moves!",
        "I bet you could text with these signs!",
        "Future of communication right here!"
    ]
    return random.choice(comments)

if __name__ == "__main__":
    main()

    
##streamlit run app.py
##test for pull