import streamlit as st
import subprocess
import cv2
import os
import multiprocessing

# ------------------- Setup Paths -------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
INPUT_FOLDER = os.path.join(BASE_DIR, "inputs")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
INPUT_IMAGE_PATH = os.path.join(INPUT_FOLDER, "input.jpg")

OUTPUT_HPF_SEQ = os.path.join(OUTPUT_FOLDER, "HPF_SEQ_outputRes.png")
OUTPUT_HPF_OMP = os.path.join(OUTPUT_FOLDER, "HPF_OMP_outputRes.png")
OUTPUT_HPF_MPI = os.path.join(OUTPUT_FOLDER, "HPF_MPI_outputRes.png")
TIME_PATH = os.path.join(OUTPUT_FOLDER, "time.txt")

EXE_HPF_SEQ = os.path.join(BASE_DIR, "HighPassFilter_seq.exe")
EXE_HPF_OMP = os.path.join(BASE_DIR, "HighPassFilter_omp.exe")
EXE_HPF_MPI = os.path.join(BASE_DIR, "HighPassFilter_mpi.exe")

# ------------------- UI Header -------------------
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🧠 ParaVision : High-Pass Filter App (C++ Accelerated)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an image, select a mode, and apply high-pass filtering using optimized C++ implementations.</p>", unsafe_allow_html=True)

# ------------------- File Upload -------------------
uploaded = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

# ------------------- Filter Mode Selection -------------------
st.divider()
st.subheader("⚙️ Choose Filter Configuration")
hpf_mode = st.selectbox("Processing Mode", ["Sequential", "OpenMP", "MPI"])

# ------------------- Parameter Inputs -------------------
if hpf_mode == "MPI":
    num_units = st.number_input("MPI: Number of Processes", min_value=1, max_value=multiprocessing.cpu_count(), value=2)
    kernel_size = st.number_input("Kernel Size (odd only)", min_value=3, max_value=11, value=3, step=2)

elif hpf_mode == "OpenMP":
    num_units = st.number_input("OpenMP: Number of Threads", min_value=1, max_value=multiprocessing.cpu_count(), value=2)
    kernel_size = st.number_input("Kernel Size (odd only)", min_value=3, max_value=11, value=3, step=2)

else:
    kernel_size = st.number_input("Kernel Size (odd only)", min_value=3, max_value=11, value=3, step=2)

st.divider()

# ------------------- Action Button -------------------
apply_filter = st.button("🚀 Apply High-Pass Filter", use_container_width=True)

# ------------------- Execution -------------------
if uploaded:
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    with open(INPUT_IMAGE_PATH, "wb") as f:
        f.write(uploaded.read())
        f.flush()
        os.fsync(f.fileno())

    st.image(INPUT_IMAGE_PATH, caption="🖼️ Original Image", use_container_width=True)

    if apply_filter:
        exe_map = {
            "Sequential": EXE_HPF_SEQ,
            "OpenMP": EXE_HPF_OMP,
            "MPI": EXE_HPF_MPI
        }
        output_map = {
            "Sequential": OUTPUT_HPF_SEQ,
            "OpenMP": OUTPUT_HPF_OMP,
            "MPI": OUTPUT_HPF_MPI
        }

        exe_path = exe_map[hpf_mode]
        output_path = output_map[hpf_mode]

        st.info(f"🔧 Running {hpf_mode} High-Pass Filter...")

        if hpf_mode == "MPI":
            subprocess.run(["mpiexec", "-n", str(num_units), exe_path,str(kernel_size)], cwd=BASE_DIR, stderr=subprocess.DEVNULL)
        elif hpf_mode == "OpenMP":
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(num_units)
            subprocess.run([exe_path,str(kernel_size)], cwd=BASE_DIR, env=env, stderr=subprocess.DEVNULL)
        else:
            subprocess.run([exe_path, str(kernel_size)], cwd=BASE_DIR, stderr=subprocess.DEVNULL)

        if os.path.exists(output_path):
            img = cv2.imread(output_path)
            col1, col2 = st.columns(2)
            col1.image(INPUT_IMAGE_PATH, caption="🖼️ Original Image", use_container_width=True)
            col2.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"🎯 Output Image ({hpf_mode})",
                       use_container_width=True)

            if os.path.exists(TIME_PATH):
                with open(TIME_PATH, "r") as f:
                    for line in f.readlines():
                        st.success(line.strip())
        else:
            st.error("⚠️ Output image not found. Please check the executable or input.")
