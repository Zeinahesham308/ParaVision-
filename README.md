# 🧠 ParaVision - High-Pass Image Filter (C++ Accelerated + Streamlit GUI)

ParaVision is a hybrid C++ + Python project designed to demonstrate the performance of different high-pass filtering implementations — **Sequential**, **OpenMP**, and **MPI** — on grayscale images.

***Python is used for GUI only ***

It includes:
- ⚙️ Optimized C++ image filters (Sequential / OpenMP / MPI)
- 🌐 A modern Python Streamlit GUI
- 📸 Upload + Visualize input/output image side-by-side
- 📊 Execution time reporting for comparison

---

## 📁 Project Structure
ParaVision-/
│
├── Streamlit_HPF/ # Python GUI <br>
│ ├── venv/ # Virtual environment <br>
│ └── main.py # Streamlit app <br>
│<br>
├── inputs/ # Input images <br>
├── outputs/ # Output results + time.txt <br>
│<br>
├── HighPassFilter_seq.cpp # Sequential C++ <br>
├── HighPassFilter_omp.cpp # OpenMP C++ <br>
├── HighPassFilter_mpi.cpp # MPI C++ <br>
│<br>
├── HighPassFilter_30.sln # Visual Studio Solution <br>
├── HighPassFilter_30.vcxproj # VS Project File <br>
├── .gitignore <br>
└── README.md <br>

<br>
---

## ⚙️ Build Instructions (C++)

### 🖥️ 1. Sequential

#### 🧱 Prerequisites:
- Visual Studio with Developer Command Prompt (`x64 Native Tools for VS`)
- OpenCV (Windows build)

#### 🛠 Build:
```bash
cd path\to\ParaVision-
cl /EHsc /I C:\path\to\opencv\build\include HighPassFilter_seq.cpp ^
 /link /LIBPATH:C:\path\to\opencv\build\x64\vc16\lib opencv_world4110.lib 
```

🚀 2. OpenMP
🧱 Prerequisites:
MSYS2 + OpenCV installed via pacman

pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-opencv

🛠 Build:
bash
```
cd /c/path/to/ParaVision-
g++ -fopenmp HighPassFilter_omp.cpp -std=c++11 -o HighPassFilter_omp.exe \
 -I/mingw64/include/opencv4 -L/mingw64/lib \
 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
```



🧪 3. MPI
🧱 Prerequisites:
Microsoft MPI SDK

Visual Studio Developer Command Prompt

🛠 Build:
bash
```
cd path\to\ParaVision-
cl /EHsc /openmp ^
 /I"C:\msys64\mingw64\include\opencv4" ^
 /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" ^
 HighPassFilter_mpi.cpp ^
 /link ^
 /LIBPATH:"C:\path\to\opencv\build\x64\vc16\lib" ^
 /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" ^
 opencv_world4110.lib msmpi.lib
```

🌐 Python GUI (Streamlit)
🧱 Prerequisites:
Python 3.7+
pip
🛠 Setup:
bash
```
cd Streamlit_HPF
python -m venv venv
venv\Scripts\activate     # Windows
pip install streamlit opencv-python
```
▶️ Run:
streamlit run main.py

Upload your image via the browser

Select execution mode: Sequential / OpenMP / MPI

If required, adjust:

Kernel size (Sequential)

Thread count (OpenMP)

Process count (MPI)

View original and filtered output side-by-side

Execution details appear below


📌 Notes
Default kernel size: 3x3

Kernel = center pixel = size² - 1, rest -1

Grayscale conversion is handled before filtering

outputs/time.txt logs execution time and parameters
