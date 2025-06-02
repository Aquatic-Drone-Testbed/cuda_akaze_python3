from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "cuda_akaze_py",
        ["cuda_akaze_py.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/home/seamate1/Documents/cuda_akaze/src/lib",
            "/usr/include/opencv4",
            "/usr/local/cuda/include",
        ],
        library_dirs=[
            "/home/seamate1/Documents/cuda_akaze/build/lib",
            "/usr/local/cuda/lib64",
            "/usr/lib",  # or your system's OpenCV lib path
        ],
        libraries=[
            "AKAZE",
            "AKAZE_CUDA",
            "cudart",
            "opencv_core",
            "opencv_imgcodecs",
            "opencv_imgproc",
            "opencv_features2d",
            "opencv_calib3d"  # 🔥 Needed for findHomography
        ],
        extra_compile_args=["-std=c++14", "-fPIC", "-O3"],
        extra_link_args=["-Wl,--no-as-needed"]
    )
]

setup(
    name="cuda_akaze_py",
    version="0.1",
    ext_modules=ext_modules,
)

