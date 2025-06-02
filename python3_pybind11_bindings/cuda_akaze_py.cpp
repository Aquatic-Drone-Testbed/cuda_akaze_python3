#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/core/core.hpp>
#include <AKAZE.h>

namespace py = pybind11;
using namespace libAKAZECU;

// Convert cv::Mat to py::array (NumPy)
py::array mat_to_numpy(const cv::Mat& mat) {
    if (!mat.isContinuous()) {
        throw std::runtime_error("Input cv::Mat is not continuous.");
    }

    std::vector<size_t> shape = { (size_t)mat.rows, (size_t)mat.cols };
    std::vector<size_t> strides = { (size_t)mat.step[0], (size_t)mat.elemSize() };

    // Handle different cv::Mat types
    std::string fmt;
    if (mat.type() == CV_8UC1) {
        fmt = py::format_descriptor<std::uint8_t>::format();
    } else if (mat.type() == CV_32FC1) {
        fmt = py::format_descriptor<float>::format();
    } else if (mat.type() == CV_64FC1) {
        fmt = py::format_descriptor<double>::format();
    } else if (mat.type() == CV_8UC3) {
        fmt = py::format_descriptor<std::uint8_t>::format(); // For RGB (3 channels)
        shape.push_back(3); // Add 3rd dimension for color channels
        strides.push_back(mat.step[1]); // Stride for 3rd dimension
    } else {
        throw std::runtime_error("Unsupported cv::Mat type.");
    }

    return py::array(py::buffer_info(
        mat.data,           // Pointer to data
        mat.elemSize(),     // Size of one item
        fmt,                 // Format
        2 + (mat.channels() > 1), // Number of dimensions (3 for color images)
        shape,               // Shape
        strides              // Strides
    ));
}

// Convert py::array to cv::Mat
cv::Mat numpy_to_mat(const py::array& array) {
    py::buffer_info info = array.request();

    // Check the number of dimensions (only 2D and 3D arrays supported)
    if (info.ndim != 2 && info.ndim != 3) {
        throw std::runtime_error("Only 2D and 3D arrays are supported for cv::Mat");
    }

    // Detect the type of data (uint8, float, double, etc.)
    int type = CV_8UC1; // Default type for grayscale images (CV_8UC1)
    if (info.format == py::format_descriptor<std::uint8_t>::format()) {
        type = CV_8UC1;
    } else if (info.format == py::format_descriptor<float>::format()) {
        type = CV_32FC1;
    } else if (info.format == py::format_descriptor<double>::format()) {
        type = CV_64FC1;
    } else {
        throw std::runtime_error("Unsupported NumPy array format.");
    }

    int rows = (int)info.shape[0];
    int cols = (int)info.shape[1];

    // Handle multi-channel images (e.g., color images)
    if (info.ndim == 3) {
        if (info.shape[2] == 3) {
            type = CV_8UC3;  // RGB image
        } else {
            throw std::runtime_error("Only 3-channel images are supported.");
        }
    }

    return cv::Mat(rows, cols, type, info.ptr);
}

PYBIND11_MODULE(cuda_akaze_py, m) {
    py::class_<AKAZEOptions>(m, "AKAZEOptions")
        .def(py::init<>())
        .def("setWidth", &AKAZEOptions::setWidth)
        .def("setHeight", &AKAZEOptions::setHeight);

    py::class_<AKAZE>(m, "AKAZE")
        .def(py::init<const AKAZEOptions&>())
        .def("Create_Nonlinear_Scale_Space", [](AKAZE& self, py::array img_array) {
            cv::Mat img = numpy_to_mat(img_array);
            return self.Create_Nonlinear_Scale_Space(img);
        })
        .def("Feature_Detection", [](AKAZE& self) {
            cv::Mat result = self.Feature_Detection_();
            return mat_to_numpy(result);
        })
        .def("Compute_Descriptors", [](AKAZE& self) {
            auto result = self.Compute_Descriptors_();  // This must return std::tuple<cv::Mat, cv::Mat>
            cv::Mat desc = std::get<0>(result);
            cv::Mat kpts = std::get<1>(result);
            return py::make_tuple(mat_to_numpy(desc), mat_to_numpy(kpts));
        });

    py::class_<Matcher>(m, "Matcher")
        .def(py::init<>())
        .def("BFMatch", [](Matcher& self, py::array desc_query, py::array desc_train) {
            cv::Mat dq = numpy_to_mat(desc_query);
            cv::Mat dt = numpy_to_mat(desc_train);
            cv::Mat result = self.bfmatch_(dq, dt);
            return mat_to_numpy(result);
        });
}
