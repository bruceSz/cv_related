package(default_visibility = ["//visibility:public"])

cc_library(
    name = "opencv_headers",
    hdrs = glob(["include/**/*.h"]) + glob(["include/**/*.hpp"]),
    strip_include_prefix = "include/opencv4",
)


cc_library(
    name = "opencv_srcs_linux",
    # Order is important, never change the order
    srcs = [
        "lib_linux/libopencv_ccalib.a",
        "lib_linux/libopencv_calib3d.a",
        "lib_linux/libopencv_features2d.a",
        "lib_linux/libopencv_flann.a",
        "lib_linux/libopencv_highgui.a",
        "lib_linux/libopencv_imgcodecs.a",
        "lib_linux/libopencv_imgproc.a",
        "lib_linux/libopencv_objdetect.a",
        # comment here so order is not changed
        "lib_linux/libopencv_core.a",
        "lib_linux/opencv4/3rdparty/liblibopenjp2.a",
        "lib_linux/opencv4/3rdparty/liblibjpeg-turbo.a",
        "lib_linux/opencv4/3rdparty/liblibpng.a",
        "lib_linux/opencv4/3rdparty/liblibtiff.a",
        "lib_linux/opencv4/3rdparty/libzlib.a",
    ],
)

cc_library(
    name = "opencv",
    deps = [
        ":opencv_headers",
        ":opencv_srcs_linux",
    ] 
)

alias(
    name = "opencv_srcs",
    actual = ":opencv",
)
