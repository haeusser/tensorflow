# Description:
#   TODO(haeusser)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "semisup",
    srcs = ["__init__.py",
           ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],    
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/slim:slim",
    ],
)

py_library(
    name = "mnist_tools",
    srcs = ["python/semisup/mnist_tools.py",
           ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],    
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/slim:slim",
        "//third_party/py/numpy",
    ],
)



filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
