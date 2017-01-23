# Description:
#   TODO(haeusser)
#   bazel build -c opt --copt=-mavx tensorflow/contrib/semisup

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "semisup",
    srcs = ["__init__.py",
            "python/semisup/semisup.py"
           ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],    
    deps = [
        #"//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:framework",
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
        #"//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:framework",
        "//tensorflow/contrib/slim:slim",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "stl10_tools",
    srcs = ["python/semisup/stl10_tools.py",
           ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],    
    deps = [
        #"//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:framework",
        "//tensorflow/contrib/slim:slim",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)


py_library(
    name = "svhn_tools",
    srcs = ["python/semisup/svhn_tools.py",
           ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],    
    deps = [
        #"//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:framework",
        "//tensorflow/contrib/slim:slim",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)


py_library(
    name = "synth_tools",
    srcs = ["python/semisup/synth_tools.py",
           ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],    
    deps = [
        #"//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:framework",
        "//tensorflow/contrib/slim:slim",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "mnist_train_eval",
    srcs = ["python/semisup/mnist_train_eval.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":semisup",
        ":mnist_tools",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        #"//tensorflow/python:framework",
    ],
)

py_binary(
    name = "train",
    srcs = ["python/semisup/train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":semisup",
        ":mnist_tools",
        ":svhn_tools",
        ":synth_tools",
        ":stl10_tools",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        #"//tensorflow/python:pywrap_tensorflow",
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
