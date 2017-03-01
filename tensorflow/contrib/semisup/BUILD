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
            "python/semisup/data_dirs.py",
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
    name = "gtsrb_tools",
    srcs = ["python/semisup/gtsrb_tools.py",
            "python/semisup/data_dirs.py",
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
            "python/semisup/data_dirs.py",
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
            "python/semisup/data_dirs.py",
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
            "python/semisup/data_dirs.py",
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
    name = "architectures",
    srcs = ["python/semisup/architectures.py",
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
        ":gtsrb_tools",
        ":svhn_tools",
        ":synth_tools",
        ":stl10_tools",
        ":architectures",        
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        #"//tensorflow/python:framework",
    ],
)


py_binary(
    name = "gts_train_eval",
    srcs = ["python/semisup/gts_train_eval.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":semisup",
        ":mnist_tools",
        ":gtsrb_tools",
        ":svhn_tools",
        ":synth_tools",
        ":stl10_tools",
        ":architectures",
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
        ":gtsrb_tools",
        ":svhn_tools",
        ":synth_tools",
        ":stl10_tools",
        ":architectures",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        #"//tensorflow/python:pywrap_tensorflow",
    ],
)


py_binary(
    name = "eval",
    srcs = ["python/semisup/eval.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":semisup",
        ":mnist_tools",
        ":gtsrb_tools",
        ":svhn_tools",
        ":synth_tools",
        ":stl10_tools",
        ":architectures",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        #"//tensorflow/python:pywrap_tensorflow",
    ],
)

py_binary(
    name = "train_imagenet",
    srcs = ["python/semisup/train_imagenet.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":semisup",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/contrib/slim:slim",        
        "//tensorflow/contrib/slim:nets",         
    ],
)

py_binary(
    name = "eval_imagenet",
    srcs = ["python/semisup/eval_imagenet.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":semisup",
        ":architectures",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/contrib/slim:slim",        
        "//tensorflow/contrib/slim:nets",         
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
