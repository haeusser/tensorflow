# Description:
#   TODO(haeusser)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "semisup_py",
    srcs = ["__init__.py",
	   ] + glob(["python/semisup/*.py"]),
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
	"//tensorflow/contrib/slim",
        #"//tensorflow/contrib/framework:framework_py",
        #"//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",        
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform",
        "//tensorflow/python:rnn",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",        
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:pywrap_tensorflow",
        "//third_party/py/numpy",
    ],
)


py_binary(
    name = "mnist_train_eval",
    srcs = ["python/semisup/mnist_train_eval.py",
           ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
	":semisup_py",
	#"//tensorflow:tensorflow_py",
        #"//tensorflow/contrib/semisup:semisup_py",
        #"//tensorflow/contrib/slim",
        #"//tensorflow/contrib/layers:layers_py",
        #"//tensorflow/python:framework",
        #"//tensorflow/python:platform",

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
