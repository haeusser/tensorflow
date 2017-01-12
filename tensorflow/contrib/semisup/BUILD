# Description:
#   TODO(haeusser)


licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "semisup_py",
    srcs = ["__init__.py"] + glob(["python/semisup/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/numpy",
    ],
)


py_binary(
    name = "mnist_train_eval",
    srcs = ["python/semisup/mnist_train_eval.py"],
    srcs_version = "PY2AND3",
    deps = [

        "//tensorflow/contrib/semisup:semisup_py",
        "//tensorflow/contrib/slim",
        "//tensorflow/python:training",

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
