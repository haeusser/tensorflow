#!/usr/bin/env bash
# to be run from tensorflow root directory
bazel build -c opt --config=cuda //tensorflow/contrib/semisup:mnist_train_eval
