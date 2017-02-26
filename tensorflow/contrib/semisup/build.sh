#!/usr/bin/env bash
# to be run from tensorflow root directory
bazel build --config=opt --config=cuda --verbose_failures //tensorflow/contrib/semisup:train
bazel build --config=opt --config=cuda --verbose_failures //tensorflow/contrib/semisup:eval
