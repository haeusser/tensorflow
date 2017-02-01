#!/usr/bin/env bash
# to be run from tensorflow root directory
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k --jobs=8 //tensorflow/contrib/semisup:train
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k --jobs=8 //tensorflow/contrib/semisup:eval
#bazel build -c opt --config=cuda --jobs=8 //tensorflow/contrib/semisup:eval
