#!/usr/bin/env bash
bazel build -c opt --config=cuda :mnist_train_eval
