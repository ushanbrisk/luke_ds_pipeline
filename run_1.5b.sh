#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/ssd2/output_test_202509251543
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5;DS_SKIP_CUDA_CHECK=1 deepspeed --master_port 5524 main_pipeline.py \
 --enable_tensorboard \
 --tensorboard_path $OUTPUT \
 --output_dir $OUTPUT &> $OUTPUT/training.log
