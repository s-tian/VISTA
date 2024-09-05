#!/bin/bash

# ==========dataset_size==========

#  task: lift
#    dataset type: ph/20_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/lift/ph/20_percent/low_dim/bc_rnn.json

#  task: lift
#    dataset type: ph/20_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/lift/ph/20_percent/image/bc_rnn.json

#  task: lift
#    dataset type: ph/50_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/lift/ph/50_percent/low_dim/bc_rnn.json

#  task: lift
#    dataset type: ph/50_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/lift/ph/50_percent/image/bc_rnn.json

#  task: lift
#    dataset type: mh/20_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/lift/mh/20_percent/low_dim/bc_rnn.json

#  task: lift
#    dataset type: mh/20_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/lift/mh/20_percent/image/bc_rnn.json

#  task: lift
#    dataset type: mh/50_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/lift/mh/50_percent/low_dim/bc_rnn.json

#  task: lift
#    dataset type: mh/50_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/lift/mh/50_percent/image/bc_rnn.json

#  task: can
#    dataset type: ph/20_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/can/ph/20_percent/low_dim/bc_rnn.json

#  task: can
#    dataset type: ph/20_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/can/ph/20_percent/image/bc_rnn.json

#  task: can
#    dataset type: ph/50_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/can/ph/50_percent/low_dim/bc_rnn.json

#  task: can
#    dataset type: ph/50_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/can/ph/50_percent/image/bc_rnn.json

#  task: can
#    dataset type: mh/20_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/can/mh/20_percent/low_dim/bc_rnn.json

#  task: can
#    dataset type: mh/20_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/can/mh/20_percent/image/bc_rnn.json

#  task: can
#    dataset type: mh/50_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/can/mh/50_percent/low_dim/bc_rnn.json

#  task: can
#    dataset type: mh/50_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/can/mh/50_percent/image/bc_rnn.json

#  task: square
#    dataset type: ph/20_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/square/ph/20_percent/low_dim/bc_rnn.json

#  task: square
#    dataset type: ph/20_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/square/ph/20_percent/image/bc_rnn.json

#  task: square
#    dataset type: ph/50_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/square/ph/50_percent/low_dim/bc_rnn.json

#  task: square
#    dataset type: ph/50_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/square/ph/50_percent/image/bc_rnn.json

#  task: square
#    dataset type: mh/20_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/square/mh/20_percent/low_dim/bc_rnn.json

#  task: square
#    dataset type: mh/20_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/square/mh/20_percent/image/bc_rnn.json

#  task: square
#    dataset type: mh/50_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/square/mh/50_percent/low_dim/bc_rnn.json

#  task: square
#    dataset type: mh/50_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/square/mh/50_percent/image/bc_rnn.json

#  task: transport
#    dataset type: ph/20_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/transport/ph/20_percent/low_dim/bc_rnn.json

#  task: transport
#    dataset type: ph/20_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/transport/ph/20_percent/image/bc_rnn.json

#  task: transport
#    dataset type: ph/50_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/transport/ph/50_percent/low_dim/bc_rnn.json

#  task: transport
#    dataset type: ph/50_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/transport/ph/50_percent/image/bc_rnn.json

#  task: transport
#    dataset type: mh/20_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/transport/mh/20_percent/low_dim/bc_rnn.json

#  task: transport
#    dataset type: mh/20_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/transport/mh/20_percent/image/bc_rnn.json

#  task: transport
#    dataset type: mh/50_percent
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/transport/mh/50_percent/low_dim/bc_rnn.json

#  task: transport
#    dataset type: mh/50_percent
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/dataset_size/transport/mh/50_percent/image/bc_rnn.json

