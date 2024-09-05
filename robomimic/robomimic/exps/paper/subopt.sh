#!/bin/bash

# ==========subopt==========

#  task: lift
#    dataset type: mh/worse
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse/low_dim/iris.json

#  task: lift
#    dataset type: mh/worse
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse/image/cql.json

#  task: lift
#    dataset type: mh/okay
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay/low_dim/iris.json

#  task: lift
#    dataset type: mh/okay
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay/image/cql.json

#  task: lift
#    dataset type: mh/better
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/better/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/better/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/better/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/better/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/better/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/better/low_dim/iris.json

#  task: lift
#    dataset type: mh/better
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/better/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/better/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/better/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/better/image/cql.json

#  task: lift
#    dataset type: mh/worse_okay
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_okay/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_okay/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_okay/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_okay/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_okay/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_okay/low_dim/iris.json

#  task: lift
#    dataset type: mh/worse_okay
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_okay/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_okay/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_okay/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_okay/image/cql.json

#  task: lift
#    dataset type: mh/worse_better
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_better/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_better/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_better/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_better/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_better/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_better/low_dim/iris.json

#  task: lift
#    dataset type: mh/worse_better
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_better/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_better/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_better/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/worse_better/image/cql.json

#  task: lift
#    dataset type: mh/okay_better
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay_better/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay_better/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay_better/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay_better/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay_better/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay_better/low_dim/iris.json

#  task: lift
#    dataset type: mh/okay_better
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay_better/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay_better/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay_better/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/lift/mh/okay_better/image/cql.json

#  task: can
#    dataset type: mh/worse
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse/low_dim/iris.json

#  task: can
#    dataset type: mh/worse
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse/image/cql.json

#  task: can
#    dataset type: mh/okay
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay/low_dim/iris.json

#  task: can
#    dataset type: mh/okay
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay/image/cql.json

#  task: can
#    dataset type: mh/better
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/better/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/better/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/better/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/better/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/better/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/better/low_dim/iris.json

#  task: can
#    dataset type: mh/better
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/better/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/better/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/better/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/better/image/cql.json

#  task: can
#    dataset type: mh/worse_okay
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_okay/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_okay/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_okay/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_okay/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_okay/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_okay/low_dim/iris.json

#  task: can
#    dataset type: mh/worse_okay
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_okay/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_okay/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_okay/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_okay/image/cql.json

#  task: can
#    dataset type: mh/worse_better
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_better/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_better/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_better/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_better/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_better/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_better/low_dim/iris.json

#  task: can
#    dataset type: mh/worse_better
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_better/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_better/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_better/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/worse_better/image/cql.json

#  task: can
#    dataset type: mh/okay_better
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay_better/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay_better/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay_better/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay_better/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay_better/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay_better/low_dim/iris.json

#  task: can
#    dataset type: mh/okay_better
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay_better/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay_better/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay_better/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/can/mh/okay_better/image/cql.json

#  task: square
#    dataset type: mh/worse
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse/low_dim/iris.json

#  task: square
#    dataset type: mh/worse
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse/image/cql.json

#  task: square
#    dataset type: mh/okay
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay/low_dim/iris.json

#  task: square
#    dataset type: mh/okay
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay/image/cql.json

#  task: square
#    dataset type: mh/better
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/better/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/better/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/better/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/better/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/better/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/better/low_dim/iris.json

#  task: square
#    dataset type: mh/better
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/better/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/better/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/better/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/better/image/cql.json

#  task: square
#    dataset type: mh/worse_okay
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_okay/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_okay/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_okay/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_okay/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_okay/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_okay/low_dim/iris.json

#  task: square
#    dataset type: mh/worse_okay
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_okay/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_okay/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_okay/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_okay/image/cql.json

#  task: square
#    dataset type: mh/worse_better
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_better/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_better/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_better/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_better/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_better/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_better/low_dim/iris.json

#  task: square
#    dataset type: mh/worse_better
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_better/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_better/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_better/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/worse_better/image/cql.json

#  task: square
#    dataset type: mh/okay_better
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay_better/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay_better/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay_better/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay_better/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay_better/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay_better/low_dim/iris.json

#  task: square
#    dataset type: mh/okay_better
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay_better/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay_better/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay_better/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/square/mh/okay_better/image/cql.json

#  task: transport
#    dataset type: mh/worse
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse/low_dim/iris.json

#  task: transport
#    dataset type: mh/worse
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse/image/cql.json

#  task: transport
#    dataset type: mh/okay
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay/low_dim/iris.json

#  task: transport
#    dataset type: mh/okay
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay/image/cql.json

#  task: transport
#    dataset type: mh/better
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/better/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/better/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/better/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/better/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/better/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/better/low_dim/iris.json

#  task: transport
#    dataset type: mh/better
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/better/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/better/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/better/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/better/image/cql.json

#  task: transport
#    dataset type: mh/worse_okay
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_okay/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_okay/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_okay/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_okay/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_okay/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_okay/low_dim/iris.json

#  task: transport
#    dataset type: mh/worse_okay
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_okay/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_okay/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_okay/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_okay/image/cql.json

#  task: transport
#    dataset type: mh/worse_better
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_better/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_better/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_better/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_better/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_better/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_better/low_dim/iris.json

#  task: transport
#    dataset type: mh/worse_better
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_better/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_better/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_better/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/worse_better/image/cql.json

#  task: transport
#    dataset type: mh/okay_better
#      hdf5 type: low_dim
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay_better/low_dim/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay_better/low_dim/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay_better/low_dim/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay_better/low_dim/cql.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay_better/low_dim/hbc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay_better/low_dim/iris.json

#  task: transport
#    dataset type: mh/okay_better
#      hdf5 type: image
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay_better/image/bc.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay_better/image/bc_rnn.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay_better/image/bcq.json
python vista/robomimic/robomimic/scripts/train.py --config vista/robomimic/robomimic/exps/paper/subopt/transport/mh/okay_better/image/cql.json

