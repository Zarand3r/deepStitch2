# Given kinematics path, process kinematics data into a numpy file
# Figure out how to get the video name from the kinematic file name
# in the lightning_train CustomDataset __getitem__ method return the kinematics too. Modify the training_step because batch is now a 3-tuple.
# input_cuda, target_cuda = self.apply_transforms_GPU(batch[0:2], random_crop=self.hparams.random_crop)
# the kinematics is just a label. The forward function still just takes input_cuda.
# Do the old stuff with first two elements (input_cuda, target_cuda), and use the third (kinematics) for target_kinematics