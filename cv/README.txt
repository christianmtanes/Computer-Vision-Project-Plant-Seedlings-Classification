the data here already split into training and validition ,
there is also the test file provided by the KAGGLE competetion.
and the data is in the following directories:

train =  training images
valid = validation images
test = test images

these directories should be placed in the same directory as the python files.

each python file runs a different model for transfer learning 
or/and different batch sizes.

a python file name consist of :
modelname_batch_size_validbatchsize_trainbatchsize.py

modelname = name of model used for transfer learning.

validbatchsize = batch size of the validation

train_size = batch size of the training.

best test results are with:
resnet50_batch_size_25_25.py  and resnet50_batch_size_25_25_data_aug.py

presentation of the project and its results can be found in
the presentation file : Plant Seedlings Classification.pptx
or Plant Seedlings Classification pdf.pdf
