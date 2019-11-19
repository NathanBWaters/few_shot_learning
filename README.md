# few_shot_learning

Apply Siamese networks to learn a similarity function.
Achieved 99.8% on training and 99.1% on validation for MNIST


## MNIST Dataset Log:
* Started with a simple CNN architecture, got around 95%
* Tried a mobile net backend, which failed spectacularly.  This is probably for
  two reasons: image-net does not have black and white number data and the model
  had too large of a capacity.
* Applied more dense layers to the end of the network which improved the model
  by a 1%
* Instead of simple binary_crossentropy, I applied a contrastive loss from 
  http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf.  This
  brought the model up to 99% accuracy on MNIST

## Omniglot Dataset Log:
* Applied the simple CNN with binary_crossentropy, got about 74%
* Contrastive loss brought the accuracy up to 86% training and 85% validation
* Applying dropout and simple numpy fliplr / flipud brought the accuracy down
  to 81% training and 76% validation
* Removing the dropout got 90% training and 89% validation, so it looks like
  the simply numpy data augmentation was effective.
* I expanded the data augmetation to include [imgaug](https://imgaug.readthedocs.io/en/latest/)
  I visualized the effects with `few_shot/visualize.py` so that the augmentations
  were at an appropriate setting.  Surprisingly, this brought the accuracy
  down to 73% training and 78% validation.  Very cool to see it perform better
  on (un-augmented) data that it hadn't seen with the validation set.
* I switched out SGD for Adam.  It slowly got up to 88% training and 86%
  validation but kept learning long after the SGD methods plateaud.
  This makes me think the learning rate of 1.0e-5 was too low.
* Adam at 1.0e-3 did not learn at all.  Stayed around 50/50.
* Adam at 1.0e-4 skyrocketed up in accuracy.  96% training and 94% validation.
* Adam at 3.0e-4 (also known as the "Magic Learning Rate") performed even better
  with 97% training and validation accuracy
* I increased the image size to 64x64 thinking the loss of information might
  be hurting the model's performance.  The model failed to learn the first time
  around for the first 15 epochs.
* I tried again, without changing anything and the model began learning just
  as well as the 32x32 image.  Wild.  Overtime, it improved slightly slower
  than the 32x32 image, so I'm going back to 32x32
* I switched the simple LeNet architecture with a DenseNet architecture.  To
  no surprise, this greatly improved the accuracy in the first 30 epochs.
  It then was followed by some sharp declines, which tells me that the learning
  rate is too high.  It eventually performed just as well as the LeNet
  architecture, despite being 10x larger of a model.
* I lowered the learning rate, and the model had a much smoother loss. However,
  I realized the DenseNets were not performing as well on the validation sets.
  Learning rate 3e-4 was about 91% accurate and LR 1e-4 was about 69% accurate.
  The model is overfitting using the DenseNet encoder.  This makese me believe
  the model is too large for the given problem.  The LeNet architecture had
  superior validation metrics than both DenseNet runs.  The validation score
  was far poorer in the 1e-4 than the 3e-4.  Humorously, after about 180 epochs
  in ended up doing slightly worse than my LeNet model.


