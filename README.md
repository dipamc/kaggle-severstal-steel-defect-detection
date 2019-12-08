# Kaggle severstal-steel-defect-detection
My solution for the Kaggle Contest - Severstal Steel Defect Detection <br> https://www.kaggle.com/c/severstal-steel-defect-detection

Solution uses efficientnetb4 with modified unet architecture. Check train_and_inference folder for full code.

## Main ideas

### Train on non-black regions
The test dataset had much more images with black boundaries. While training with black augmentation was one option, it didn't yield as much results as training only on non-black regions. It is easy to crop the black region out with simple thresholding.
<br><br>

### Modified dice loss with gumbel noise

Adding noise drawn from gumbel distribution to the predictions allows use of a smaller smoothness parameters without blowing up the gradients. This was especially helpful for the noisy labels in this competition

```
def gumbel_softmax_dice_loss(y_true, y_pred, temperature=1.):
    smooth = 0.01
    _, h, w, c = y_pred.get_shape().as_list()
    y_pred_f = tf.reshape(y_pred, [-1, h*w, c])
    y_true_f = tf.reshape(y_true, [-1, h*w, c])

    z = tf.random_uniform(tf.shape(y_pred_f))
    gumbel_z = tf.log(-tf.log(z))

    y_pred_tempscaled = y_pred_f / temperature
    y_pred_relaxedsample = tf.nn.softmax(y_pred_tempscaled - gumbel_z)
    intersection = tf.reduce_sum(y_true_f * y_pred_relaxedsample, axis=1)
    masksum = tf.reduce_sum(y_true_f + y_pred_relaxedsample, axis=1)
    dice = (2.*intersection + smooth) / (masksum + smooth)
    return 1 - tf.reduce_mean(dice)
```
<br>

### Train on crop inference on full size
Since the images are large, it helps to run fully convolutional architecture on crops for larger batch size hence cleaner gradients. Since the labels are noisy, the cropping should be done so that a good portion of the mask is visible not just a few pixels. This is done as preprocessing before training to save time on multiple runs. See eda folder for code.
