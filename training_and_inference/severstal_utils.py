import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import cv2
from keras.utils import Sequence
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras import backend as K
import os
import random
from time import time

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

def load_severstal_data(base_path='/home/isdgenomics/users/dipamcha/kaggle/severstal-steel-defect-detection/data/',
                        seed=0, masks_path=None, num_folds=5, fold=1):
    """ train_fault_only - Train on only faults, will train on only faulty images if true """
    train_df = pd.read_csv(base_path + 'train.csv')
    train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
    train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
    
    if masks_path is not None:
        mask_df = pd.read_pickle(masks_path)
    else:
        masks_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
        masks_df.sort_values('hasMask', ascending=False, inplace=True)
    
    random_state = np.random.RandomState(seed=seed)
    all_idx = list(mask_df.index)
    random_state.shuffle(all_idx)
    
    assert fold <= num_folds and fold >= 1, "Fold number should be between 1 to num_folds"
    num_val = len(all_idx)//num_folds
    vs, ve = (fold-1)*num_val, fold*num_val
    val_idx = all_idx[vs:ve]
    train_idx = all_idx[:vs] + all_idx[ve:]
    
    return train_df, mask_df, train_idx, val_idx


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle, shape=(256,1600)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.bool)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def build_masks(rles, input_shape, background=True):
    depth = len(rles)
    if background:
        depth += 1
    height, width = input_shape
    masks = np.zeros((height, width, depth), np.bool)
    
    for i, rle in enumerate(rles):
        if type(rle) is str:
            masks[:, :, i] = rle2mask(rle, (width, height))
            
    if background:
        masks[:,:,-1] = np.logical_not(np.logical_or.reduce(masks, axis=-1))
    
    return masks

def build_rles(masks):
    width, height, depth = masks.shape
    
    rles = [mask2rle(masks[:, :, i])
            for i in range(depth)]
    
    return rles

def dice_coef(y_true, y_pred, softmax_preds=True, thresh=0.5): # competition metric is mean over all individual masks
    _, h, w, c = y_pred.get_shape().as_list()
    if softmax_preds:
        y_pred_flat = tf.reshape(y_pred, [-1, h*w, c])
        y_pred_argmax = tf.math.argmax(y_pred_flat, axis=-1)
        y_pred_onehot = tf.one_hot(y_pred_argmax, c, on_value=1, off_value=0)
        y_pred_f = tf.cast(tf.reshape(y_pred_onehot, [-1, h*w, c]), tf.float32)[:,:,:-1]
        y_true_f = tf.reshape(y_true, [-1, h*w, c])[:,:,:-1]
    else:
        y_pred_f = tf.cast(tf.reshape(y_pred > thresh, [-1, h*w, c]), tf.float32)
        y_true_f = tf.reshape(y_true, [-1, h*w, c])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    masksum = tf.reduce_sum(y_true_f + y_pred_f, axis=1)
    dice = (2.*intersection + K.epsilon()) / (masksum + K.epsilon())
    return tf.reduce_mean(dice)

def gumbel_dice_loss(y_true, y_pred, temperature=1.):
    smooth = 0.1 #1 #### 0.1 gives better dice score for noisy labels with this loss
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

def softmax_dice_loss(y_true, y_pred, alpha=1., gumbel_temp=0.1):
    if alpha > 0:
        loss = categorical_crossentropy(y_true, y_pred, from_logits=True) + \
                alpha*gumbel_dice_loss(y_true, y_pred, temperature=gumbel_temp)
        return loss
    else:
        return categorical_crossentropy(y_true, y_pred, from_logits=True)
    
def argmax_predictions(predictions):
    predflat = np.reshape(predictions, (-1, predictions.shape[-1]))
    p_am = np.argmax(predflat, axis=-1)
    outs = np.zeros(predflat.shape)
    outs[np.arange(p_am.size), p_am] = 1
    outs = np.reshape(outs, predictions.shape)
    return outs

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='./train_images', background=True,
                 batch_size=32, dim=(256, 1600),
                 n_classes=4, seed=2019, shuffle=False,
                 use_nonblackregion=False,
                 random_flips=False,
                 sampling_weights=None,
                 faultidx = None,
                 faultloc_keys = None,
                 dual_task_classifier=False,
                 output_layer_names=None,
                 mean_std = (0., 1.),
                 augment=False,
                 wraparound_collate=False):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list(list_IDs)
        if wraparound_collate:
            extra = (-len(self.list_IDs)) % self.batch_size
            self.list_IDs.extend(self.list_IDs[:extra])
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = np.random.RandomState(seed=seed)
        self.background = background
        if self.background:
            self.n_classes += 1
        if random_flips:
            self.flips_dict = {0: None, 1: 0, 2: 1, 3: (0,1)}
        else:
            self.flips_dict = {0: None}
        
        self.sampling_weights = sampling_weights
        if sampling_weights is not None:
            self.faultidx = {}
            for key in faultloc_keys:
                self.faultidx[key] = list(self.df.loc[self.list_IDs][self.df[key].apply(lambda x: len(x) > 0)].index)
            self.faultloc_keys = faultloc_keys
        
        self.use_nonblackregion = use_nonblackregion
        if use_nonblackregion:
            self.crop_size = [256, 320]   ############ Hardcoded crop size - Fix later
        self.dual_task_classifier = dual_task_classifier
        if dual_task_classifier:
            assert output_layer_names is not None, "Layer names are required for dual task classifier mode"
        self.layer_names = output_layer_names
        self.augment = augment
        self.mean_std = mean_std
        self.cache_x = {}
        self.cache_y = {}

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        all_len = len(self.list_IDs)
        return all_len // self.batch_size
    
    def get_random_batch(self):
        index = self.random_state.randint(self.__len__())
        return self.__getitem__(index)

    def __getitem__(self, index):
        'Generate one batch of data'
        layer_names = self.layer_names
        
        if self.sampling_weights is not None:
            assert self.use_nonblackregion, "Only supported when using non black regions"
            list_IDs_batch = []
            fault_selects = self.random_state.choice(self.faultloc_keys, p=self.sampling_weights, 
                                                     size=self.batch_size)
            for fs in fault_selects:
                list_IDs_batch.append(self.random_state.choice(self.faultidx[fs]))
        else:
            list_IDs_batch = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]
        flips_batch = self.random_state.randint(len(self.flips_dict), size=self.batch_size)
        X = self.__generate_X(list_IDs_batch)

        if self.use_nonblackregion:
            nbr_batch = self.df.loc[list_IDs_batch]['nonBlackRegion']
            X_crops = np.empty((self.batch_size, *self.crop_size, 3))
            ccrops = np.empty(self.batch_size,  dtype=int)
            for bi, nbr in enumerate(nbr_batch):
                if self.sampling_weights is not None:
                    list_ID, fault_ID = list_IDs_batch[bi], fault_selects[bi]
                    croplocs_list = self.df.loc[list_ID][fault_ID]
                    ccrops[bi] = self.random_state.choice(croplocs_list)
                    ccrops[bi] =min(ccrops[bi], 1600-320) #################### HACK ... FIX PROPERLY LATER
                        
                elif nbr[0] < self.dim[1] - self.crop_size[1]:
                    croplimit = max(nbr[0], nbr[1]-self.crop_size[1]) + 1
                    ccrops[bi] = self.random_state.choice(np.arange(nbr[0], croplimit))
                else:
                    ccrops[bi] = self.dim[1] - self.crop_size[1]
                cs, ce = ccrops[bi], ccrops[bi]+self.crop_size[1]
                assert cs >= 0, "Crop start %i out of range" % cs
                assert ce <= self.dim[1], "Crop end %i out of range" % ce
                
                flipaxis = self.flips_dict[flips_batch[bi]]
                if flipaxis is not None:
                    X_crops[bi] = np.flip(X[bi, :, cs:ce, :], axis=flipaxis)
                else:
                    X_crops[bi] = X[bi, :, cs:ce, :]
                
            X = X_crops

        
        if self.mode == 'fit':
            ticgeny = time()
            y = self.__generate_y(list_IDs_batch)
            
            if self.use_nonblackregion:
                y_crops = np.empty((self.batch_size, *self.crop_size, self.n_classes))
                for bi in range(self.batch_size):
                    cs, ce = ccrops[bi], ccrops[bi]+self.crop_size[1]
                    
                    flipaxis = self.flips_dict[flips_batch[bi]]
                    if flipaxis is not None:
                        y_crops[bi] = np.flip(y[bi, :, cs:ce, :], axis=flipaxis)
                    else:
                        y_crops[bi] = y[bi, :, cs:ce, :]
                        
                y = y_crops
            if self.dual_task_classifier:
                y_hasfaults = np.float32(np.any(y, axis=(1,2))[:, :-1])
                return X, {layer_names['segmentation']: y, 
                           layer_names['auxilliary']: y,
                           layer_names['classification']: y_hasfaults}
            else:
                return X, y
        
        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.random_state.shuffle(self.list_IDs)
            
    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 3))
        
        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].loc[ID]
            img_path = self.base_path + '/' + im_name
            img = self.__load_rgb(img_path)

            # Store samples
            X[i,] = img

        return X
    
    def __generate_y(self, list_IDs_batch):
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].loc[ID]
            masks = self.cache_y.get(im_name, None)
            if masks is None:
                image_df = self.target_df[self.target_df['ImageId'] == im_name]

                rles = image_df['EncodedPixels'].values
                masks = build_masks(rles, input_shape=self.dim, background=self.background)
                self.cache_y[im_name] = masks
            y[i, ] = np.float32(masks)

        return y
    
    
    def __load_rgb(self, img_path):
        img_name = img_path.split('/')[-1]
        sliceimg = self.cache_x.get(img_name, None)
        if sliceimg is None:
            img = cv2.imread(img_path)
            # Saving only one channel because images are actually gray
            self.cache_x[img_name] = img[:,:,0]
        else:
            # Back to 3 channels
            img = np.broadcast_to(sliceimg.copy()[..., None], shape=(*sliceimg.shape, 3))
        img = img.astype(np.float32) / 255.
        img = (img - self.mean_std[0]) / self.mean_std[1]
        if self.augment:
            if self.random_state.rand() < 0.8: # random brightness
                img = img + (self.random_state.rand() - 0.5) * 0.6
            if self.random_state.rand() < 0.5: # gaussian noise
                strength = self.random_state.rand()*0.07
                img = img + self.random_state.randn(*img.shape[:2], 1) * strength
        
        return img