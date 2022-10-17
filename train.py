import os

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from FaashionSegNet import fashionsegnet
from fashionsegnet_training import (CE, Focal_Loss, dice_loss_with_CE,
                                  dice_loss_with_Focal_Loss)
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader import PSPnetDataset
from utils.utils_metrics import Iou_score, f_score


if __name__ == "__main__":     

    num_classes = 6
    
    #---------------------------------------------------------#
    #   Pretained Weight
    #---------------------------------------------------------#
    model_path          = ""

    input_shape         = [480, 480]

    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    Freeze_lr           = 5e-4
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr         = 5e-5

    VOCdevkit_path      = 'VOCdevkit'

    dice_loss       = True

    focal_loss      = True

    cls_weights     = np.ones([num_classes], np.float32)

    Freeze_Train    = True

    num_workers     = 1

    model = fashionsegnet([input_shape[0], input_shape[1], 3], num_classes)

    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model.load_weights(model_path, by_name=True, skip_mismatch=True)

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()

    logging         = TensorBoard(log_dir = 'logs/')
    checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                        monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
    reduce_lr       = ExponentDecayScheduler(decay_rate = 0.94, verbose = 1)
    early_stopping  = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    loss_history    = LossHistory('logs/')

    freeze_layers = 172
    
    if focal_loss:
        if dice_loss:
            loss = dice_loss_with_Focal_Loss(cls_weights)
        else:
            loss = Focal_Loss(cls_weights)
    else:
        if dice_loss:
            loss = dice_loss_with_CE(cls_weights)
        else:
            loss = CE(cls_weights)

    if Freeze_Train:
        for i in range(freeze_layers): model.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        epoch_step      = len(train_lines) // batch_size
        epoch_step_val  = len(val_lines) // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small for training. Please expand the dataset.")
        
        model.compile(loss = loss,
                optimizer = Adam(lr=lr),
                metrics = [f_score()])

        train_dataloader    = PSPnetDataset(train_lines, input_shape, batch_size, num_classes, VOCdevkit_path)
        val_dataloader      = PSPnetDataset(val_lines, input_shape, batch_size, num_classes, VOCdevkit_path)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), batch_size))
        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataloader,
            validation_steps    = epoch_step_val,
            epochs              = end_epoch,
            initial_epoch       = start_epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )
    
    if Freeze_Train:
        for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        epoch_step      = len(train_lines) // batch_size
        epoch_step_val  = len(val_lines) // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small for training. Please expand the dataset.")
        
        model.compile(loss = loss,
                optimizer = Adam(lr=lr),
                metrics = [f_score()])

        train_dataloader    = PSPnetDataset(train_lines, input_shape, batch_size, num_classes, True, VOCdevkit_path)
        val_dataloader      = PSPnetDataset(val_lines, input_shape, batch_size, num_classes, False, VOCdevkit_path)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), batch_size))
        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataloader,
            validation_steps    = epoch_step_val,
            epochs              = end_epoch,
            initial_epoch       = start_epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )
