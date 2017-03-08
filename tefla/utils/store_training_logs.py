import os
import pickle


class EpochObject(object):
    def __init__(self, epoch, val_accuracy, val_kappa, training_loss, validation_loss, loss_ratio):
        self.epoch = epoch
        self.val_accuracy = val_accuracy
        self.val_kappa = val_kappa
        self.training_loss = training_loss
        self.validation_loss = validation_loss
        self.loss_ratio = loss_ratio


def delete_file(file_path):
    if os.path.exists(file_path):
        os.system('rm -r ' + file_path)


def store_logs(epoch, val_accuracy, val_kappa, training_loss, validation_loss, loss_ratio):
    with open('run_script_logs.pkl', 'ab') as out:
        epoch_values = EpochObject(epoch, val_accuracy, val_kappa, training_loss, validation_loss, loss_ratio)
        pickle.dump(epoch_values, out, pickle.HIGHEST_PROTOCOL)



