import os
import pickle
class epoch_object(object):
    def __init__(self,epoch,val_accuracy,val_kappa,training_loss,validation_loss,loss_ratio):
        self.epoch=epoch
        self.val_accuracy=val_accuracy
        self.val_kappa=val_kappa
        self.training_loss=training_loss
        self.validation_loss=validation_loss
        self.loss_ratio=loss_ratio

def delete_file(file_path):
    input='y'
    if (os.path.exists(file_path)):
        input = raw_input("Run_script Logs Already Exists... Do you want to overwrite ?[y/[n]]")
        if input=='y':
            os.system('rm -r ' +file_path)
            # os.mknod(file_path)
        else:
            print 'New logs will not be formed...'
    else:
        pass
        # os.mknod(file_path)
    return input

def store_logs(epoch,val_accuracy,val_kappa,training_loss,validation_loss,loss_ratio):
    with open('run_script_logs.pkl','ab') as out:
        epoch_values=epoch_object(epoch,val_accuracy,val_kappa,training_loss,validation_loss,loss_ratio)
        pickle.dump(epoch_values,out,pickle.HIGHEST_PROTOCOL)



