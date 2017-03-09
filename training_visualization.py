
import click
import tefla.train_withf as train_withf
import tefla.train as train
import shutil
import pickle
import os

# logger = logging.getLogger().addHandler(logging.StreamHandler())

def copy_and_rename(data_dir,tag):
    shutil.copy2(data_dir + "predictions/" + tag + "/features.npy", data_dir)
    os.rename(data_dir + "features.npy", data_dir + tag + "_features.npy")

def analyze_data():
    lists = []
    with (open('run_script_logs.pkl', 'rb')) as file:
        while True:
         try:
             lists.append(pickle.load(file))
         except EOFError:
             break
    return lists

def find_best_epoch(list):
    # filter by best accuracy
    all_val_accuracy = [obj.val_accuracy for obj in list]
    max_accuracy=max(all_val_accuracy)
    all_max_accuracy=[i for i, j in enumerate(all_val_accuracy) if j == max_accuracy]
    all_max_val_accuracy_list = [list[j] for i, j in enumerate(all_max_accuracy)]
    # filter by best kappa
    all_val_kappa = [obj.val_kappa for obj in all_max_val_accuracy_list]
    max_kappa = max(all_val_kappa)
    all_max_kappa = [i for i, j in enumerate(all_val_kappa) if j == max_kappa]
    all_max_val_kappa_list = [all_max_val_accuracy_list[j] for i, j in enumerate(all_max_kappa)]
    # filter by max training/validation loss ratio
    all_loss_ratio = [obj.loss_ratio for obj in all_max_val_kappa_list]
    max_loss_ratio = max(all_loss_ratio)
    all_loss_ratio_list = [i for i, j in enumerate(all_loss_ratio) if j == max_loss_ratio]
    all_max_loss_ratio_list=[all_max_val_kappa_list[j] for i, j in enumerate(all_loss_ratio_list)]
    # filter by min validation loss
    all_val_loss = [obj.validation_loss for obj in all_max_loss_ratio_list]
    val_loss=min(all_val_loss)
    all_val_loss_list = [i for i, j in enumerate(all_val_loss) if j ==val_loss]
    all_min_val_loss_list = [all_max_loss_ratio_list[j] for i, j in enumerate(all_val_loss_list)]
    # filter by min training loss
    all_training_loss= [obj.training_loss for obj in all_min_val_loss_list]
    train_loss=min(all_training_loss)
    all_train_loss_list = [i for i, j in enumerate(all_training_loss) if j == train_loss]
    all_min_train_loss_list = [all_min_val_loss_list[j] for i, j in enumerate(all_train_loss_list)]

    return all_min_train_loss_list

@click.command()
@click.pass_context
def visual_training(ctx):
    data_dir="../../csx/data/"

    start_epoch = raw_input("Please enter the start epoch ")
    resume_lr = raw_input("Please enter the learning rate ")
    ctx.invoke(train_withf.main, model="examples/imagenet_tl_feature_extract/bottleneck_model.py",
               training_cnf="examples/imagenet_tl_feature_extract/train_cnf.py",
               data_dir=data_dir, resume_lr=float(resume_lr), start_epoch=int(start_epoch),visuals=True)
    print "Analyzing Data..."
    list = analyze_data()
    print ("Finding best Weights ...")
    epoch = find_best_epoch(list)
    print ("The best weights are of ")
    print ("epoch " + epoch[0].epoch.__str__())
visual_training()