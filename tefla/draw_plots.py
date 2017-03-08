import matplotlib.pyplot as plt


def subplots2(ax, epoch_list, train_loss_list, validation_loss_list, validation_accuracy_list, validation_kappa_list,
              epoch, epoch_validation_accuracy, epoch_validation_kappa, epoch_training_loss, epoch_validation_loss):
    ax[0].scatter(epoch, epoch_training_loss, c='r', s=50)
    ax[0].plot(epoch_list, train_loss_list, 'red', label='Training loss')
    ax[0].scatter(epoch, epoch_validation_loss, c='g', s=50)
    ax[0].plot(epoch_list, validation_loss_list, 'green', label='Validation loss')
    ax[0].set_title('Epoch vs loss')
    plt.pause(.001)
    ax[1].scatter(epoch, epoch_validation_accuracy, c='b', s=50)
    ax[1].plot(epoch_list, validation_accuracy_list, 'blue')
    ax[1].set_title('Epoch vs Validation accuracy')
    plt.pause(.001)
    ax[2].scatter(epoch, epoch_validation_kappa, c='y', s=50)
    ax[2].plot(epoch_list, validation_kappa_list, 'yellow')
    ax[2].set_title('Epoch vs Validation Kappa')
    plt.pause(.001)
