from experiment import classic_experiment as ce
from architecture import classic_architectures as ca
from functionalities import gpu

if __name__ == "__main__":
    # Set Training Parameters
    num_epoch = 100
    batch_size = 128
    lr_init = 1e-3
    milestones = [50, 80, 100]
    get_model = ca.mnist_model
    modelname = "classic_mnist"
    number_dev = 0
    device = gpu.get_device(number_dev)

    # Training
    cl_exp = ce.classic_experiment(num_epoch, batch_size, lr_init, milestones, get_model, modelname, device)
    cl_exp.get_dataset("mnist")
    cl_exp.train()
    cl_exp.plot_accuracy()