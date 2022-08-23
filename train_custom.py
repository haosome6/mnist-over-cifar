import torch, torchvision
from torch import nn, optim
from src.models.model import Generator, Discriminator
from src.utils.data import get_mnist_loader, get_cifar_loader, get_mnist_over_cifar_loader
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def create_model(args):
    G = Generator(args.g_input_dim)
    D = Discriminator()

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')
    return G, D

def compute_acc(preds, labels):
	correct = 0
	preds_ = preds.data.max(1)[1]
	correct = preds_.eq(labels.data).cpu().sum()
	acc = float(correct) / float(len(labels.data)) * 100.0
	return acc

def gan_training_loop(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # eval_noise = torch.FloatTensor(args.batch_size, args.g_input_dim, 1, 1).normal_(0, 1)

    eval_noise = torch.FloatTensor(args.batch_size, args.g_input_dim, 1, 1).normal_(0, 1)
    eval_noise_ = np.random.normal(0, 1, (args.batch_size, args.g_input_dim))
    eval_label = np.random.randint(0, 10, args.batch_size)
    eval_onehot = np.zeros((args.batch_size, 10))
    eval_onehot[np.arange(args.batch_size), eval_label] = 1
    eval_noise_[np.arange(args.batch_size), :10] = eval_onehot[np.arange(args.batch_size)]
    eval_noise_ = (torch.from_numpy(eval_noise_))
    eval_noise.data.copy_(eval_noise_.view(args.batch_size, args.g_input_dim, 1, 1))
    eval_noise = eval_noise.to(device)


    # create generator and discriminator
    G, D = create_model(args)

    # get parameters of the model 
    g_params = G.parameters()
    d_params = D.parameters()

    # create optimizers
    g_optimizer = optim.Adam(g_params, args.lr)
    d_optimizer = optim.Adam(d_params, args.lr)

    losses = {'iteration': [], 'D_fake_loss': [], 'D_real_loss': [], 'G_loss': []}

    source_loss = nn.BCELoss()
    class_loss = nn.NLLLoss()

    real_label = torch.FloatTensor(args.batch_size).to(device)
    real_label.fill_(1)
    fake_label = torch.FloatTensor(args.batch_size).to(device)
    fake_label.fill_(0)

    mnist_over_cifar_train_loader, mnist_over_cifar_test_loader = get_mnist_over_cifar_loader(args.batch_size)

    for epoch in range(args.num_epochs + 1):

        for idx, (imgs, mnist_labels) in enumerate(mnist_over_cifar_train_loader):

            # train Discriminator with real data
            d_optimizer.zero_grad()

            imgs, mnist_labels = imgs.to(device), mnist_labels.to(device)

            predicted_source, predicted_class = D(imgs)
            source_error = source_loss(predicted_source, real_label)
            class_error = class_loss(predicted_class, mnist_labels)
            D_real_loss = source_error + class_error
            D_real_loss.backward()
            d_optimizer.step()

            D_real_class_acc = compute_acc(predicted_class, mnist_labels)

            # train Discriminator with fake data
            noise = np.random.normal(0, 1, (args.batch_size, args.g_input_dim))
            G_labels = np.random.randint(0, 10, args.batch_size)

            noise = ((torch.from_numpy(noise)).float())
            noise = noise.to(device)
            G_labels = ((torch.from_numpy(G_labels)).long())
            G_labels = G_labels.to(device)

            G_imgs = G(noise)

            predicted_source_G, predicted_class_G = D(G_imgs.detach())
            source_error_G = source_loss(predicted_source_G, fake_label)
            class_error_G = class_loss(predicted_class_G, G_labels)
            D_fake_loss = source_error_G + class_error_G
            D_fake_loss.backward()
            d_optimizer.step()

            # train Generator
            G.zero_grad()
            predicted_source_G, predicted_class_G = D(G_imgs)
            source_error_G = source_loss(predicted_source_G, real_label)
            class_error_G = class_loss(predicted_class_G, G_labels)
            G_loss = source_error_G + class_error_G
            G_loss.backward()
            g_optimizer.step()

        # TODO: add losses
        # losses['']

        # result
        if epoch % args.log_step == 0:
            # print("Epoch--[{} / {}], Loss_Discriminator--[{}], Loss_Generator--[{}],Accuracy--[{}]".format(epoch, args.num_epochs, error_fake, error_gen,accuracy))
            print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(epoch, args.num_epochs, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))
            constructed = G(eval_noise)
            torchvision.utils.save_image(
                constructed.data,
                '%s/results_epoch_%04d.png' % ('results', epoch)
            )



args = AttrDict()
args_dict = {
    'lr': 1e-4,
    'num_epochs': 2000,
    'batch_size': 512,
    'g_input_dim': 64,
    'log_step': 200
}
args.update(args_dict)

gan_training_loop(args)