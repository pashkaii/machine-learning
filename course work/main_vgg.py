import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast
from vgg import VGG


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg(batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs['E'], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg(False, pretrained, progress, **kwargs)


def forward_model(model, testloader):
    total_correct = 0
    total_images = 0
    i = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            if i % 100 == 0:
                print(i)
            i = i + 1

    model_accuracy = total_correct / total_images * 100
    return model_accuracy


if __name__ == '__main__':
    batch_size = 16

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # приведение данных в нужный формат и нормализация

    # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) - установка среднего и стандартного отклонения для каждого канала RGB
    testset = torchvision.datasets.CIFAR10(root='data/test/',
                                           train=False,
                                          download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False)

    '''
        dataiter = iter(trainloader)
        images, labels = dataiter.next()

        fig, axes = plt.subplots(1, len(images), figsize=(12, 2.5))
        for idx, image in enumerate(images):
            axes[idx].imshow(convert_to_imshow_format(image))
            axes[idx].set_title(classes[labels[idx]])
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
        plt.show()
    '''

    model = vgg19()
    model_accuracy = forward_model(model, testloader)
    print('Точность модели : {:.2f}%'.format(model_accuracy))
