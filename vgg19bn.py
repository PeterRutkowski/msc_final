import torch
import layer_outputs
import numpy as np
from torchvision.models import vgg19_bn
from torchvision import transforms


class VGG19bn:
    def __init__(self, layers=[57, 60]):
        self.layers = layers
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = vgg19_bn(pretrained=True).to(self.device)
        self.model.eval()
        self.outputs = layer_outputs.LayerOutputs()
        self.__hook_model()

        self.preprocess = transforms.Compose([transforms.Resize([224, 224]),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                   std=[0.229, 0.224, 0.225])])

    def __hook_model(self):
        for i in range(53):
            self.model.features[i].register_forward_hook(self.outputs)
        self.model.avgpool.register_forward_hook(self.outputs)
        for i in range(7):
            self.model.classifier[i].register_forward_hook(self.outputs)

    def predict(self, img):
        tensor = self.preprocess(img)
        tensor = tensor.unsqueeze(0).to(self.device)
        self.model(tensor)
        features = torch.cat(tuple([self.outputs.outputs[layer] for layer in self.layers]), dim=1)
        self.outputs.clear()

        return np.squeeze(features.cpu().detach().numpy())
