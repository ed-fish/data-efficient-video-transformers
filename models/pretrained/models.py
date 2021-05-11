import torchvision.models as torch_models
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch


class EmbeddingExtractor:
    def __init__(self, config):
        self.image_net = torch_models.resnet50(pretrained=True)
        self.video_net = torch_models.video.r3d_18(pretrained=True)
        self.location_net = torch_models.resnet50(pretrained=False)
        self.audio_net = torch.hub.load('harritaylor/torchvggish', 'vggish')
        location_weights = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        self.location_net.load_state_dict(model_zoo.load_url(location_weights))
        self.depth_net = torch.hub.load("intel-isl/MiDas", "MiDaS")
        self.location_net.fc = Identity()
        self.image_net.fc = Identity()
        self.video_net.fc = Identity()
        models = [self.depth_net, self.location_net,
                  self.image_net, self.video_net, self.audio_net]
        self.device = torch.device(config["gpu"].get(int))
        self.init_models(models)

    def init_models(self, models):
        for m in models:
            print("initialising", m)
            m = m.to(self.device)
            m = m.eval()

    def forward_img(self, tensor):
        tensor = tensor.to(self.device)
        output = self.image_net.forward(tensor)
        return output

    def forward_location(self, tensor):
        tensor = tensor.to(self.device)
        output = self.location_net.forward(tensor)
        return output

    def forward_depth(self, tensor):
        tensor = tensor.to(self.device)
        output = self.depth_net.forward(tensor)
        output = self.depth_network_pool(output)
        return output

    def forward_video(self, tensor_stack):
        tensor_stack = tensor_stack.to(self.device)
        output = self.video_net.forward(tensor_stack)
        return output

    def forward_audio(self, audio_sample):
        output = self.audio_net.forward(audio_sample)
        return output

    def depth_network_pool(self, depth_output):
        depth_output = torch.flatten(depth_output, start_dim=1).unsqueeze(0)
        pool = torch.nn.AdaptiveAvgPool2d((1, 2048)).to(self.device)
        depth_output = pool(depth_output)
        depth_output = depth_output.squeeze(0)
        return depth_output


class Identity(nn.Module):
    def forward(self, x):
        return x

class ImgNetResNet(pl.LightningModule)
