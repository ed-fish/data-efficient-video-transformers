import torchvision.models as torch_models
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
import torch


class EmbeddingExtractor:
    def __init__(self, config):
        self.image_net = torch_models.resnet50(pretrained=True)
        self.video_net = torch_models.video.r3d_18(pretrained=True)
        self.location_net = torch_models.resnet50(pretrained=False)
        # self.audio_net = torch.hub.load('harritaylor/torchvggish', 'vggish')
        location_weights = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        self.location_net.load_state_dict(model_zoo.load_url(location_weights))
#       self.depth_net = torch.hub.load("intel-isl/MiDas", "MiDaS")
        self.location_net.fc = Identity()
#       self.depth_net.scratch = Identity()
        self.image_net.fc = Identity()
        self.video_net.fc = Identity()
        self.device = torch.device(config["gpu"].get(int))

    def init_models(self, m):
        m = m.to(self.device)
        m = m.eval()

    def forward_img(self, tensor):
        self.init_models(self.image_net)
        with torch.no_grad():
            tensor = tensor.to(self.device)
            output = self.image_net.forward(tensor).cpu()
            return output

    def forward_location(self, tensor):
        self.init_models(self.location_net)
        with torch.no_grad():
            tensor = tensor.to(self.device)
            output = self.location_net.forward(tensor).cpu()
            return output

    def forward_depth(self, tensor):
        self.init_models(self.depth_net)
        with torch.no_grad():
            tensor = tensor.to(self.device)
            output = self.depth_net.forward(tensor).cpu()
            return output

    def forward_video(self, tensor_stack):
        self.init_models(self.video_net)
        with torch.no_grad():
            tensor_stack = tensor_stack.to(self.device)
            output = self.video_net.forward(tensor_stack).cpu()
            return output

    def forward_audio(self, audio_sample):
        output = self.audio_net.forward(audio_sample)
        return output

    def depth_network_pool(self, depth_output):

        with torch.no_grad():
            depth_output = torch.flatten(depth_output, start_dim=1).unsqueeze(0)
            pool = ((1, 2048)).to(self.device)
            depth_output = pool(depth_output)
            depth_output = depth_output.squeeze(0)
            output = depth_output.cpu()
            return output

    def return_expert_for_key(self, key, raw_tensor):

        output = []
        if key == "image":
            for img in raw_tensor:
                img = img.squeeze(1)
                output.append(self.forward_img(img).to('cpu'))
            output = torch.stack(output)
            output = output.transpose(0, 2)
            output = F.adaptive_avg_pool1d(output, 1)
            output = output.transpose(1, 0).squeeze(2)

        if key == "motion" or key == "video":
            with torch.no_grad():
                raw_tensor = raw_tensor.unsqueeze(0)
                output = self.forward_video(raw_tensor)

        if key == "location":
            for img in raw_tensor:
                with torch.no_grad():
                    img = img.squeeze(1)
                    output.append(self.forward_location(img).cpu())
            output = torch.stack(output)
            output = output.transpose(0, 2)
            output = F.adaptive_avg_pool1d(output, 1)
            output = output.transpose(1, 0).squeeze(2)

        return output


    def return_expert_for_key_pretrained(self, key, raw_tensor):

        output = []
        if key == "image":
            output = torch.stack(raw_tensor)
            output = output.transpose(0, 2)
            output = F.adaptive_avg_pool1d(output, 1)
            output = output.transpose(1, 0).squeeze(2)
            output= output.squeeze(1)
            print(output.shape)

        if key == "motion" or key == "video":
            output = raw_tensor[0].unsqueeze(0)
            print(output.shape)

        if key == "location":
            output = torch.stack(raw_tensor)
            output = output.transpose(0, 2)
            output = F.adaptive_avg_pool1d(output, 1)
            output = output.transpose(1, 0).squeeze(2)
            output = output.squeeze(1)
            print(output.shape)

        return output



class Identity(nn.Module):
    def forward(self, x):
        return x

