
class Feature_Pyramid_Mid(nn.Module):
    def __init__(self):
        super(Feature_Pyramid_Mid, self).__init__()
        self.pool_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=14),
        )
        self.channels_reduce = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, mid):
        pyramid = self.pool_branch(mid)
        output = self.channels_reduce(pyramid)
        return output


class Feature_Pyramid_High(nn.Module):
    def __init__(self):
        super(Feature_Pyramid_High, self).__init__()
        self.pool_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=7),
        )
        self.channels_reduce = nn.Conv2d(512, 512, kernel_size=1)

    def forward(self, high):
        pyramid = self.pool_branch(high)
        return pyramid


class Feature_Pyramid_low(nn.Module):
    def __init__(self):
        super(Feature_Pyramid_low, self).__init__()
        self.pool_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=28),
        )
        self.channels_reduce = nn.Conv2d(128, 128, kernel_size=1)

    def forward(self, low):
        pyramid = self.pool_branch(low)
        output = self.channels_reduce(pyramid)
        return output


class TPN(pl.LightningModule):
    def __init__(self):
        super(TPN, self).__init__()
        self.net = custom_resnet.resnet34(True)
        self.pyramid_low = Feature_Pyramid_low()
        self.pyramid_mid = Feature_Pyramid_Mid()
        self.pyramid_high = Feature_Pyramid_High()
        self.reason = Reasoning()
        # self.fusion = nn.Sequential(nn.ReLU(), nn.Linear())

    def forward(self, x):
        low, mid, high = self.net(x)
        low_0 = self.pyramid_low(low).squeeze()
        mid_0 = self.pyramid_mid(mid).squeeze()
        high_0 = self.pyramid_high(high).squeeze()
        cnn_out = torch.cat((high_0, mid_0, low_0), dim=-1).unsqueeze(0)
        frame_feature = cnn_out.view(-1, 4 * 5, 896)
        output = self.reason(cnn_out)
        return output


def sum_group(x, groups=2):
    batch, pics, vector = x.size()
    concatenation = []
    for group_num in range(int(pics / groups)):
        segments = x[:, groups*group_num: groups*(group_num+1), :]
        segments = torch.sum(segments, dim=1)
        concatenation.append(segments)
    concatenation = torch.cat(concatenation, dim=1)
    return concatenation


class Reasoning(nn.Module):
    def __init__(self, num_segments=4, num_frames=5, num_class=15, img_dim=896, max_group=4, start=2):
        super(Reasoning, self).__init__()
        self.num_segments = num_segments
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_dim
        self.num_groups = max_group
        self.start = start
        self.relation = nn.ModuleList()
        self.classifier_scales = nn.ModuleList()
        num_bottleneck = 512
        for scales in range(self.start, self.num_groups+1):
            fc_fusion = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.img_feature_dim * int(self.num_segments *
                          self.num_frames/scales), num_bottleneck),
                nn.ReLU(),
                nn.Dropout(p=0.6),
                nn.Linear(num_bottleneck, num_bottleneck),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(num_bottleneck, self.num_class),
                nn.Sigmoid(),
                # nn.ReLU(),
                # nn.Dropout(p=0.6),
            )
            self.relation += [fc_fusion]
            # classifier = nn.Linear(num_bottleneck, self.num_class)
            # self.classifier_scales += [classifier]

    def forward(self, x):
        prediction = 0
        for segment_group in range(self.start, self.num_groups+1):
            segments = sum_group(x, groups=segment_group)
            segments = self.relation[segment_group-self.start](segments)
            prediction = prediction + segments
        return prediction / (self.num_groups-self.start+1)
