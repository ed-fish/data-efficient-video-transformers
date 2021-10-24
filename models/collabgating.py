
class CollaborativeGating(pl.LightningModule):
    def __init__(self):
        super(CollaborativeGating, self).__init__()
        self.proj_input = 2048
        self.proj_embedding_size = 2048
        self.projection = nn.Linear(self.proj_input, self.proj_embedding_size)
        self.cg = ContextGating(self.proj_input)
        self.geu = GatedEmbeddingUnit(self.proj_input, 1024,  False)

    def pad(self, tensor):
        tensor = tensor.unsqueeze(0)
        curr_expert = F.interpolate(tensor, 2048)
        curr_expert = curr_expert.squeeze(0)
        return curr_expert

    def forward(self, batch):
        batch_list = []
        for scenes in batch:  # this will be batches
            scene_list = []
            # first expert popped off
            for experts in scenes:
                expert_attention_vec = []
                for i in range(len(experts)):
                    curr_expert = experts.pop(0)
                    if curr_expert.shape[1] != 2048:
                        curr_expert = self.pad(curr_expert)

                    # compare with all other experts
                    curr_expert = self.projection(curr_expert)
                    t_i_list = []
                    for c_expert in experts:
                        # through g0 to get feature embedding t_i
                        if c_expert.shape[1] != 2048:
                            c_expert = self.pad(c_expert)
                        c_expert = self.projection(c_expert)
                        t_i = curr_expert + c_expert  # t_i maps y1 to y2
                        t_i_list.append(t_i)
                    t_i_summed = torch.stack(t_i_list, dim=0).sum(
                        dim=0)  # all other features
                    # attention vector for all comparrisons
                    expert_attention = self.projection(t_i_summed)
                    expert_attention_comp = self.cg(
                        curr_expert, expert_attention)  # gated version
                    expert_attention_vec.append(expert_attention_comp)
                    experts.append(curr_expert)
                expert_attention_vec = torch.stack(expert_attention_vec, dim=0).sum(
                    dim=0)  # concat all attention vectors
                # apply gated embedding
                expert_vector = self.geu(expert_attention_vec)
                scene_list.append(expert_vector)
            scene_stack = torch.stack(scene_list)
            batch_list.append(scene_stack)
        batch = torch.stack(batch_list, dim=0)
        batch = batch.squeeze(2)
        return batch


class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension, use_bn):
        super(GatedEmbeddingUnit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        # self.cg = ContextGating(output_dimension, add_batch_norm=use_bn)

    def forward(self, x):
        x = self.fc(x)
        # x = self.cg(x)
        x = F.normalize(x)
        return x


class ContextGating(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(ContextGating, self).__init__()
        # self.add_batch_norm = add_batch_norm
        # self.batch_norm = nn.BatchNorm1d(dimension)
        # self.batch_norm2 = nn.BatchNorm1d(dimension)

    def forward(self, x, x1):

        # if self.add_batch_norm:
        #     x = self.batch_norm(x)
        #     x1 = self.batch_norm2(x1)
        t = x + x1
        x = torch.cat((x, t), -1)
        return F.glu(x, -1)
