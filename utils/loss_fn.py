import torch
import torch.nn as nn


'''
    Define some loss functions
'''

class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))
    

# ------------------------
# Contrastive Loss (InfoNCE)
# ------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_feat, text_feat):
        image_feat = nn.functional.normalize(image_feat, dim=1)
        text_feat = nn.functional.normalize(text_feat, dim=1)
        logits_per_image = torch.matmul(image_feat, text_feat.t()) / self.temperature
        logits_per_text = logits_per_image.t()
        targets = torch.arange(image_feat.size(0)).to(image_feat.device)
        loss_i2t = nn.functional.cross_entropy(logits_per_image, targets)
        loss_t2i = nn.functional.cross_entropy(logits_per_text, targets)
        return (loss_i2t + loss_t2i) / 2