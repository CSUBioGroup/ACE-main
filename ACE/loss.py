import torch
import torch.nn.functional as F

class InfoNCE_s(torch.nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, 
                                                           dtype=bool)).float())
            
    def forward(self, simi):
        simi_max, _ = torch.max(simi, dim=1, keepdim=True)
        simi = simi - simi_max.detach()

        sim_ij = torch.diag(simi, self.batch_size)
        sim_ji = torch.diag(simi, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        negative_logits = (torch.exp(simi) * self.negatives_mask).sum(dim=1)
        loss = -(positives - torch.log(negative_logits)).mean()
        
        return loss

class InfoNCE_s3(torch.nn.Module):
    def __init__(self, batch_size, rep=3):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * rep, batch_size * rep, 
                                                           dtype=bool)).float())
            
    def forward(self, simi, iids):
        simi_max, _ = torch.max(simi, dim=1, keepdim=True)
        simi = simi - simi_max.detach()

        pos_mask = (iids.view(-1, 1) == iids.view(1, -1)).float()
        pos_mask *= self.negatives_mask

        positives = (simi * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
        negatives = (torch.exp(simi) * self.negatives_mask).sum(dim=1)
        loss = -(positives - torch.log(negatives)).mean()

        return loss

class Npos_NCE(torch.nn.Module):
    def __init__(self, batch_size, rep=3):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * rep, batch_size * rep, 
                                                           dtype=bool)).float())
            
    def forward(self, simi, iids):
        simi_max, _ = torch.max(simi, dim=1, keepdim=True)
        simi = simi - simi_max.detach()

        pos_mask = (iids.view(-1, 1) == iids.view(1, -1)).float()
        pos_mask *= self.negatives_mask

        positives = (simi * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
        negatives = (torch.exp(simi) * self.negatives_mask).sum(dim=1)
        loss = -(positives - torch.log(negatives)).mean()

        return loss
    