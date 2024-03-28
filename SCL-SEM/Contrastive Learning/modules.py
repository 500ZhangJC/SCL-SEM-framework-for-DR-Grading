from tkinter import simpledialog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, pretrained=True, head='mlp', K=16384 ,dim_in=2048, feat_dim=128):
        super(ContrastiveModel, self).__init__()

        self.net = backbone(pretrained=pretrained)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
    
    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        x = F.normalize(x, dim=1)
        return x


# from https://github.com/HobbitLong/SupContrast
class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None,if_t=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)    
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]     #feature = 16*2*128    batch * transformer * vector
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)   # 32*128   32 = 2*batchsize
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]   #???
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature     # 32 *128
            anchor_count = contrast_count         # 2 transformer size
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(    #32 * 32 
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability   
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()   #32*32 

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)   #32*32
        # mask-out self-contrast cases
        logits_mask = torch.scatter(      
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask   
        '''
        0   1
          0   1
        1   0
          1   0
        '''

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask   #32*32 

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # return loss,similar_matrix
        return loss


class WarmupLRScheduler():
    def __init__(self, optimizer, warmup_epochs, initial_lr):
        self.epoch = 0
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr

    def step(self):
        if self.epoch <= self.warmup_epochs:
            self.epoch += 1
            curr_lr = (self.epoch / self.warmup_epochs) * self.initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = curr_lr

    def is_finish(self):
        return self.epoch >= self.warmup_epochs


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self,backbone, pretrained=True,head='mlp',dim_in=2048,feat_dim=128, K=14400, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = backbone(pretrained=pretrained)
        self.encoder_k = backbone(pretrained=pretrained)

        if head == 'mlp':  # hack: brute-force replacement
            self.encoder_q = nn.Sequential(self.encoder_q, nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, feat_dim))
            self.encoder_k= nn.Sequential(self.encoder_k, nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, feat_dim))

        elif head == 'linear':
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(feat_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()  #等同于with torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        if self.K % batch_size != 0:
            print("Batchsize and K number dismatch!",self.K,batch_size)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k,im_healthy,is_train=True,is_student=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        
        with torch.no_grad():  # no gradient to keys
            if is_train and is_student:    
                self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            if is_train:   
                h = self.encoder_k(im_healthy)
                h = nn.functional.normalize(h,dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)    #einstan 
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_train:
            self._dequeue_and_enqueue(h)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
