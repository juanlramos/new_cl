import torch
import torch.nn as nn
import torchvision.models as models

class MoCo(nn.Module):

    def __init__(
        self,
        base_encoder,
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        mlp=False,
        pretrained=False
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        mlp: False = MoCo v1 / True = MoCo v2
        pretrained: utiliza pesos pré-treinados da ImageNet
        """

        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # Cria os dois encoders
        if pretrained:
            self.encoder_q = base_encoder(
                weights=models.ResNet50_Weights.DEFAULT
            )

            self.encoder_k = base_encoder(
                weights=models.ResNet50_Weights.DEFAULT
            )

            # Substitui a camada final da ResNet50
            # para produzir vetores de dimensão 'dim'
            self.encoder_q.fc = nn.Linear(
                self.encoder_q.fc.in_features,
                dim
            )

            self.encoder_k.fc = nn.Linear(
                self.encoder_k.fc.in_features,
                dim
            )
        else:
            self.encoder_q = base_encoder(num_classes=dim)
            self.encoder_k = base_encoder(num_classes=dim)

        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]

            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                self.encoder_q.fc
            )

            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Cria a fila
        self.register_buffer(
            "queue",
            torch.randn(dim, K)
        )

        self.queue = nn.functional.normalize(
            self.queue,
            dim=0
        )

        self.register_buffer(
            "queue_ptr",
            torch.zeros(1, dtype=torch.long)
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[0])
        assert self.K % batch_size == 0  # K deve ser divisível pelo batch_size
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        # Utilizada somente em treinamento distribuído (DDP).
        # Não utilizada no treinamento com uma única GPU.
        
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        # Utilizada somente em treinamento distribuído (DDP).
        # Não utilizada no treinamento com uma única GPU.
        
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        # Query features
        q = self.encoder_q(im_q)  
        q = nn.functional.normalize(q, dim=1)

        # Key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k) 
            k = nn.functional.normalize(k, dim=1)

        # Calcula os Logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(
            logits.shape[0],
            dtype=torch.long,
            device=logits.device
        )

        self._dequeue_and_enqueue(k)
        return logits, labels


@torch.no_grad()
def concat_all_gather(tensor):
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]

    torch.distributed.all_gather(
        tensors_gather,
        tensor,
        async_op=False
    )

    return torch.cat(tensors_gather, dim=0)