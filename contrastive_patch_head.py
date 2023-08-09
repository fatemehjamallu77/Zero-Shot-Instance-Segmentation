class PatchDCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.nce_T = 0.07
    def forward(self, feat_q, feat_k,weight=None):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        batch_dim_for_bmm=1
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        if weight is not None:
            l_neg_curbatch *= weight

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        logits = (l_neg-l_pos)/self.opt.nce_T
        v = torch.logsumexp(logits, dim=1)
        loss_vec = torch.exp(v-v.detach())

        # for exp
        out_dummy = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        CELoss_dummy = self.cross_entropy_loss(out_dummy, torch.zeros(out_dummy.size(0), dtype=torch.long, device=feat_q.device))

        loss = loss_vec.mean()-1+CELoss_dummy.detach()

        return self.opt.lambda_dce*loss
