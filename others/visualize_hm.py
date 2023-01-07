import cv2
import torch

from datasets import SampleHeatMapDataset
from model import CNN18_Backbone


def select_topk(heat_map, K=100):
    """
    Args:
        heat_map: heat_map in [N, C, H, W]
        K: top k samples to be selected
        score: detection threshold
    Returns:
    """
    batch, cls, height, width = heat_map.size()

    # First select topk scores in all classes and batchs
    # [N, C, H, W] -----> [N, C, H*W]
    heat_map = heat_map.view(batch, cls, -1)
    # Both in [N, C, K]
    topk_scores_all, topk_inds_all = torch.topk(heat_map, K)

    # topk_inds_all = topk_inds_all % (height * width) # todo: this seems redudant
    topk_ys = (topk_inds_all / width).float()
    topk_xs = (topk_inds_all % width).float()

    # assert isinstance(topk_xs, torch.cuda.FloatTensor)
    # assert isinstance(topk_ys, torch.cuda.FloatTensor)

    # Select topK examples across channel
    # [N, C, K] -----> [N, C*K]
    topk_scores_all = topk_scores_all.view(batch, -1)
    # Both in [N, K]
    topk_scores, topk_inds = torch.topk(topk_scores_all, K)
    topk_clses = (topk_inds / K).float()

    # assert isinstance(topk_clses, torch.cuda.FloatTensor)

    # First expand it as 3 dimension
    topk_inds_all = _gather_feat(topk_inds_all.view(batch, -1, 1), topk_inds).view(
        batch, K
    )
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_inds).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_inds).view(batch, K)

    return topk_scores, topk_inds_all, topk_clses, topk_ys, topk_xs


def _gather_feat(feat, ind):
    """
    Select specific indexs on featuremap
    Args:
        feat: all results in 3 dimensions
        ind: positive index
    Returns:
    """
    channel = feat.size(-1)
    ind = ind.unsqueeze(-1).expand(ind.size(0), ind.size(1), channel)
    feat = feat.gather(1, ind)

    return feat


if __name__ == "__main__":
    n_samples = 5
    n_dim = 4
    image_size = 224 // 4

    model = CNN18_Backbone(3)  # FCNN()
    model.load_state_dict(torch.load("./weights.pth"))
    model.eval()
    dataset = SampleHeatMapDataset(path="./dataset/5_XYHW/train")
    x, y = dataset.__getitem__(0)
    y_pred = model(x.unsqueeze(0))
    # print(y_pred.shape)

    output_img = y_pred[0][0].detach().numpy()
    topk_scores, topk_inds_all, topk_clses, topk_ys, topk_xs = select_topk(y_pred, K=5)
    # print(topk_scores)
    # print(topk_inds_all)
    # print(topk_clses)

    for i in topk_inds_all.numpy()[0]:
        print(f"Row: {i//image_size/image_size}, Col: {i%image_size/image_size}")

    cv2.imshow("image", output_img)
    cv2.waitKey(10000)
