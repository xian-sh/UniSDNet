from terminaltables import AsciiTable
from tqdm import tqdm
import logging
import torch
from dtfnet.data.datasets.utils import iou, score2d_to_moments_scores
from dtfnet.utils.comm import is_main_process


def nms(moments, scores, thresh):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True
    return moments[~suppressed]


def evaluate(cfg, dataset, predictions, nms_thresh, epoch, recall_metrics=(1, 5)):
    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """
    if not is_main_process():
        return
    if cfg.DATASETS.NAME == "tacos":
        iou_metrics = (0.1, 0.3, 0.5)
        table = [['R@1,IoU0.1', 'R@1,IoU0.3', 'R@1,IoU0.5', 'R@5,IoU0.1', 'R@5,IoU0.3', 'R@5,IoU0.5', 'mIoU']]
    elif cfg.DATASETS.NAME == "activitynet":
        iou_metrics = (0.3, 0.5, 0.7)
        table = [['R@1,IoU0.3', 'R@1,IoU0.5', 'R@1,IoU0.7', 'R@5,IoU0.3', 'R@5,IoU0.5', 'R@5,IoU0.7', 'mIoU']]
    elif cfg.DATASETS.NAME == "charades":
        iou_metrics = (0.3, 0.5, 0.7)
        table = [['R@1,IoU0.3', 'R@1,IoU0.5', 'R@1,IoU0.7', 'R@5,IoU0.3', 'R@5,IoU0.5', 'R@5,IoU0.7', 'mIoU']]
    else:
        raise NotImplementedError("No support for %s dataset!" % cfg.DATASETS.NAME)
    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("dtf.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)  # 2, 4
    recall_metrics = torch.tensor(recall_metrics)  # tensor([1, 5])
    iou_metrics = torch.tensor(iou_metrics)  # tensor([0.3000, 0.5000, 0.7000, 0.0000])
    num_clips = predictions[0]['iou'].shape[-1]

    recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)
    recall_m_iou = 0.
    num_instance = 0
    s = []
    res = []
    for idx, result2d in tqdm(enumerate(predictions)):   # each video
        score2d = torch.pow(result2d['contrastive'] * 0.5 + 0.5, cfg.TEST.CONTRASTIVE_SCORE_POW) * result2d['iou']
        duration = dataset.get_duration(idx)
        sentences = dataset.get_sentence(idx)
        gt_moments = dataset.get_moment(idx)
        relults = {
            'idx': idx,
            'vid': dataset.get_vid(idx),
            'duration': duration,
            'sentences': sentences,
            'gt_moments': gt_moments.tolist(),
            'moments': [],
            'scores': [],
        }

        for gt_moment, pred_score2d in zip(gt_moments, score2d):  # each sentence
            num_instance += 1
            candidates, scores = score2d_to_moments_scores(pred_score2d, num_clips, duration)  # candidates: 1104,2
            moments = nms(candidates, scores, nms_thresh)  # 150左右， 2
            relults['moments'].append(moments[0].tolist())
            s1 = iou(moments[:1], gt_moment)
            relults['scores'].append(s1)
            for i, r in enumerate(recall_metrics):
                mious = iou(moments[:r], gt_moment)
                bools = mious[:, None].expand(r, num_iou_metrics) >= iou_metrics
                recall_x_iou[i] += bools.any(dim=0)
                if i == 0:
                    recall_m_iou += mious.item()
        w = torch.stack(relults['scores'], dim=0).mean(dim=0)
        s.append(w)
        res.append(relults)
    x = torch.argsort(torch.stack(s, dim=0), dim=0, descending=True).view(-1)
    import json
    res_new = {}
    for h in x:
        # print(res[h])
        res_new.update({
            f"{res[h]['vid']}": {
                'idx': res[h]['idx'],
                # 'vid': res[h]['vid'],
                'duration': res[h]['duration'],
                'sentences': res[h]['sentences'],
                'gt_moments': res[h]['gt_moments'],
                'pred_moments': res[h]['moments'],
                'pred_scores': res[h]['scores'],
                }
                })
    path = cfg.OUTPUT_DIR + f'/results{epoch}.json'
#     path = None
#     import os
#     for i in range(15):
#         path = file_path.format(i)
#         if os.path.exists(path):
#             continue
#         else:
#             break
    
#     result_path = cfg.OUTPUT_DIR + 'results.json'
    with open(path, "w") as f:
        json.dump(res_new, f, indent=4, separators=(',', ': '))
        
        
    recall_m_iou /= num_instance
    recall_x_iou /= num_instance
    l = ['{:.02f}'.format(recall_x_iou[i][j]*100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)]
    l = l + ['{:.02f}'.format(recall_m_iou*100)]
    table.append(l)
    table = AsciiTable(table)
    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('\n' + table.table)


    result_dict = {}
    for i in range(num_recall_metrics):
        for j in range(num_iou_metrics):
            result_dict['R@{},IoU@{:.01f}'.format(recall_metrics[i], torch.round(iou_metrics[j]*100)/100)] = recall_x_iou[i][j]
    result_dict['R@{1},mIoU'] = recall_m_iou
    best_r1 = sum(recall_x_iou[0])/num_iou_metrics
    best_r5 = sum(recall_x_iou[1])/num_iou_metrics
    result_dict['Best_R1'] = best_r1
    result_dict['Best_R5'] = best_r5
    return result_dict

# def evaluate(cfg, dataset, predictions, nms_thresh, recall_metrics=(1, 5)):
#     """evaluate dataset using different methods based on dataset type.
#     Args:
#     Returns:
#     """
#     if not is_main_process():
#         return
#     if cfg.DATASETS.NAME == "tacos":
#         iou_metrics = (0.1, 0.3, 0.5)
#     elif cfg.DATASETS.NAME == "activitynet":
#         iou_metrics = (0.3, 0.5, 0.7)
#     elif cfg.DATASETS.NAME == "charades":
#         iou_metrics = (0.5, 0.7)
#     else:
#         raise NotImplementedError("No support for %s dataset!" % cfg.DATASETS.NAME)
#     dataset_name = dataset.__class__.__name__
#     logger = logging.getLogger("mmn.inference")
#     logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))
#     num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
#     recall_metrics = torch.tensor(recall_metrics)
#     iou_metrics = torch.tensor(iou_metrics)
#     num_clips = predictions[0]['iou'].shape[-1]
#     table = [['R@{},IoU@{:.01f}'.format(i, torch.round(j*100)/100) for i in recall_metrics for j in iou_metrics]]
#     recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)
#     num_instance = 0
#     for idx, result2d in tqdm(enumerate(predictions)):   # each video
#         score2d = torch.pow(result2d['contrastive'] * 0.5 + 0.5, cfg.TEST.CONTRASTIVE_SCORE_POW) * result2d['iou']
#         duration = dataset.get_duration(idx)
#         gt_moments = dataset.get_moment(idx)
#         for gt_moment, pred_score2d in zip(gt_moments, score2d):  # each sentence
#             num_instance += 1
#             candidates, scores = score2d_to_moments_scores(pred_score2d, num_clips, duration)
#             moments = nms(candidates, scores, nms_thresh)
#             for i, r in enumerate(recall_metrics):
#                 mious = iou(moments[:r], gt_moment)
#                 bools = mious[:, None].expand(r, num_iou_metrics) >= iou_metrics
#                 recall_x_iou[i] += bools.any(dim=0)
#     recall_x_iou /= num_instance
#     table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
#     table = AsciiTable(table)
#     for i in range(num_recall_metrics*num_iou_metrics):
#         table.justify_columns[i] = 'center'
#     logger.info('\n' + table.table)
#     result_dict = {}
#     for i in range(num_recall_metrics):
#         for j in range(num_iou_metrics):
#             result_dict['R@{},IoU@{:.01f}'.format(recall_metrics[i], torch.round(iou_metrics[j]*100)/100)] = recall_x_iou[i][j]
#     best_r1 = sum(recall_x_iou[0])/num_iou_metrics
#     best_r5 = sum(recall_x_iou[1])/num_iou_metrics
#     result_dict['Best_R1'] = best_r1
#     result_dict['Best_R5'] = best_r5
#     return result_dict

