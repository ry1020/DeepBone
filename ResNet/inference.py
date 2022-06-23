import time
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from utils import AverageMeter


def inference(data_loader, model, result_path, class_names,
              output_topk):
    print('inference')

    model.eval()

    results = {'results': defaultdict(list)}

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):

            outputs = model(inputs)

            for j in range(outputs.size(0)):
                results['results'][video_ids[j]].append({
                    'segment': segments[j],
                    'output': outputs[j]
                })

            print('[{}/{}]\t'.format(
                      i + 1,
                      len(data_loader)))

    inference_results = {'results': {}}
    if not no_average:
        for video_id, video_results in results['results'].items():
            video_outputs = [
                segment_result['output'] for segment_result in video_results
            ]
            video_outputs = torch.stack(video_outputs)
            average_scores = torch.mean(video_outputs, dim=0)
            inference_results['results'][video_id] = get_video_results(
                average_scores, class_names, output_topk)
    else:
        for video_id, video_results in results['results'].items():
            inference_results['results'][video_id] = []
            for segment_result in video_results:
                segment = segment_result['segment']
                result = get_video_results(segment_result['output'],
                                           class_names, output_topk)
                inference_results['results'][video_id].append({
                    'segment': segment,
                    'result': result
                })

    with result_path.open('w') as f:
        json.dump(inference_results, f)
