from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path

anno_json = str(Path("./anno.json"))
pred_json = str(Path("./results.json"))

anno = COCO(anno_json)
pred = anno.loadRes(pred_json)
eval = COCOeval(anno, pred, 'bbox')
eval.evaluate()
eval.accumulate()
eval.summarize()
map, map50 = eval.stats[:2] # update results (mAP@0.5:0.95, mAP@0.5)

print(eval.stats)