from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import convert_dt as dt
import convert_gt as gt

gt.run()
dt.run()

annoFile = "./data/gt_val.json"
cocoGT = COCO(annoFile)

resFile = "./data/result_val.json"
cocoDT = cocoGT.loadRes(resFile)

imgIds = sorted(cocoGT.getImgIds())

annType = 'bbox'
cocoEval = COCOeval(cocoGT, cocoDT, annType)
cocoEval.params.imgIds = imgIds
cocoEval.params.maxDets = [1, 100, 1000]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()