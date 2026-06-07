import pandas as pd
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ================== 1. 读取干扰信息 ==================
csv_path = "corruption_info.csv"
df = pd.read_csv(csv_path)

# ================== 2. 读取 COCO 标注 ==================
ann_path = "corrupted_annotations.json"  # 你的噪声测试集标注
coco_gt = COCO(ann_path)


# 假设你有一个函数返回预测结果 JSON 文件路径：
# 例如 predict_results/{filename}.json
def get_prediction_file(filename):
    return f"predict_results/{filename}.json"


# ================== 3. 计算每张图片 mAP ==================
mAP_list = []

for idx, row in df.iterrows():
    filename = row['filename']
    pred_file = get_prediction_file(filename)

    if not Path(pred_file).exists():
        print(f"预测文件不存在: {pred_file}")
        mAP_list.append(None)
        continue

    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    # 只评估当前图片
    img_id = coco_gt.getImgIds(imgIds=[int(Path(filename).stem)])  # 注意 img_id 对应你 COCO JSON
    coco_eval.params.imgIds = img_id

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 保存 mAP@0.5
    mAP_list.append(coco_eval.stats[1] * 100)  # stats[1] 是 AP@0.5

df['mAP@0.5'] = mAP_list

# ================== 4. 保存带 mAP 的 CSV ==================
df.to_csv("corruption_info_with_map.csv", index=False)
print("✅ 已生成带 mAP 的 CSV，可以直接画图！")