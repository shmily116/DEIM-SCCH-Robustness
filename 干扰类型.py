import cv2
import numpy as np
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import copy


# ==================== 1. 干扰类定义 ====================
class MaritimeCorruption:
    """海事图像干扰生成器"""

    def __init__(self, severity: int = 3):
        self.severity = max(1, min(5, severity))

    def gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        sigma = [5, 10, 15, 20, 25][self.severity - 1]
        noise = np.random.normal(0, sigma, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def motion_blur(self, img: np.ndarray) -> np.ndarray:
        kernel_size = [5, 9, 13, 17, 21][self.severity - 1]
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(img, -1, kernel)

    def brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        alpha = [0.7, 0.5, 0.35, 0.25, 0.15][self.severity - 1]
        beta = [-30, -50, -70, -90, -110][self.severity - 1]
        return np.clip(img * alpha + beta, 0, 255).astype(np.uint8)

    def defocus_blur(self, img: np.ndarray) -> np.ndarray:
        kernel_size = [3, 5, 7, 9, 11][self.severity - 1]
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def lens_droplet(self, img: np.ndarray) -> np.ndarray:
        result = img.copy()
        h, w = img.shape[:2]
        num_drops = [3, 7, 12, 18, 25][self.severity - 1]
        for _ in range(num_drops):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(8, 20)
            overlay = result.copy()
            cv2.circle(overlay, (x, y), radius, (180, 200, 220), -1)
            result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
            cv2.circle(result, (x - radius // 3, y - radius // 3), radius // 5, (255, 255, 255), -1)
        return result

    def sea_glare(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        intensity = [0.3, 0.5, 0.7, 0.85, 0.95][self.severity - 1]
        glare_mask = np.zeros((h, w), dtype=np.float32)
        water_region = h // 2
        for _ in range(self.severity):
            center_x = np.random.randint(w // 4, 3 * w // 4)
            center_y = np.random.randint(water_region, h)
            radius = np.random.randint(w // 6, w // 3)
            cv2.circle(glare_mask, (center_x, center_y), radius, 1.0, -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (51, 51), 0)
        glare_mask_3ch = np.stack([glare_mask, glare_mask, glare_mask], axis=2)
        img_float = img.astype(np.float32)
        glare_float = np.ones_like(img_float) * 255.0
        alpha = intensity * glare_mask_3ch
        result = img_float * (1 - alpha) + glare_float * alpha
        return np.clip(result, 0, 255).astype(np.uint8)

    def rain_snow(self, img: np.ndarray) -> np.ndarray:
        result = img.copy()
        h, w = img.shape[:2]
        num_lines = [30, 80, 150, 250, 400][self.severity - 1]
        for _ in range(num_lines):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            length = np.random.randint(15, 40)
            angle = np.random.choice([-30, -20, -10, 0, 10, 20, 30])
            rad = np.radians(angle)
            x2 = int(x1 + length * np.cos(rad))
            y2 = int(y1 + length * np.sin(rad))
            color = (200, 210, 230) if self.severity < 4 else (255, 255, 255)
            cv2.line(result, (x1, y1), (x2, y2), color, 2)
        return result

    def sea_fog(self, img: np.ndarray) -> np.ndarray:
        fog_density = [0.2, 0.35, 0.5, 0.65, 0.8][self.severity - 1]
        h, w = img.shape[:2]
        result = img.copy().astype(np.float32)
        fog = np.ones_like(result, dtype=np.float32) * 255.0
        for i in range(h):
            depth_factor = i / h
            alpha = fog_density * depth_factor
            result[i, :] = result[i, :] * (1 - alpha) + fog[i, :] * alpha
        result = np.clip(result, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1] * (1 - fog_density * 0.5)).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def wave_reflection(self, img: np.ndarray, return_maps: bool = False) -> np.ndarray:
        """波浪反射，可选返回位移场用于标签变换"""
        h, w = img.shape[:2]
        wave_strength = [5, 10, 15, 20, 25][self.severity - 1]

        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        wave_x = wave_strength * np.sin(2 * np.pi * map_y / (h / 4))
        wave_y = wave_strength * 0.5 * np.cos(2 * np.pi * map_x / (w / 6))

        map_x = map_x + wave_x
        map_y = map_y + wave_y

        result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        num_highlights = [20, 50, 100, 180, 300][self.severity - 1]
        for _ in range(num_highlights):
            x, y = np.random.randint(0, w), np.random.randint(h // 2, h)
            cv2.circle(result, (x, y), np.random.randint(1, 4), (255, 255, 245), -1)

        if return_maps:
            return result, map_x, map_y
        return result

    def apply(self, img: np.ndarray, corruption_type: str, return_maps: bool = False):
        """统一调用接口"""
        if corruption_type == 'wave_reflection' and return_maps:
            return self.wave_reflection(img, return_maps=True)

        methods = {
            'gaussian_noise': self.gaussian_noise,
            'motion_blur': self.motion_blur,
            'brightness_contrast': self.brightness_contrast,
            'defocus_blur': self.defocus_blur,
            'lens_droplet': self.lens_droplet,
            'sea_glare': self.sea_glare,
            'rain_snow': self.rain_snow,
            'sea_fog': self.sea_fog,
            'wave_reflection': self.wave_reflection,
        }
        return methods[corruption_type](img)


# ==================== 2. 标签变换函数 ====================
def warp_bbox(bbox: List[float], map_x: np.ndarray, map_y: np.ndarray, img_w: int, img_h: int) -> List[float]:
    """
    对单个边界框进行变换（COCO格式: [x, y, width, height]）
    将边界框的四个角点映射后重新计算最小外接矩形
    """
    x, y, w, h = bbox

    # 边界框的四个角点
    corners = np.array([
        [x, y],
        [x + w, y],
        [x, y + h],
        [x + w, y + h]
    ], dtype=np.float32)

    # 映射角点
    mapped_corners = []
    for cx, cy in corners:
        # 确保坐标在有效范围内
        cx = np.clip(cx, 0, img_w - 1)
        cy = np.clip(cy, 0, img_h - 1)
        new_x = map_x[int(cy), int(cx)]
        new_y = map_y[int(cy), int(cx)]
        mapped_corners.append([new_x, new_y])

    mapped_corners = np.array(mapped_corners)

    # 计算新的边界框
    min_x = np.min(mapped_corners[:, 0])
    min_y = np.min(mapped_corners[:, 1])
    max_x = np.max(mapped_corners[:, 0])
    max_y = np.max(mapped_corners[:, 1])

    # 确保边界框在图像范围内
    min_x = np.clip(min_x, 0, img_w)
    min_y = np.clip(min_y, 0, img_h)
    max_x = np.clip(max_x, 0, img_w)
    max_y = np.clip(max_y, 0, img_h)

    new_w = max_x - min_x
    new_h = max_y - min_y

    # 过滤无效边界框
    if new_w <= 0 or new_h <= 0:
        return None

    return [float(min_x), float(min_y), float(new_w), float(new_h)]


def transform_coco_annotations(annotations: List[Dict], map_x: np.ndarray, map_y: np.ndarray,
                               img_w: int, img_h: int) -> List[Dict]:
    """变换COCO格式的所有标注"""
    new_annotations = []
    for ann in annotations:
        new_bbox = warp_bbox(ann['bbox'], map_x, map_y, img_w, img_h)
        if new_bbox is not None:
            new_ann = copy.deepcopy(ann)
            new_ann['bbox'] = new_bbox
            # 更新面积
            new_ann['area'] = new_bbox[2] * new_bbox[3]
            new_annotations.append(new_ann)
    return new_annotations


# ==================== 3. 随机干扰生成函数 ====================
def apply_random_corruption_with_labels(
        img: np.ndarray,
        annotations: List[Dict],
        corruption_types: List[str],
        severity_range: Tuple[int, int] = (1, 5),
        num_corruptions: Tuple[int, int] = (1, 2)
) -> Tuple[np.ndarray, List[Dict], List[str], List[int]]:
    """
    对单张图片随机应用1-2种干扰，并同步变换标注
    """
    n = random.randint(num_corruptions[0], num_corruptions[1])
    selected_types = random.sample(corruption_types, min(n, len(corruption_types)))
    selected_severities = [random.randint(severity_range[0], severity_range[1]) for _ in selected_types]

    corrupted = img.copy()
    current_annotations = copy.deepcopy(annotations)
    h, w = img.shape[:2]

    for corr_type, severity in zip(selected_types, selected_severities):
        generator = MaritimeCorruption(severity=severity)

        if corr_type == 'wave_reflection':
            # 波浪反射需要变换标注
            corrupted, map_x, map_y = generator.apply(corrupted, corr_type, return_maps=True)
            current_annotations = transform_coco_annotations(current_annotations, map_x, map_y, w, h)
        else:
            # 其他干扰不影响标注位置
            corrupted = generator.apply(corrupted, corr_type)

    return corrupted, current_annotations, selected_types, selected_severities


def generate_random_corrupted_dataset(
        img_dir: str,
        json_path: str,
        output_dir: str,
        corruption_types: List[str],
        severity_range: Tuple[int, int] = (1, 5),
        num_corruptions: Tuple[int, int] = (1, 2),
        save_info: bool = True
):
    """
    为测试集生成随机干扰版本，同步变换JSON标注

    Args:
        img_dir: 原始测试图像目录
        json_path: 原始COCO格式JSON标注文件路径
        output_dir: 输出目录
        corruption_types: 干扰类型列表
        severity_range: 严重等级范围
        num_corruptions: 干扰数量范围
        save_info: 是否保存干扰信息
    """
    # 加载原始JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 构建图像ID到文件名的映射
    img_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    img_id_to_info = {img['id']: img for img in coco_data['images']}

    # 按图像ID分组标注
    img_id_to_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_annotations:
            img_id_to_annotations[img_id] = []
        img_id_to_annotations[img_id].append(ann)

    # 获取所有测试图像
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    test_images = []
    for ext in img_extensions:
        test_images.extend(Path(img_dir).glob(ext))

    print(f"找到 {len(test_images)} 张测试图像")
    print(f"JSON中包含 {len(coco_data['images'])} 张图像信息")

    if len(test_images) == 0:
        print("错误：未找到任何图像文件，请检查路径！")
        return

    # 创建输出目录
    output_path = Path(output_dir)
    corrupted_img_dir = output_path / "corrupted_images"
    corrupted_img_dir.mkdir(parents=True, exist_ok=True)

    # 准备新的COCO数据
    new_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': [],
        'annotations': []
    }

    # 记录干扰信息
    info_records = []
    next_ann_id = 1

    # 处理每张图片
    for img_path in tqdm(test_images, desc="生成随机干扰图像"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告：无法读取图像 {img_path}，跳过")
            continue

        # 获取对应的图像ID（通过文件名匹配）
        img_filename = img_path.name
        img_id = None
        for fid, fname in img_id_to_filename.items():
            if fname == img_filename:
                img_id = fid
                break

        if img_id is None:
            print(f"警告：未找到图像 {img_filename} 的标注，跳过")
            continue

        # 获取该图像的标注
        original_annotations = img_id_to_annotations.get(img_id, [])

        # 应用随机干扰并变换标注
        corrupted, transformed_annotations, types, severities = apply_random_corruption_with_labels(
            img, original_annotations, corruption_types, severity_range, num_corruptions
        )

        # 保存干扰图像
        save_path = corrupted_img_dir / img_filename
        cv2.imwrite(str(save_path), corrupted)

        # 更新COCO图像信息（保持原尺寸，因为干扰不改变尺寸）
        img_info = copy.deepcopy(img_id_to_info[img_id])
        img_info['file_name'] = img_filename  # 保持原文件名
        new_coco_data['images'].append(img_info)

        # 更新标注
        for ann in transformed_annotations:
            new_ann = copy.deepcopy(ann)
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = img_id
            new_coco_data['annotations'].append(new_ann)
            next_ann_id += 1

        # 记录信息
        info_records.append({
            'filename': img_filename,
            'corruption_types': '+'.join(types),
            'severities': '+'.join(map(str, severities)),
            'num_corruptions': len(types),
            'original_annotations': len(original_annotations),
            'transformed_annotations': len(transformed_annotations)
        })

    # 保存新的JSON标注文件
    new_json_path = output_path / "corrupted_annotations.json"
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_coco_data, f, indent=2)

    # 保存干扰信息
    if save_info and info_records:
        import csv
        csv_path = output_path / "corruption_info.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'corruption_types', 'severities',
                                                   'num_corruptions', 'original_annotations',
                                                   'transformed_annotations'])
            writer.writeheader()
            writer.writerows(info_records)

        # 打印统计信息
        print(f"\n📊 干扰统计:")
        print(f"   - 总图像数: {len(info_records)}")

        single = sum(1 for r in info_records if r['num_corruptions'] == 1)
        double = sum(1 for r in info_records if r['num_corruptions'] == 2)
        print(f"   - 单干扰: {single} 张 ({single / len(info_records) * 100:.1f}%)")
        print(f"   - 双干扰: {double} 张 ({double / len(info_records) * 100:.1f}%)")

        type_counts = {}
        for r in info_records:
            for t in r['corruption_types'].split('+'):
                type_counts[t] = type_counts.get(t, 0) + 1
        print(f"\n   - 各干扰类型出现次数:")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"       {t}: {count} 次 ({count / len(info_records) * 100:.1f}%)")

        total_orig = sum(r['original_annotations'] for r in info_records)
        total_trans = sum(r['transformed_annotations'] for r in info_records)
        print(f"\n   - 标注统计:")
        print(f"       原始标注总数: {total_orig}")
        print(f"       变换后标注总数: {total_trans}")
        if total_orig > 0:
            print(f"       标注保留率: {total_trans / total_orig * 100:.1f}%")

    print(f"\n✅ 生成完成！")
    print(f"   输出目录: {output_path}")
    print(f"   干扰图像: {corrupted_img_dir}")
    print(f"   新JSON标注: {new_json_path}")


# ==================== 4. 复制干净图像和标注 ====================
def copy_clean_data(img_dir: str, json_path: str, output_dir: str):
    """复制干净图像和标注到输出目录"""
    output_path = Path(output_dir)
    clean_img_dir = output_path / "clean_images"
    clean_img_dir.mkdir(parents=True, exist_ok=True)

    # 复制图像
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    count = 0
    for ext in img_extensions:
        for img_path in Path(img_dir).glob(ext):
            shutil.copy2(img_path, clean_img_dir / img_path.name)
            count += 1

    # 复制JSON
    clean_json_path = output_path / "clean_annotations.json"
    shutil.copy2(json_path, clean_json_path)

    print(f"✅ 已复制 {count} 张干净图像到: {clean_img_dir}")
    print(f"✅ 已复制标注文件到: {clean_json_path}")


# ==================== 5. 主程序入口 ====================
if __name__ == "__main__":
    # ===== 请修改以下路径为您的实际路径 =====

    # 原始测试集目录（存放测试图像）
    TEST_IMG_DIR = r"E:/Seadronesee/whole/images/test"  # ← 请修改

    # 原始COCO格式JSON标注文件路径
    TEST_JSON_PATH = r"E:/Seadronesee/whole/annotations/test.json"  # ← 请修改

    # 输出目录
    OUTPUT_DIR = r"E:/Seadronesee/whole/images/test-bad"

    # 9种干扰类型
    CORRUPTION_TYPES = [
        'gaussian_noise', 'motion_blur', 'brightness_contrast',
        'defocus_blur', 'lens_droplet', 'sea_glare',
        'rain_snow', 'sea_fog', 'wave_reflection'
    ]

    # 配置
    SEVERITY_RANGE = (1, 5)  # 严重等级范围
    NUM_CORRUPTIONS = (1, 2)  # 每张图片随机1-2种干扰

    # 可选：是否同时复制干净数据
    COPY_CLEAN = True

    # 执行生成
    generate_random_corrupted_dataset(
        img_dir=TEST_IMG_DIR,
        json_path=TEST_JSON_PATH,
        output_dir=OUTPUT_DIR,
        corruption_types=CORRUPTION_TYPES,
        severity_range=SEVERITY_RANGE,
        num_corruptions=NUM_CORRUPTIONS,
        save_info=True
    )

    # 可选：复制干净数据作为对比
    if COPY_CLEAN:
        copy_clean_data(TEST_IMG_DIR, TEST_JSON_PATH, OUTPUT_DIR)

    print("\n🎉 全部完成！")
    print(f"   输出结构:")
    print(f"   {OUTPUT_DIR}/")
    print(f"   ├── clean_images/              # 原始干净图像")
    print(f"   ├── clean_annotations.json     # 原始标注")
    print(f"   ├── corrupted_images/          # 随机干扰后的图像")
    print(f"   ├── corrupted_annotations.json # 变换后的标注")
    print(f"   └── corruption_info.csv        # 干扰信息记录")