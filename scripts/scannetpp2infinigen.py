# -*- coding: utf-8 -*-
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, List, Union, Dict
import numpy as np
from PIL import Image

def get_image_size_pillow(img_path: str or Path) -> tuple[int, int] or None:
    """
    使用Pillow读取图片宽高（width, height）
    :param img_path: 图片路径
    :return: (宽度, 高度) 或 None（读取失败）
    """
    img_path = str(img_path)
    try:
        with Image.open(img_path) as img:
            width, height = img.size  # Pillow直接返回 (宽, 高)
            return (width, height)
    except FileNotFoundError:
        logger.warning(f"图片文件不存在：{img_path}")
        return None
    except UnidentifiedImageError:
        logger.error(f"Pillow无法识别图片格式：{img_path}")
        return None
    except Exception as e:
        logger.error(f"读取图片尺寸失败（Pillow）：{img_path} | 错误：{e}")
        return None

# ===================== 全局配置 =====================
# 原始数据根目录
RAW_ROOT_DIR = "./scannetpp_sampled"
# 输出数据根目录
OUTPUT_ROOT_DIR = "./scannetpp_sampled_modified"
# 标注文件名配置
ANNO_CONFIG = {
    "iphone": "obj_annotation.json",
    "dslr": "obj_annotation_dslr.json"
}
# 图片文件夹名配置
IMG_DIR_CONFIG = {
    "iphone": "iphone",
    "dslr": "dslr"
}
# ===================== 配置结束 =====================

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ScanNetPPDataProcessor:
    """ScanNet++数据处理类：读取标注、修改结构、迁移图片、分目录存储"""

    def __init__(self, raw_root: str = RAW_ROOT_DIR, output_root: str = OUTPUT_ROOT_DIR):
        """
        初始化处理器
        :param raw_root: 原始数据根目录
        :param output_root: 输出数据根目录
        """
        self.raw_root = Path(raw_root)
        self.output_root = Path(output_root)
        # 确保输出根目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        # 获取所有场景目录
        self.scene_dirs: List[Path] = self._get_valid_scene_dirs()
        logger.info(f"初始化完成，共发现 {len(self.scene_dirs)} 个有效场景")

    def _get_valid_scene_dirs(self) -> List[Path]:
        """获取原始目录下所有有效的场景文件夹（仅保留目录）"""
        if not self.raw_root.exists():
            logger.error(f"原始根目录不存在：{self.raw_root}")
            return []
        scene_dirs = [d for d in self.raw_root.iterdir() if d.is_dir()]
        return scene_dirs

    def _load_json(self, file_path: Path) -> Optional[Union[dict, list]]:
        """安全读取JSON文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败：{file_path} | 错误：{str(e)[:100]}")
            return None
        except FileNotFoundError:
            logger.warning(f"JSON文件不存在：{file_path}")
            return None
        except Exception as e:
            logger.error(f"读取JSON失败：{file_path} | 错误：{str(e)[:100]}")
            return None

    def _save_json(self, data: Union[dict, list], file_path: Path) -> bool:
        """安全保存JSON文件"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    ensure_ascii=False,
                    indent=4
                )
            return True
        except Exception as e:
            logger.error(f"保存JSON失败：{file_path} | 错误：{str(e)[:100]}")
            return False
        
    def _convert_obb_to_aabb_format(
        self,
        centroid: List[float],
        axes_lengths: List[float],
        normalized_axes: List[float],
        coord_precision: int = 6
    ) -> Optional[Dict]:
        """
        将OBB标注格式转换为目标AABB格式
        :param centroid: OBB中心坐标 [x, y, z]（对应原anno['objects'][obj_id]['obb']['centroid']）
        :param axes_lengths: OBB轴长度 [l1, l2, l3]（对应原anno['objects'][obj_id]['obb']['axesLengths']）
        :param normalized_axes: OBB归一化旋转矩阵（9个值，对应原anno['objects'][obj_id]['obb']['normalizedAxes']）
        :param coord_precision: 坐标精度（保留小数位数，避免浮点冗余）
        :return: 转换后的字典（None表示输入无效）
        """
        # ========== 1. 输入校验 ==========
        # 检查输入长度
        if len(centroid) != 3:
            print(f"错误：中心坐标长度无效，需3个值，实际{len(centroid)}个 → {centroid}")
            return None
        if len(axes_lengths) != 3:
            print(f"错误：轴长度无效，需3个值，实际{len(axes_lengths)}个 → {axes_lengths}")
            return None
        if len(normalized_axes) != 9:
            print(f"错误：旋转矩阵无效，需9个值，实际{len(normalized_axes)}个 → {normalized_axes}")
            return None
        
        # 转换为numpy数组（方便矩阵运算）
        try:
            centroid_np = np.array(centroid, dtype=np.float64)
            axes_lengths_np = np.array(axes_lengths, dtype=np.float64)
            # 旋转矩阵reshape为3x3（每行对应local_x/y/z）
            rotation_matrix = np.array(normalized_axes, dtype=np.float64).reshape(3, 3)
        except ValueError as e:
            print(f"错误：输入数值转换失败 → {e}")
            return None

        # ========== 2. 构建3d_center（直接映射，保留精度） ==========
        # 保留指定小数位数，避免浮点冗余
        center_3d = [round(x, coord_precision) for x in centroid_np.tolist()]

        # ========== 3. 构建axis_directions（局部坐标系三轴） ==========
        # normalized_axes是3x3旋转矩阵，每行对应local_x/local_y/local_z的方向向量
        axis_directions = {
            "local_x": [round(x, coord_precision) for x in rotation_matrix[0].tolist()],
            "local_y": [round(x, coord_precision) for x in rotation_matrix[1].tolist()],
            "local_z": [round(x, coord_precision) for x in rotation_matrix[2].tolist()]
        }

        # ========== 4. 计算AABB包围盒（min/max/dimensions） ==========
        # 步骤1：计算OBB的半长（轴长度/2）
        half_lengths = axes_lengths_np / 2.0

        # 步骤2：计算OBB的8个顶点（中心 ± 半长×各轴方向）
        # 生成所有顶点的偏移组合（±1, ±1, ±1）
        signs = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ])
        # 计算每个顶点的坐标：centroid + sign × half_lengths × rotation_matrix
        vertices = []
        for sign in signs:
            vertex = centroid_np + rotation_matrix @ (sign * half_lengths)
            vertices.append(vertex)
        vertices_np = np.array(vertices)

        # 步骤3：计算AABB的min/max（所有顶点的x/y/z极值）
        min_coords = vertices_np.min(axis=0)  # 所有顶点x/y/z的最小值
        max_coords = vertices_np.max(axis=0)  # 所有顶点x/y/z的最大值

        # 步骤4：计算AABB的尺寸（max - min）
        dimensions = max_coords - min_coords

        # 步骤5：格式化AABB数据（保留精度）
        bbox_3d_aabb = {
            "min": {
                "x": round(min_coords[0], coord_precision),
                "y": round(min_coords[1], coord_precision),
                "z": round(min_coords[2], coord_precision)
            },
            "max": {
                "x": round(max_coords[0], coord_precision),
                "y": round(max_coords[1], coord_precision),
                "z": round(max_coords[2], coord_precision)
            },
            "dimensions": {
                "x": round(dimensions[0], coord_precision),
                "y": round(dimensions[1], coord_precision),
                "z": round(dimensions[2], coord_precision)
            }
        }

        # ========== 5. 组装最终结果 ==========
        result = {
            "3d_center": center_3d,
            "axis_directions": axis_directions,
            "bbox_3d_aabb": bbox_3d_aabb
        }

        return result

    def _process_annotation(self, raw_data: Union[dict, list], scene_id: str, data_type: str) -> Union[dict, list]:
        """
        【预留接口】修改标注数据结构
        :param raw_data: 原始JSON数据
        :param scene_id: 场景编号
        :param data_type: 数据类型（iphone/dslr）
        :return: 修改后的数据
        """
        # ===================== 你的修改逻辑写这里 =====================
        logger.info(f"[{scene_id}_{data_type}] 原始数据长度：{len(raw_data) if isinstance(raw_data, list) else 'dict'}")

        scene_objs = {}
        scene_objs_loc_list = {} # id: (category, 3d_center)
        scene_objs_id = 0
        modified_data = {
            "scene_id": scene_id,
            "data_type": data_type,
            "cameras": {},
            "objects": scene_objs
        }
        for img in raw_data:
            img_path = os.path.join(RAW_ROOT_DIR, img.get("image_path", ""))+'.jpg'
            W, H = get_image_size_pillow(img_path)
            new_image_path = f"{scene_id}_{data_type}/images/{Path(img.get('image_path','')).name}"
            camera_name = Path(img.get("image_path","")).stem
            camera_objs = {}
            modified_data["cameras"][camera_name] = {
                "intrinsics": img.get("intrinsic", []),
                "extrinsics": img.get("extrinsic", []),
                "width": W,
                "height": H,
                "image_path": new_image_path
            }
            for obj in img.get("objects", []):
                obj_id = scene_objs_id
                obj_category = obj.get("category", "unknown")
                obj_3d_center = obj.get("3D_location", [])
                if (obj_category, obj_3d_center) in scene_objs_loc_list.values():
                    # 该物体已存在，复用ID
                    obj_id = list(scene_objs_loc_list.keys())[list(scene_objs_loc_list.values()).index((obj_category, obj_3d_center))]
                else:
                    # 新物体，新增记录
                    scene_objs_loc_list[obj_id] = (obj_category, obj_3d_center)
                    convert_results = self._convert_obb_to_aabb_format(obj_3d_center,obj['3D_size'],obj['3D_rotation'])
                    modified_data["objects"][str(obj_id)] = {
                        "category": obj_category,
                        "3d_center": obj_3d_center,
                        "axis_directions": convert_results["axis_directions"],
                        "bbox_3d_aabb": convert_results["bbox_3d_aabb"]
                    }
                    scene_objs_id += 1

                camera_objs[str(obj_id)] = {
                    "object_index": obj_id,
                    "bbox_2d": {
                        "min_x": obj.get("2D_bbox", [])[0],
                        "min_y": obj.get("2D_bbox", [])[1],
                        "max_x": obj.get("2D_bbox", [])[2],
                        "max_y": obj.get("2D_bbox", [])[3]
                    }
                }
            modified_data["cameras"][camera_name]["objects"] = camera_objs

        # ===================== 修改逻辑结束 =====================
        return modified_data

    def _copy_image_files(self, src_img_dir: Path, dst_img_dir: Path) -> None:
        """
        迁移图片文件（保留原目录结构 + 去掉重复的图片扩展名）
        :param src_img_dir: 原始图片目录
        :param dst_img_dir: 目标图片目录
        """
        if not src_img_dir.exists():
            logger.warning(f"图片目录不存在，跳过迁移：{src_img_dir}")
            return

        # 定义支持的图片扩展名（小写）
        img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        # 先获取所有图片文件（不区分大小写）
        img_files = [f for f in src_img_dir.rglob("*") if f.suffix.lower() in img_extensions]

        if not img_files:
            logger.warning(f"图片目录下无有效图片：{src_img_dir}")
            return

        # 批量复制（带进度条）
        for img_file in tqdm(img_files, desc=f"迁移{src_img_dir.name}图片", unit="张", leave=False):
            # ========== 核心修改：去掉重复的扩展名 ==========
            # 1. 拆分文件名和扩展名（处理重复后缀，如 frame001.jpg.jpg）
            file_stem = img_file.stem  # 先取第一层stem（如 frame001.jpg → frame001）
            file_ext = img_file.suffix.lower()  # 原始扩展名（小写）
            
            # 2. 循环检查stem是否仍包含图片扩展名，直到无重复
            while any(file_stem.lower().endswith(ext) for ext in img_extensions):
                # 去掉stem末尾的重复扩展名
                file_stem = Path(file_stem).stem
            
            # 3. 重构无重复扩展名的文件名
            new_filename = f"{file_stem}{file_ext}"
            # ========== 重构目标路径 ==========
            # 保留原相对目录结构，但替换文件名
            rel_path = img_file.relative_to(src_img_dir)
            # 替换文件名（保留目录，只改文件名）
            dst_rel_path = rel_path.with_name(new_filename)
            dst_file = dst_img_dir / dst_rel_path
            
            # 确保目标目录存在
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                shutil.copy2(img_file, dst_file)  # 保留文件元数据
                logger.debug(f"成功迁移并修复文件名：{img_file.name} → {new_filename}")
            except Exception as e:
                logger.error(f"复制图片失败：{img_file} → {dst_file} | 错误：{str(e)[:50]}")

    def process_single_scene(self, scene_dir: Path) -> None:
        """
        处理单个场景：分别处理iphone/dslr数据
        :param scene_dir: 原始场景目录
        """
        scene_id = scene_dir.name
        logger.info(f"\n开始处理场景：{scene_id}")

        # 遍历处理iphone和dslr数据
        for data_type in ["iphone", "dslr"]:
            # 1. 构建路径
            src_anno_path = scene_dir / ANNO_CONFIG[data_type]  # 原始标注文件
            src_img_dir = scene_dir / IMG_DIR_CONFIG[data_type]  # 原始图片目录
            dst_scene_dir = self.output_root / f"{scene_id}_{data_type}"  # 目标场景目录
            dst_anno_path = dst_scene_dir / ANNO_CONFIG[data_type]  # 目标标注文件
            dst_img_dir = dst_scene_dir / 'images' #IMG_DIR_CONFIG[data_type]  # 目标图片目录

            # 2. 处理标注文件
            raw_anno = self._load_json(src_anno_path)
            if raw_anno is not None:
                modified_anno = self._process_annotation(raw_anno, scene_id, data_type)
                if self._save_json(modified_anno, dst_anno_path):
                    logger.info(f"[{scene_id}_{data_type}] 标注文件已保存：{dst_anno_path}")
            else:
                logger.error(f"[{scene_id}_{data_type}] 标注文件处理失败，跳过")
                continue  # 标注文件失败则跳过图片迁移

            # 3. 迁移图片文件
            self._copy_image_files(src_img_dir, dst_img_dir)

    def run(self) -> None:
        """执行全量处理：遍历所有场景+进度条"""
        if not self.scene_dirs:
            logger.warning("无有效场景可处理，程序退出")
            return

        # 全局进度条
        for scene_dir in tqdm(self.scene_dirs[:1], desc="总处理进度", unit="场景"):
            self.process_single_scene(scene_dir)

        logger.info("\n========== 所有场景处理完成 ==========")
        logger.info(f"原始目录：{self.raw_root}")
        logger.info(f"输出目录：{self.output_root}")


if __name__ == "__main__":
    # 初始化处理器并执行
    processor = ScanNetPPDataProcessor(
        raw_root=RAW_ROOT_DIR,
        output_root=OUTPUT_ROOT_DIR
    )
    processor.run()
