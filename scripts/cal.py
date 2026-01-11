# -*- coding: gbk -*-
import os
import json
from pathlib import Path

def analyze_scannetpp_dataset(root_folder):
    """
    分析 ScanNet++ 数据集文件夹结构，统计图像数量和检查标注文件
    
    Args:
        root_folder (str): 数据集根目录路径
    """
    root_path = Path(root_folder)
    scene_folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    total_images = 0
    scenes_without_annotation = []
    
    print(f"{'Scene':<20} {'Image Count':>12} {'Has Annotation':>15}")
    print("-" * 50)
    
    for scene in scene_folders:
        # 检查 iPhone 图像文件夹
        iphone_path = scene / "iphone"
        image_count = 0
        
        if iphone_path.exists():
            # 统计图像文件数量（假设是.jpg或.png格式）
            image_count = len([f for f in iphone_path.glob("*.[jJ][pP][gG]")]) + \
                         len([f for f in iphone_path.glob("*.[pP][nN][gG]")])
        
        # 检查标注文件
        annotation_path = scene / "obj_annotation.json"
        has_annotation = annotation_path.exists()
        
        if not has_annotation:
            scenes_without_annotation.append(scene.name)
        
        print(f"{scene.name:<20} {image_count:>12} {str(has_annotation):>15}")
        
        total_images += image_count
    
    print("\n" + "=" * 50)
    print(f"Total scenes: {len(scene_folders)}")
    print(f"Total images: {total_images}")
    
    if scenes_without_annotation:
        print(f"\nScenes missing obj_annotation.json ({len(scenes_without_annotation)}):")
        for scene in scenes_without_annotation:
            print(f" - {scene}")
    else:
        print("\nAll scenes have obj_annotation.json")

if __name__ == "__main__":
    dataset_path = input("Enter the path to your ScanNet++ dataset folder: ")
    analyze_scannetpp_dataset(dataset_path)