
import os
import time
import json
import warnings
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
matplotlib.use('Agg')  # 使用非交互式后端


def _configure_matplotlib_fonts() -> str:
    """
    Configure a CJK-capable font for matplotlib.
    Returns the selected font family name.
    """
    # Prefer common CJK fonts on Windows/macOS/Linux.
    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "NSimSun",
        "KaiTi",
        "FangSong",
        "PingFang SC",
        "Heiti SC",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]

    available_font_names = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((name for name in font_candidates if name in available_font_names), None)

    if selected_font:
        matplotlib.rcParams["font.family"] = [selected_font, "DejaVu Sans"]
    else:
        # Fallback to DejaVu Sans and suppress the known missing-glyph warning.
        matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
        warnings.filterwarnings(
            "ignore",
            message=r"Glyph \d+ .* missing from font\(s\) DejaVu Sans\.",
            category=UserWarning,
        )

    matplotlib.rcParams["axes.unicode_minus"] = False
    return selected_font or "DejaVu Sans"


DEFAULT_PLOT_FONT_FAMILY = _configure_matplotlib_fonts()


class ExDarkValidator:
    """ExDark验证集评估器"""
    
    def __init__(
        self,
        model_path: str,
        val_images_dir: str,
        img_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_half: bool = True,
        save_results: bool = True,
        output_dir: str = 'exdark_validation_results',
        auto_scale_labels_for_resolution: bool = True,
        label_scale_base: int = 1280,
        label_scale_min: float = 1.0,
        label_scale_max: float = 4.0,
        plot_line_width: int = None,
        plot_font_size: int = None
    ):
        """
        初始化验证器
        
        参数:
            model_path: 模型权重文件路径
            val_images_dir: 验证集图像目录
            img_size: 输入图像尺寸
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            device: 运行设备
            use_half: 是否使用FP16半精度
            save_results: 是否保存结果
            output_dir: 输出目录
            auto_scale_labels_for_resolution: 是否按图像分辨率自动放大预测标签
            label_scale_base: 标签缩放基准分辨率（长边）
            label_scale_min: 标签最小缩放倍率
            label_scale_max: 标签最大缩放倍率
            plot_line_width: 手动指定预测框线宽（None则自动）
            plot_font_size: 手动指定预测字体大小（None则自动）
        """
        self.model_path = Path(model_path)
        self.val_images_dir = Path(val_images_dir)
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.use_half = use_half and device == 'cuda'
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        self.auto_scale_labels_for_resolution = auto_scale_labels_for_resolution
        self.label_scale_base = max(1, int(label_scale_base))
        self.label_scale_min = float(label_scale_min)
        self.label_scale_max = float(label_scale_max)
        self.plot_line_width = plot_line_width
        self.plot_font_size = plot_font_size
        
        # 检查路径
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        if not self.val_images_dir.exists():
            raise FileNotFoundError(f"验证集目录不存在: {self.val_images_dir}")
        
        # 创建输出目录
        if self.save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.predictions_dir = self.output_dir / 'predictions'
            self.predictions_dir.mkdir(exist_ok=True)
            self.heatmaps_dir = self.output_dir / 'heatmaps'
            self.heatmaps_dir.mkdir(exist_ok=True)
        
        # 加载模型
        print(f"\n{'='*70}")
        print(f"ExDark验证集评估")
        print(f"{'='*70}")
        print(f"模型路径: {self.model_path}")
        print(f"验证集目录: {self.val_images_dir}")
        print(f"设备: {self.device}")
        print(f"图像尺寸: {self.img_size}")
        print(f"置信度阈值: {self.conf_threshold}")
        print(f"IOU阈值: {self.iou_threshold}")
        print(f"使用FP16: {self.use_half}")
        print(f"预测标签自动缩放: {self.auto_scale_labels_for_resolution}")
        if self.plot_line_width is not None:
            print(f"手动线宽: {self.plot_line_width}")
        if self.plot_font_size is not None:
            print(f"手动字体大小: {self.plot_font_size}")
        print(f"{'='*70}\n")
        
        print("正在加载模型...")
        self.model = YOLO(str(self.model_path))
        print("模型加载完成！\n")
        
        # 统计信息
        self.results_data = []
        self.inference_times = []
        self.detection_stats = defaultdict(int)
    
    def get_image_files(self):
        """获取所有图像文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.val_images_dir.glob(f'*{ext}'))
            image_files.extend(self.val_images_dir.glob(f'*{ext.upper()}'))
        
        return sorted(image_files)

    def _get_prediction_plot_style(self, image_shape: tuple):
        """
        根据图像分辨率计算可视化线宽和字体大小。
        """
        img_h, img_w = image_shape[:2]
        if self.auto_scale_labels_for_resolution:
            scale = float(np.clip(max(img_h, img_w) / float(self.label_scale_base),
                                  self.label_scale_min,
                                  self.label_scale_max))
        else:
            scale = 1.0

        line_width = self.plot_line_width if self.plot_line_width is not None else max(2, int(round(3 * scale)))
        font_size = self.plot_font_size if self.plot_font_size is not None else max(2, int(round(2 * scale)))
        return line_width, font_size
    
    def generate_heatmap(self, image_path: Path, detections: list, img_shape: tuple):
        """
        生成检测热力图
        
        参数:
            image_path: 图像路径
            detections: 检测结果列表
            img_shape: 图像形状 (height, width, channels)
        """
        if len(detections) == 0:
            return
        
        # 读取原始图像
        img = cv2.imread(str(image_path))
        if img is None:
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_shape[:2]
        
        # 创建热力图矩阵
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # 根据检测框和置信度生成热力图
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 创建高斯分布的热力图
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            
            # 使用高斯核生成热力图
            sigma_x = width / 4
            sigma_y = height / 4
            
            for y in range(max(0, y1), min(h, y2)):
                for x in range(max(0, x1), min(w, x2)):
                    # 计算高斯权重
                    dx = (x - center_x) / sigma_x if sigma_x > 0 else 0
                    dy = (y - center_y) / sigma_y if sigma_y > 0 else 0
                    weight = np.exp(-(dx**2 + dy**2) / 2) * conf
                    heatmap[y, x] = max(heatmap[y, x], weight)
        
        # 归一化热力图
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原始图像
        axes[0].imshow(img_rgb)
        axes[0].set_title('原始图像', fontsize=14, fontproperties='SimHei')
        axes[0].axis('off')
        
        # 热力图
        im = axes[1].imshow(heatmap, cmap='jet', alpha=1.0)
        axes[1].set_title('检测热力图', fontsize=14, fontproperties='SimHei')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 叠加图像
        axes[2].imshow(img_rgb)
        axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[2].set_title('叠加热力图', fontsize=14, fontproperties='SimHei')
        axes[2].axis('off')
        
        # 添加检测信息
        info_text = f'检测数: {len(detections)}\n'
        info_text += f'平均置信度: {np.mean([d["confidence"] for d in detections]):.3f}'
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存热力图
        output_path = self.heatmaps_dir / f"{image_path.stem}_heatmap.jpg"
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def evaluate_single_image(self, image_path: Path, save_visualization: bool = False, save_heatmap: bool = False):
        """
        评估单张图像
        
        参数:
            image_path: 图像路径
            save_visualization: 是否保存可视化结果
            save_heatmap: 是否保存热力图
        
        返回:
            dict: 包含检测结果和推理时间的字典
        """
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"警告: 无法读取图像 {image_path}")
            return None
        
        # 推理计时
        start_time = time.time()
        
        results = self.model.predict(
            img,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            half=self.use_half,
            verbose=False
        )
        
        # 同步CUDA
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 提取检测结果
        result = results[0]
        boxes = result.boxes
        
        detections = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0].cpu().numpy()),
                    'class_id': int(box.cls[0].cpu().numpy()),
                    'class_name': self.model.names[int(box.cls[0].cpu().numpy())]
                }
                detections.append(detection)
                self.detection_stats[detection['class_name']] += 1
        
        # 保存可视化结果（使用YOLO原生的plot方法，标签清晰不重叠）
        if save_visualization and self.save_results:
            line_width, font_size = self._get_prediction_plot_style(img.shape)

            # 使用YOLO原生的plot方法生成标注图像
            annotated_img = result.plot(
                line_width=line_width,  # 按分辨率自适应（4K会显著增大）
                font_size=font_size,    # 按分辨率自适应（4K会显著增大）
                labels=True,   # 显示标签
                conf=True,     # 显示置信度
                boxes=True     # 显示边框
            )
            
            output_path = self.predictions_dir / f"{image_path.stem}_pred.jpg"
            cv2.imwrite(str(output_path), annotated_img)
        
        # 生成热力图
        if save_heatmap and self.save_results and len(detections) > 0:
            self.generate_heatmap(image_path, detections, img.shape)
        
        result_dict = {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'inference_time_ms': inference_time,
            'num_detections': len(detections),
            'detections': detections
        }
        
        return result_dict
    
    def evaluate_all(self, save_visualizations: bool = True, save_heatmaps: bool = True, max_images: int = None):
        """
        评估所有验证集图像
        
        参数:
            save_visualizations: 是否保存所有可视化结果
            save_heatmaps: 是否保存热力图
            max_images: 最大评估图像数量（None表示全部）
        """
        # 获取所有图像文件
        image_files = self.get_image_files()
        total_images = len(image_files)
        
        if total_images == 0:
            print(f"错误: 在 {self.val_images_dir} 中未找到图像文件")
            return
        
        print(f"找到 {total_images} 张验证集图像\n")
        
        # 限制评估数量
        if max_images is not None and max_images < total_images:
            image_files = image_files[:max_images]
            print(f"将评估前 {max_images} 张图像\n")
        
        # 开始评估
        print("开始评估...")
        if save_visualizations:
            print(f"检测结果将保存到: {self.predictions_dir}")
        if save_heatmaps:
            print(f"热力图将保存到: {self.heatmaps_dir}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for idx, image_path in enumerate(image_files, 1):
            # 评估单张图像
            result = self.evaluate_single_image(image_path, save_visualizations, save_heatmaps)
            
            if result is not None:
                self.results_data.append(result)
                self.inference_times.append(result['inference_time_ms'])
                
                # 打印进度
                if idx % 10 == 0 or idx == len(image_files):
                    avg_time = np.mean(self.inference_times)
                    avg_fps = 1000.0 / avg_time if avg_time > 0 else 0
                    print(f"进度: {idx}/{len(image_files)} | "
                          f"当前推理时间: {result['inference_time_ms']:.2f}ms | "
                          f"平均推理时间: {avg_time:.2f}ms | "
                          f"平均FPS: {avg_fps:.2f}")
        
        total_time = time.time() - start_time
        
        # 打印统计结果
        self.print_statistics(total_time)
        
        # 保存结果
        if self.save_results:
            self.save_evaluation_results()
            
        # 打印保存位置
        if save_visualizations:
            print(f"\n所有检测结果图像已保存到: {self.predictions_dir.absolute()}")
        if save_heatmaps:
            print(f"所有热力图已保存到: {self.heatmaps_dir.absolute()}")
    
    def print_statistics(self, total_time: float):
        """打印统计信息"""
        print(f"\n{'='*70}")
        print(f"评估完成")
        print(f"{'='*70}")
        
        if len(self.results_data) == 0:
            print("没有成功评估的图像")
            return
        
        # 基本统计
        print(f"\n基本信息:")
        print(f"  总图像数: {len(self.results_data)}")
        print(f"  总运行时间: {total_time:.2f} 秒")
        print(f"  平均每张图像时间: {total_time/len(self.results_data):.2f} 秒")
        
        # 推理时间统计
        inference_times = np.array(self.inference_times)
        print(f"\n推理时间统计 (毫秒):")
        print(f"  平均: {np.mean(inference_times):.2f} ms")
        print(f"  中位数: {np.median(inference_times):.2f} ms")
        print(f"  最小: {np.min(inference_times):.2f} ms")
        print(f"  最大: {np.max(inference_times):.2f} ms")
        print(f"  标准差: {np.std(inference_times):.2f} ms")
        print(f"  P95: {np.percentile(inference_times, 95):.2f} ms")
        print(f"  P99: {np.percentile(inference_times, 99):.2f} ms")
        
        # FPS统计
        avg_inference_time = np.mean(inference_times)
        avg_fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0
        max_fps = 1000.0 / np.min(inference_times) if np.min(inference_times) > 0 else 0
        
        print(f"\nFPS统计:")
        print(f"  平均FPS: {avg_fps:.2f}")
        print(f"  理论最大FPS: {max_fps:.2f}")
        
        # 检测统计
        total_detections = sum(r['num_detections'] for r in self.results_data)
        avg_detections = total_detections / len(self.results_data)
        
        print(f"\n检测统计:")
        print(f"  总检测数: {total_detections}")
        print(f"  平均每张图像检测数: {avg_detections:.2f}")
        print(f"  有检测的图像数: {sum(1 for r in self.results_data if r['num_detections'] > 0)}")
        print(f"  无检测的图像数: {sum(1 for r in self.results_data if r['num_detections'] == 0)}")
        
        # 类别统计
        if self.detection_stats:
            print(f"\n各类别检测数量:")
            for class_name, count in sorted(self.detection_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {count}")
        
        print(f"\n{'='*70}\n")
    
    def save_evaluation_results(self):
        """保存评估结果到JSON文件"""
        # 计算统计信息
        inference_times = np.array(self.inference_times)
        
        summary = {
            'model_path': str(self.model_path),
            'val_images_dir': str(self.val_images_dir),
            'device': self.device,
            'img_size': self.img_size,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'use_half': self.use_half,
            'total_images': len(self.results_data),
            'total_detections': sum(r['num_detections'] for r in self.results_data),
            'avg_detections_per_image': sum(r['num_detections'] for r in self.results_data) / len(self.results_data) if self.results_data else 0,
            'inference_time_stats': {
                'avg_ms': float(np.mean(inference_times)),
                'median_ms': float(np.median(inference_times)),
                'min_ms': float(np.min(inference_times)),
                'max_ms': float(np.max(inference_times)),
                'std_ms': float(np.std(inference_times)),
                'p95_ms': float(np.percentile(inference_times, 95)),
                'p99_ms': float(np.percentile(inference_times, 99))
            },
            'fps_stats': {
                'avg_fps': float(1000.0 / np.mean(inference_times)) if np.mean(inference_times) > 0 else 0,
                'max_fps': float(1000.0 / np.min(inference_times)) if np.min(inference_times) > 0 else 0
            },
            'class_statistics': dict(self.detection_stats),
            'detailed_results': self.results_data
        }
        
        # 保存完整结果
        output_file = self.output_dir / 'evaluation_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        print(f"详细结果已保存到: {output_file}")
        
        # 保存简要统计
        summary_file = self.output_dir / 'evaluation_summary.json'
        summary_only = {k: v for k, v in summary.items() if k != 'detailed_results'}
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_only, f, indent=4, ensure_ascii=False)
        print(f"统计摘要已保存到: {summary_file}")
    
    def generate_overall_heatmap_analysis(self):
        """生成整体热力图分析报告"""
        if len(self.results_data) == 0:
            return
        
        print(f"\n{'='*70}")
        print("生成整体热力图分析...")
        print(f"{'='*70}\n")
        
        # 统计检测位置分布
        all_centers = []
        all_confidences = []
        
        for result in self.results_data:
            for det in result['detections']:
                bbox = det['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                all_centers.append([center_x, center_y])
                all_confidences.append(det['confidence'])
        
        if len(all_centers) == 0:
            print("没有检测结果，跳过整体热力图分析")
            return
        
        all_centers = np.array(all_centers)
        all_confidences = np.array(all_confidences)
        
        # 创建整体分析图
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. 检测中心点分布散点图
        scatter = axes[0, 0].scatter(all_centers[:, 0], all_centers[:, 1],
                                     c=all_confidences, cmap='viridis',
                                     alpha=0.6, s=50)
        axes[0, 0].set_title('检测中心点分布 (颜色表示置信度)', fontsize=14, fontproperties='SimHei')
        axes[0, 0].set_xlabel('X坐标', fontsize=12, fontproperties='SimHei')
        axes[0, 0].set_ylabel('Y坐标', fontsize=12, fontproperties='SimHei')
        axes[0, 0].invert_yaxis()
        plt.colorbar(scatter, ax=axes[0, 0], label='置信度')
        
        # 2. 置信度分布直方图
        axes[0, 1].hist(all_confidences, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('置信度分布', fontsize=14, fontproperties='SimHei')
        axes[0, 1].set_xlabel('置信度', fontsize=12, fontproperties='SimHei')
        axes[0, 1].set_ylabel('频数', fontsize=12, fontproperties='SimHei')
        axes[0, 1].axvline(np.mean(all_confidences), color='red',
                          linestyle='--', linewidth=2, label=f'平均值: {np.mean(all_confidences):.3f}')
        axes[0, 1].legend(prop={'family': 'SimHei'})
        
        # 3. 检测数量分布
        detection_counts = [r['num_detections'] for r in self.results_data]
        axes[1, 0].hist(detection_counts, bins=range(0, max(detection_counts)+2),
                       color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('每张图像检测数量分布', fontsize=14, fontproperties='SimHei')
        axes[1, 0].set_xlabel('检测数量', fontsize=12, fontproperties='SimHei')
        axes[1, 0].set_ylabel('图像数量', fontsize=12, fontproperties='SimHei')
        
        # 4. 类别分布饼图
        if self.detection_stats:
            class_names = list(self.detection_stats.keys())
            class_counts = list(self.detection_stats.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
            wedges, texts, autotexts = axes[1, 1].pie(class_counts, labels=class_names,
                                                       autopct='%1.1f%%', colors=colors,
                                                       startangle=90)
            axes[1, 1].set_title('类别检测分布', fontsize=14, fontproperties='SimHei')
            
            # 设置字体
            for text in texts:
                text.set_fontproperties('SimHei')
        else:
            axes[1, 1].text(0.5, 0.5, '无类别统计数据',
                           ha='center', va='center', fontsize=14, fontproperties='SimHei')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # 保存整体分析图
        analysis_path = self.output_dir / 'overall_heatmap_analysis.jpg'
        plt.savefig(str(analysis_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"整体热力图分析已保存到: {analysis_path.absolute()}")
        
        # 生成统计报告
        report_path = self.output_dir / 'heatmap_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("热力图分析报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"总检测数: {len(all_centers)}\n")
            f.write(f"总图像数: {len(self.results_data)}\n")
            f.write(f"平均每张图像检测数: {len(all_centers)/len(self.results_data):.2f}\n\n")
            
            f.write("置信度统计:\n")
            f.write(f"  平均值: {np.mean(all_confidences):.4f}\n")
            f.write(f"  中位数: {np.median(all_confidences):.4f}\n")
            f.write(f"  最小值: {np.min(all_confidences):.4f}\n")
            f.write(f"  最大值: {np.max(all_confidences):.4f}\n")
            f.write(f"  标准差: {np.std(all_confidences):.4f}\n\n")
            
            f.write("检测位置统计:\n")
            f.write(f"  X坐标平均值: {np.mean(all_centers[:, 0]):.2f}\n")
            f.write(f"  Y坐标平均值: {np.mean(all_centers[:, 1]):.2f}\n")
            f.write(f"  X坐标标准差: {np.std(all_centers[:, 0]):.2f}\n")
            f.write(f"  Y坐标标准差: {np.std(all_centers[:, 1]):.2f}\n\n")
            
            if self.detection_stats:
                f.write("类别统计:\n")
                for class_name, count in sorted(self.detection_stats.items(),
                                               key=lambda x: x[1], reverse=True):
                    percentage = (count / len(all_centers)) * 100
                    f.write(f"  {class_name}: {count} ({percentage:.2f}%)\n")
        
        print(f"热力图分析报告已保存到: {report_path.absolute()}")


def main():
    """主函数"""
    
    # 配置参数
    MODEL_PATH = r"C:\Users\10443\Desktop\access\Comparative\visdrone\yolo11s\weights\best.pt"
    VAL_IMAGES_DIR = r"C:\Users\10443\Desktop\access\img"
    
    # 评估参数
    IMG_SIZE = 640
    CONF_THRESHOLD = 0.25  # 置信度阈值
    IOU_THRESHOLD = 0.5   # IOU阈值
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_HALF = True  # 是否使用FP16半精度（仅GPU）
    
    # 输出参数
    SAVE_RESULTS = True  # 是否保存结果
    SAVE_VISUALIZATIONS = True  # 是否保存所有带检测框的图像
    SAVE_HEATMAPS = True  # 是否保存热力图
    OUTPUT_DIR = 'nod_GDIP-val11'
    MAX_IMAGES = 500 # 最大评估图像数量，None表示全部（可以设置如100来测试）
    AUTO_SCALE_LABELS_FOR_RESOLUTION = True  # 4K图建议保持True
    PLOT_LINE_WIDTH = None  # 例如设为 8；None表示自动
    PLOT_FONT_SIZE = None   # 例如设为 6；None表示自动
    
    try:
        # 创建验证器
        validator = ExDarkValidator(
            model_path=MODEL_PATH,
            val_images_dir=VAL_IMAGES_DIR,
            img_size=IMG_SIZE,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            device=DEVICE,
            use_half=USE_HALF,
            save_results=SAVE_RESULTS,
            output_dir=OUTPUT_DIR,
            auto_scale_labels_for_resolution=AUTO_SCALE_LABELS_FOR_RESOLUTION,
            plot_line_width=PLOT_LINE_WIDTH,
            plot_font_size=PLOT_FONT_SIZE
        )
        
        # 执行评估
        validator.evaluate_all(
            save_visualizations=SAVE_VISUALIZATIONS,
            save_heatmaps=SAVE_HEATMAPS,
            max_images=MAX_IMAGES
        )
        
        # 生成整体热力图分析
        if SAVE_HEATMAPS:
            validator.generate_overall_heatmap_analysis()
        
        print("\n评估完成！")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
