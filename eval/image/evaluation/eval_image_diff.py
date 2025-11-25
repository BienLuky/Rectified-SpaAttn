import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import argparse
import os
import lpips
import torch
import logging
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# 设置日志级别为 WARNING，这样 INFO 级别的日志将不会输出
logging.getLogger('lpips').setLevel(logging.WARNING)

# 全局初始化 LPIPS 评估器，避免反复加载模型
lpips_loss_fn = lpips.LPIPS(net='alex')
if torch.cuda.is_available():
    lpips_loss_fn.cuda()


def calculate_metrics(image_path1, image_path2):
    # 读取两张图像
    try:
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
    except Exception as e:
        print(f"读取图像时出错: {e}")
        return None, None, None, None, None, None

    # 检查图像是否成功读取
    if img1 is None or img2 is None:
        print(f"无法读取图像文件: {image_path1} 或 {image_path2}")
        return None, None, None, None, None, None

    # 将图像转换为灰度图像，因为 SSIM 通常在灰度图像上计算
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算 SSIM
    ssim_score = ssim(gray1, gray2)

    # 计算 PSNR
    psnr_score = psnr(img1, img2)

    # 计算 LPIPS
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    if torch.cuda.is_available():
        img1_tensor = img1_tensor.cuda()
        img2_tensor = img2_tensor.cuda()
    with torch.no_grad():
        lpips_score = lpips_loss_fn(img1_tensor, img2_tensor).item()

    # 计算余弦相似度
    vec1 = img1.flatten().reshape(1, -1)
    vec2 = img2.flatten().reshape(1, -1)
    cosine_score = cosine_similarity(vec1, vec2)[0][0]

    # 计算 Relative L1
    l1_error = np.sum(np.abs(img1 - img2))
    original_l1_norm = np.sum(np.abs(img1))
    relative_l1_score = l1_error / original_l1_norm if original_l1_norm != 0 else 0

    # 计算 RMSE
    mse = np.mean((img1 - img2) ** 2)
    rmse_score = np.sqrt(mse)

    return ssim_score, psnr_score, lpips_score, cosine_score, relative_l1_score, rmse_score


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='计算两个文件夹中图像文件的多个指标')
    # 添加文件夹路径参数，定义为选项参数
    parser.add_argument('--path1', type=str, help='第一个文件夹的路径')
    parser.add_argument('--path2', type=str, help='第二个文件夹的路径')
    parser.add_argument(
        "--log_file",
        default= "/mnt/nas/lizhikai/Liu/GRAT/eval/image/eval.txt",
        type=str,
        help="log file",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 获取文件夹路径
    path1 = args.path1
    path2 = args.path2

    # 检查文件夹是否存在
    if not os.path.exists(path1) or not os.path.exists(path2):
        print("指定的文件夹不存在，请检查路径。")
    else:
        # 获取两个文件夹中的所有图像文件
        image_extensions = ('.png', '.jpg', '.jpeg')
        image_files1 = [os.path.join(path1, f) for f in os.listdir(path1) if f.lower().endswith(image_extensions)]
        image_files2 = [os.path.join(path2, f) for f in os.listdir(path2) if f.lower().endswith(image_extensions)]

        # 确保两个文件夹中的图像文件数量相同
        if len(image_files1) != len(image_files2):
            print("两个文件夹中的图像文件数量不匹配，请检查。")
        else:
            all_image_ssim_scores = []
            all_image_psnr_scores = []
            all_image_lpips_scores = []
            all_image_cosine_scores = []
            all_image_relative_l1_scores = []
            all_image_rmse_scores = []
            image_metrics = {}  # 用于存储每个图像的指标

            with tqdm(total=len(image_files1), desc="计算图像指标", disable=True) as pbar:
                for img1, img2 in zip(image_files1, image_files2):
                    ssim_score, psnr_score, lpips_score, cosine_score, relative_l1_score, rmse_score = calculate_metrics(
                        img1, img2)
                    if ssim_score is not None and psnr_score is not None and lpips_score is not None and cosine_score is not None and relative_l1_score is not None and rmse_score is not None:
                        all_image_ssim_scores.append(ssim_score)
                        all_image_psnr_scores.append(psnr_score)
                        all_image_lpips_scores.append(lpips_score)
                        all_image_cosine_scores.append(cosine_score)
                        all_image_relative_l1_scores.append(relative_l1_score)
                        all_image_rmse_scores.append(rmse_score)
                        image_metrics[os.path.basename(img2)] = {
                            'SSIM': ssim_score,
                            'PSNR': psnr_score,
                            'LPIPS': lpips_score,
                            'Cosine Similarity': cosine_score,
                            'Relative L1': relative_l1_score,
                            'RMSE': rmse_score
                        }
                    pbar.update(1)

            if all_image_ssim_scores and all_image_psnr_scores and all_image_lpips_scores and all_image_cosine_scores and all_image_relative_l1_scores and all_image_rmse_scores:
                overall_average_ssim = np.mean(all_image_ssim_scores)
                overall_average_psnr = np.mean(all_image_psnr_scores)
                overall_average_lpips = np.mean(all_image_lpips_scores)
                overall_average_cosine = np.mean(all_image_cosine_scores)
                overall_average_relative_l1 = np.mean(all_image_relative_l1_scores)
                overall_average_rmse = np.mean(all_image_rmse_scores)
                print(f"路径 1: {path1}")
                print(f"路径 2: {path2}")
                print("\n所有图像的平均指标:")
                print(f"  平均 SSIM: {overall_average_ssim}")
                print(f"  平均 PSNR: {overall_average_psnr}")
                print(f"  平均 LPIPS: {overall_average_lpips}")
                print(f"  平均余弦相似度: {overall_average_cosine}")
                print(f"  平均 Relative L1: {overall_average_relative_l1}")
                print(f"  平均 RMSE: {overall_average_rmse}")

                # 获取 path2 的上一级目录
                output_dir = os.path.dirname(path2)
                output_file = os.path.join(output_dir, 'eval.txt')

                if args.log_file:
                    with open(args.log_file, 'a') as f:
                # 将评估结果写入文件
                # with open(output_file, 'w') as f:
                        f.write(f"路径 1: {path1}\n")
                        f.write(f"路径 2: {path2}\n")
                        f.write("\n所有图像的平均指标:\n")
                        f.write(f"  平均 SSIM: {overall_average_ssim}\n")
                        f.write(f"  平均 PSNR: {overall_average_psnr}\n")
                        f.write(f"  平均 LPIPS: {overall_average_lpips}\n")
                        f.write(f"  平均余弦相似度: {overall_average_cosine}\n")
                        f.write(f"  平均 Relative L1: {overall_average_relative_l1}\n")
                        f.write(f"  平均 RMSE: {overall_average_rmse}\n")
                        f.write("\n每个图像的指标:\n")
                        for image_name, metrics in image_metrics.items():
                            f.write(f"图像: {image_name}\n")
                            f.write(f"  SSIM: {metrics['SSIM']}\n")
                            f.write(f"  PSNR: {metrics['PSNR']}\n")
                            f.write(f"  LPIPS: {metrics['LPIPS']}\n")
                            f.write(f"  余弦相似度: {metrics['Cosine Similarity']}\n")
                            f.write(f"  Relative L1: {metrics['Relative L1']}\n")
                            f.write(f"  RMSE: {metrics['RMSE']}\n")
            else:
                print("没有有效的指标分数，请检查图像文件。")
