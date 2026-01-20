import requests
import json
import re
import csv
from PIL import Image, ImageDraw, ImageFont
import os
import time
import xml.etree.ElementTree as ET
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from volcenginesdkarkruntime import Ark

# 读取提示词
script_dir = os.path.dirname(os.path.abspath(__file__))
qwen_client = None
doubao_client_18 = None
_clients_lock = threading.Lock()


def _load_env_file(env_path: str) -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def init_clients() -> None:
    global qwen_client, doubao_client_18
    if qwen_client is not None and doubao_client_18 is not None:
        return
    with _clients_lock:
        if qwen_client is not None and doubao_client_18 is not None:
            return
        _load_env_file(os.path.join(script_dir, ".env"))
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        ark_api_key = os.getenv("ARK_API_KEY", "").strip()
        if not dashscope_api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY，请在 .env 或环境变量中配置")
        if not ark_api_key:
            raise RuntimeError("缺少 ARK_API_KEY，请在 .env 或环境变量中配置")

        qwen_client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=dashscope_api_key,
        )
        doubao_client_18 = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=ark_api_key,
        )

prompt_candidates = [
    os.path.join(script_dir, 'prompts', 'prompt原始.md'),
    os.path.join(script_dir, 'prompts', 'prompt优化.md'),
]
prompt_path = next((p for p in prompt_candidates if os.path.exists(p)), None)
if prompt_path is None:
    raise FileNotFoundError(f"未找到提示词文件，已尝试：{prompt_candidates}")
data_dir = os.path.join(script_dir, 'data')
outputs_dir = os.path.join(script_dir, 'outputs')
images_dir = os.path.join(data_dir, 'images')
annotations_dir = os.path.join(data_dir, 'annotations')
outputs_bboxes_dir = os.path.join(outputs_dir, 'bboxes')
outputs_visualizations_dir = os.path.join(outputs_dir, 'visualizations')
outputs_csv_dir = os.path.join(outputs_dir, 'csv')
outputs_tmp_dir = os.path.join(outputs_dir, 'tmp')

os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)
os.makedirs(outputs_bboxes_dir, exist_ok=True)
os.makedirs(outputs_visualizations_dir, exist_ok=True)
os.makedirs(outputs_csv_dir, exist_ok=True)
os.makedirs(outputs_tmp_dir, exist_ok=True)
with open(prompt_path, 'r', encoding='utf-8') as f:
    prompt = f.read().strip()

# 图片URL列表
image_urls = [
    "https://zhuoyu.tos-cn-beijing.volces.com/img1.jpg",
    "https://zhuoyu.tos-cn-beijing.volces.com/img2.jpg"
]

# 下载图片
def download_image(url, save_path="temp_image.jpg"):
    """下载图片到本地，失败时尝试使用本地同名文件"""
    # 从URL中提取图片名称
    image_name = os.path.basename(url)
    # 构建本地图片的完整路径
    local_image_path = os.path.join(images_dir, image_name)
    
    # 先检查本地是否有同名图片
    if os.path.exists(local_image_path):
        print(f"使用本地{image_name}")
        return local_image_path
    
    # 构建临时图片的完整路径
    temp_image_path = os.path.join(outputs_tmp_dir, save_path)
    
    # 尝试下载图片
    response = requests.get(url)
    if response.status_code == 200 and response.content:
        with open(temp_image_path, 'wb') as f:
            f.write(response.content)
        return temp_image_path
    else:
        # 下载失败，再次检查本地是否有同名图片
        if os.path.exists(local_image_path):
            print(f"图片下载失败，状态码：{response.status_code}，使用本地{image_name}")
            return local_image_path
        else:
            raise Exception(f"图片下载失败，状态码：{response.status_code}，本地也没有找到{image_name}")

# 解析XML文件提取标准答案bbox

def parse_xml_groundtruth(image_url):
    """从XML文件中提取标准答案的bbox信息"""
    # 从URL中提取图片名称
    image_name = os.path.basename(image_url)
    # 构建对应的XML文件路径
    xml_name = os.path.splitext(image_name)[0] + '.xml'
    xml_path = os.path.join(annotations_dir, xml_name)
    
    # 检查XML文件是否存在
    if not os.path.exists(xml_path):
        raise Exception(f"未找到对应的标准答案文件：{xml_path}")
    
    groundtruth_bboxes = []
    
    try:
        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 提取所有object的bndbox信息
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                groundtruth_bboxes.append([xmin, ymin, xmax, ymax])
    except Exception as e:
        raise Exception(f"解析XML文件出错：{str(e)}")
    
    return groundtruth_bboxes

# 计算IoU值
def calculate_iou(box1, box2):
    """计算两个bbox之间的IoU值"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # 计算交集的坐标
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    
    # 计算交集面积
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
    
    # 计算两个bbox的面积
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    
    # 计算并集面积
    union_area = box1_area + box2_area - inter_area
    
    # 计算IoU
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

# 为模型预测的bbox与标准答案bbox计算IoU
def calculate_model_iou(model_bboxes, groundtruth_bboxes):
    """为模型预测的所有bbox与标准答案的bbox计算IoU值"""
    iou_results = []
    
    for pred_box in model_bboxes:
        max_iou = 0.0
        matched_gt_box = None
        
        # 寻找与当前预测框IoU最大的标准答案框
        for gt_box in groundtruth_bboxes:
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                matched_gt_box = gt_box
        
        iou_results.append({
            'pred_box': pred_box,
            'gt_box': matched_gt_box,
            'iou': max_iou
        })
    
    return iou_results

# 提取bbox信息
def extract_bboxes(model_output):
    """从模型输出中提取原始bbox信息"""
    # 递归从JSON结构中提取bbox
    def extract_from_json(data):
        bboxes = []
        if isinstance(data, dict):
            for key, value in data.items():
                if key == 'bbox':
                    bboxes.append(value)
                else:
                    bboxes.extend(extract_from_json(value))
        elif isinstance(data, list):
            for item in data:
                bboxes.extend(extract_from_json(item))
        return bboxes
    
    original_bboxes = []
    try:
        # 尝试解析JSON
        if isinstance(model_output, str):
            parsed = json.loads(model_output)
        else:
            parsed = model_output
        original_bboxes = extract_from_json(parsed)
    except json.JSONDecodeError:
        # 如果不是JSON，尝试用正则表达式提取
        if isinstance(model_output, str):
            bbox_pattern = r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]'
            matches = re.findall(bbox_pattern, model_output)
            for match in matches:
                try:
                    bbox = json.loads(match)
                    if len(bbox) == 4:
                        original_bboxes.append(bbox)
                except:
                    continue
    
    return original_bboxes

# 获取下一个序列编号
def get_next_sequence_number():
    """获取下一个可用的序列编号（0-100）"""
    for i in range(101):  # 从0到100
        img_path = os.path.join(outputs_visualizations_dir, f"result_with_bboxes_{i}.jpg")
        bbox_path = os.path.join(outputs_bboxes_dir, f"bbox_info_{i}.txt")
        if not os.path.exists(img_path) and not os.path.exists(bbox_path):
            return i
    return 0  # 如果所有编号都已使用，返回0

# 调整bbox坐标
def adjust_bboxes(bboxes, image_width, image_height):
    """根据图片尺寸调整bbox坐标
    1. 如果bbox坐标在0-1之间，认为是相对坐标，转换为绝对像素值
    2. 否则直接使用模型返回的坐标，确保为整数
    """
    adjusted_bboxes = []
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        
        # 检查是否为相对坐标（值在0-1之间）
        if max(x1, y1, x2, y2) <= 1.0:
            # 转换为绝对像素值
            x1 = int(x1 * image_width)
            y1 = int(y1 * image_height)
            x2 = int(x2 * image_width)
            y2 = int(y2 * image_height)
        else:
            # 直接使用模型返回的坐标，确保为整数
            x1 = int(x1*image_width/1000)
            y1 = int(y1*image_height/1000)
            x2 = int(x2*image_width/1000)
            y2 = int(y2*image_height/1000)
        
        # 确保坐标在图片范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)
        
        adjusted_bboxes.append([x1, y1, x2, y2])
    return adjusted_bboxes

# 绘制bbox
def draw_bboxes(image_path, qwen_bboxes, doubao_bboxes_18, output_path,
               qwen_iou_results=None, doubao_18_iou_results=None):
    """在图片上绘制bboxes"""
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        # 获取图片尺寸
        width, height = img.size
        
        # 调整bbox坐标
        adjusted_qwen_bboxes = adjust_bboxes(qwen_bboxes, width, height)
        adjusted_doubao_bboxes_18 = adjust_bboxes(doubao_bboxes_18, width, height)
        
        # 尝试加载系统字体
        try:
            # 使用系统默认字体，增大字体大小到20以确保清晰可见
            font = ImageFont.truetype("Arial.ttf", 40)
        except IOError:
            # 如果Arial不可用，使用默认字体
            font = ImageFont.load_default()
        
        # 在左上角添加两行备注
        legend_texts = [
            "red: Qwen-vl-plus",
            "blue: doubao-seed-1.8"
        ]
        legend_colors = ["red", "blue"]
        
        # 绘制图例，从左上角开始，每行间距30像素
        for i, (text, color) in enumerate(zip(legend_texts, legend_colors)):
            y_position = 20 + i * 30
            draw.text((20, y_position), text, fill=color, font=font)
        
        # 绘制Qwen3的bbox（红色）
        for i, bbox in enumerate(adjusted_qwen_bboxes):
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # 添加IoU信息
            if qwen_iou_results and i < len(qwen_iou_results):
                iou = qwen_iou_results[i]['iou']
                # 在bbox左上角显示IoU值
                draw.text((x1, y1 - 30), f"IoU: {iou:.2f}", fill="red", font=font)
        
        # 绘制豆包1.8的bbox（蓝色）
        for i, bbox in enumerate(adjusted_doubao_bboxes_18):
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
            # 添加IoU信息
            if doubao_18_iou_results and i < len(doubao_18_iou_results):
                iou = doubao_18_iou_results[i]['iou']
                # 在bbox左上角显示IoU值
                draw.text((x1, y1 - 30), f"IoU: {iou:.2f}", fill="blue", font=font)
        
        img.save(output_path, "JPEG")
        
        return adjusted_qwen_bboxes, adjusted_doubao_bboxes_18

# 保存bbox信息到文件
def save_bbox_info(seq_num, qwen_bboxes, doubao_bboxes_18, 
                  qwen_iou_results=None, doubao_18_iou_results=None):
    """保存bbox信息到文件"""
    bbox_path = os.path.join(outputs_bboxes_dir, f"bbox_info_{seq_num}.txt")
    with open(bbox_path, 'w', encoding='utf-8') as f:
        # 写入Qwen模型的bbox和IoU
        f.write("Qwen模型bbox结果：\n")
        for i, bbox in enumerate(qwen_bboxes):
            x1, y1, x2, y2 = bbox
            if qwen_iou_results and i < len(qwen_iou_results):
                iou = qwen_iou_results[i]['iou']
                f.write(f"<bbox>{x1} {y1} {x2} {y2}</bbox> IoU: {iou:.4f}\n")
            else:
                f.write(f"<bbox>{x1} {y1} {x2} {y2}</bbox>\n")
        
        # 写入豆包1.8模型的bbox和IoU
        f.write("\n豆包1.8模型bbox结果：\n")
        for i, bbox in enumerate(doubao_bboxes_18):
            x1, y1, x2, y2 = bbox
            if doubao_18_iou_results and i < len(doubao_18_iou_results):
                iou = doubao_18_iou_results[i]['iou']
                f.write(f"<bbox>{x1} {y1} {x2} {y2}</bbox> IoU: {iou:.4f}\n")
            else:
                f.write(f"<bbox>{x1} {y1} {x2} {y2}</bbox>\n")
    return bbox_path

# 生成CSV结果文件
def generate_csv_results(results):
    """将所有运行结果输出到CSV文件，格式参考给定的CSV文件"""
    # 生成CSV文件名，包含当前时间
    current_time = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(outputs_csv_dir, f"model_comparison_results_{current_time}.csv")
    
    # CSV列标题
    headers = [
        "图片名称", "运行序号", "提问内容", 
        "seed18模型回答", "seed18-bbox", "seed18推理时间(s)", "seed18输入tokens", "seed18输出tokens", "seed18总tokens", "seed18平均IoU",
        "Qwen模型回答", "Qwen-bbox", "Qwen推理时间(s)", "Qwen输入tokens", "Qwen输出tokens", "Qwen总tokens", "Qwen平均IoU",
        "对比图像", "处理状态", "处理耗时"
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for result in results:
            # 提取图片名称
            image_name = os.path.basename(result["image_url"])
            
            # 准备bbox字符串
            def format_bboxes(bboxes):
                return "".join([f"<bbox>{x1} {y1} {x2} {y2}</bbox>" for x1, y1, x2, y2 in bboxes])
            
            # 从输出中提取bbox信息
            seed18_bboxes = extract_bboxes(result["doubao_18_output"])
            qwen_bboxes = extract_bboxes(result["qwen_output"])
            
            # 格式化bbox
            seed18_bbox_str = format_bboxes(seed18_bboxes)
            qwen_bbox_str = format_bboxes(qwen_bboxes)
            
            # 对比图像路径
            comparison_image = os.path.basename(result["output_image_path"])
            
            # 处理状态
            status = "成功"
            
            # 处理耗时（总耗时）
            total_time = result["qwen_inference_time"] + result["doubao_18_inference_time"]
            
            # 准备一行数据
            row = [
                image_name, 
                result["run_id"], 
                prompt, 
                
                result["doubao_18_output"], 
                seed18_bbox_str, 
                result["doubao_18_inference_time"], 
                result["doubao_18_prompt_tokens"], 
                result["doubao_18_completion_tokens"], 
                result["doubao_18_total_tokens"], 
                result.get("avg_doubao_18_iou", 0.0), 
                
                result["qwen_output"], 
                qwen_bbox_str, 
                result["qwen_inference_time"], 
                result["qwen_prompt_tokens"], 
                result["qwen_completion_tokens"], 
                result["qwen_total_tokens"], 
                result.get("avg_qwen_iou", 0.0), 
                
                comparison_image, 
                status, 
                total_time
            ]
            
            writer.writerow(row)
    
    print(f"CSV结果文件已生成：{csv_path}")
    return csv_path

# 处理单张图片的单个运行实例
def process_image_instance(image_url, run_id, all_results):
    """处理单张图片的单个运行实例"""
    print(f"\n=== 开始处理图片 {image_url} 的第 {run_id+1} 次运行 ===")
    
    try:
        # 为每个线程生成唯一的临时文件名
        import threading
        thread_id = threading.get_ident()
        # 下载图片
        temp_image_path = download_image(image_url, f"temp_image_{thread_id}_{run_id}.jpg")
        
        init_clients()

        # 调用Qwen模型
        print(f"[{run_id+1}] 正在调用Qwen3模型...")
        start_time = time.time()
        qwen_response = qwen_client.chat.completions.create(
            model="qwen3-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            extra_body={"enable_thinking":True,
            "vl_high_resolution_images":True
            },
            temperature=0.1,
            top_p=0.7
        )
        end_time = time.time()
        qwen_inference_time = end_time - start_time
        
        # 输出token数和耗时
        qwen_prompt_tokens = 'N/A'
        qwen_completion_tokens = 'N/A'
        qwen_total_tokens = 'N/A'
        if hasattr(qwen_response, 'usage'):
            qwen_prompt_tokens = getattr(qwen_response.usage, 'prompt_tokens', 'N/A')
            qwen_completion_tokens = getattr(qwen_response.usage, 'completion_tokens', 'N/A')
            qwen_total_tokens = getattr(qwen_response.usage, 'total_tokens', 'N/A')
        print(f"[{run_id+1}] Qwen3模型消耗token数: 输入={qwen_prompt_tokens}, 输出={qwen_completion_tokens}, 总计={qwen_total_tokens}")
        print(f"[{run_id+1}] Qwen3模型推理耗时: {qwen_inference_time:.2f} 秒")
        
        # 调用豆包1.8模型
        print(f"[{run_id+1}] 正在调用豆包1.8模型...")
        start_time = time.time()
        doubao_response_18 = doubao_client_18.chat.completions.create(
            model="ep-20260106104004-9tpbd",  # 使用指定的模型ID
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url},"detail":"high"}
                    ]
                }
            ],
            extra_body={
        "thinking": {
            "type": "enabled",  # 是否使用深度思考能力
           "reasoning": {"effort": "high"}#思考长度
        }
    },
            temperature=0.1,
            top_p=0.7
        )
        end_time = time.time()
        doubao_18_inference_time = end_time - start_time
        
        # 输出token数和耗时
        doubao_18_prompt_tokens = 'N/A'
        doubao_18_completion_tokens = 'N/A'
        doubao_18_total_tokens = 'N/A'
        if hasattr(doubao_response_18, 'usage'):
            doubao_18_prompt_tokens = getattr(doubao_response_18.usage, 'prompt_tokens', 'N/A')
            doubao_18_completion_tokens = getattr(doubao_response_18.usage, 'completion_tokens', 'N/A')
            doubao_18_total_tokens = getattr(doubao_response_18.usage, 'total_tokens', 'N/A')
        print(f"[{run_id+1}] 豆包1.8模型消耗token数: 输入={doubao_18_prompt_tokens}, 输出={doubao_18_completion_tokens}, 总计={doubao_18_total_tokens}")
        print(f"[{run_id+1}] 豆包1.8模型推理耗时: {doubao_18_inference_time:.2f} 秒")
        
        # 提取bboxes
        print(f"[{run_id+1}] 正在提取bbox信息...")
        qwen_output = qwen_response.choices[0].message.content
        doubao_output_18 = doubao_response_18.choices[0].message.content
        
        qwen_bboxes = extract_bboxes(qwen_output)
        doubao_bboxes_18 = extract_bboxes(doubao_output_18)
        
        # 解析XML获取标准答案bbox
        print(f"[{run_id+1}] 正在解析标准答案文件...")
        groundtruth_bboxes = parse_xml_groundtruth(image_url)
        
        # 获取序列编号
        seq_num = get_next_sequence_number()
        
        # 绘制并保存结果图片
        print(f"[{run_id+1}] 正在绘制结果图片...")
        output_image_path = os.path.join(outputs_visualizations_dir, f"result_with_bboxes_{seq_num}.jpg")
        adjusted_qwen_bboxes, adjusted_doubao_bboxes_18 = draw_bboxes(
            temp_image_path, qwen_bboxes, doubao_bboxes_18, output_image_path
        )
        
        # 计算各模型的IoU
        print(f"[{run_id+1}] 正在计算IoU值...")
        qwen_iou_results = calculate_model_iou(adjusted_qwen_bboxes, groundtruth_bboxes)
        doubao_18_iou_results = calculate_model_iou(adjusted_doubao_bboxes_18, groundtruth_bboxes)
        
        # 保存bbox信息
        print(f"[{run_id+1}] 正在保存bbox信息...")
        bbox_info_path = save_bbox_info(seq_num, adjusted_qwen_bboxes, adjusted_doubao_bboxes_18, 
                                     qwen_iou_results, doubao_18_iou_results)
        
        # 清理临时文件，如果使用的是临时下载的图片
        if "temp_image" in temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        # 计算平均IoU，只包含IoU大于0.1的结果
        def calculate_average_iou(iou_results):
            # 筛选IoU大于0.1的结果
            valid_iou_results = [result for result in iou_results if result['iou'] > 0.1]
            if not valid_iou_results:
                return 0.0
            return sum(result['iou'] for result in valid_iou_results) / len(valid_iou_results)
        
        avg_qwen_iou = calculate_average_iou(qwen_iou_results)
        avg_doubao_18_iou = calculate_average_iou(doubao_18_iou_results)
        
        print(f"[{run_id+1}] 图片处理完成！结果图片已保存为：{output_image_path}")
        print(f"[{run_id+1}] bbox信息已保存为：{bbox_info_path}")
        print(f"[{run_id+1}] Qwen3模型检测到 {len(adjusted_qwen_bboxes)} 个目标，平均IoU：{avg_qwen_iou:.4f}")
        print(f"[{run_id+1}] 豆包1.8模型检测到 {len(adjusted_doubao_bboxes_18)} 个目标，平均IoU：{avg_doubao_18_iou:.4f}")
        print(f"=== 图片 {image_url} 的第 {run_id+1} 次运行处理结束 ===")
        
        # 返回结果信息，用于写入CSV
        result = {
            "image_url": image_url,
            "run_id": run_id,
            "seq_num": seq_num,
            "qwen_output": qwen_output,
            "qwen_prompt_tokens": qwen_prompt_tokens,
            "qwen_completion_tokens": qwen_completion_tokens,
            "qwen_total_tokens": qwen_total_tokens,
            "qwen_inference_time": qwen_inference_time,
            "doubao_18_output": doubao_output_18,
            "doubao_18_prompt_tokens": doubao_18_prompt_tokens,
            "doubao_18_completion_tokens": doubao_18_completion_tokens,
            "doubao_18_total_tokens": doubao_18_total_tokens,
            "doubao_18_inference_time": doubao_18_inference_time,
            "output_image_path": output_image_path,
            "bbox_info_path": bbox_info_path,
            "groundtruth_bboxes": groundtruth_bboxes,
            "qwen_iou_results": qwen_iou_results,
            "doubao_18_iou_results": doubao_18_iou_results,
            "avg_qwen_iou": avg_qwen_iou,
            "avg_doubao_18_iou": avg_doubao_18_iou
        }
        
        return result
    except Exception as e:
        print(f"处理图片 {image_url} 的第 {run_id+1} 次运行时出错: {str(e)}")
        return None

# 主函数
def main():
    all_results = []
    max_concurrent = 5
    runs_per_image = 5
    
    # 为每张图片创建5个任务
    tasks = []
    for image_idx, image_url in enumerate(image_urls):
        for run_id in range(runs_per_image):
            tasks.append((image_url, run_id, all_results))
    
    # 使用线程池并发执行任务
    print(f"开始并发处理 {len(tasks)} 个任务，最大并发数: {max_concurrent}")
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_image_instance, task[0], task[1], task[2]): task for task in tasks}
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"任务 {task} 执行失败: {str(e)}")
    
    # 生成CSV文件
    print("\n正在生成CSV结果文件...")
    generate_csv_results(all_results)
    
    print("\n所有图片处理完成！")
    print(f"总计处理: {len(all_results)} 个成功运行实例")

if __name__ == "__main__":
    main()
