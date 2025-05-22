import os
import shutil
import json
import numpy as np
from PIL import Image, ImageFile
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from .segmentation import MySegFormer_0409

ImageFile.LOAD_TRUNCATED_IMAGES = True

# The SegFormer weights which are knowledge distillation from SAM
weight_dir = "/Users/eatappleagain/trial/RipplesDetection/ripples_api/app/resource/state_dict/segformer_data_size_300.pth"
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

model = MySegFormer_0409(num_classes=2, backbone="b0")
_state = torch.load(weight_dir, map_location=device)
model.load_state_dict(_state)
model.to(device).eval()

_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

def _predict_mask(img: Image.Image) -> np.ndarray:
    inp = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(inp)["out"]
    mask = output.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
    return mask

def _apply_magic(mask: np.ndarray, magic_mask_path: str) -> np.ndarray:
    if magic_mask_path and os.path.isfile(magic_mask_path):
        magic = Image.open(magic_mask_path).convert("L")
        m_arr = np.array(magic)
        mask[m_arr == 0] = 0
    return mask

def current_feed_status(
    frames_dir: str,
    resource_dir: str,
    magic_mask_path: str = None,
    expected_range: tuple[int,int] = (10000,30000)
) -> dict:
    """
    分析最新一張frame的水花 pixel 數量，並根據 expected_range 回報魚隻搶食狀態
    輸出：
      - overlay 圖存成 resource_dir/overlay_<timestamp>.png
      - pixel 把水花pixel數量存在 JSON 
    """
    raw = os.path.join(frames_dir, "raw_image")
    # Get the latest image
    fn = sorted([f for f in os.listdir(raw) if f.endswith(".png")])[-1]
    image = Image.open(os.path.join(raw, fn)).convert("RGB")

    mask = _predict_mask(image)
    mask = _apply_magic(mask, magic_mask_path)
    pixel_count = int(np.sum(mask == 1))

    # save mask
    os.makedirs(resource_dir, exist_ok=True)
    overlay_arr = np.array(image.resize(mask.shape[::-1]))/255.0
    overlay_arr[mask == 1, 0] = 1.0
    overlay = Image.fromarray((overlay_arr*255).astype(np.uint8))
    overlay_path = os.path.join(resource_dir, f"overlay_{fn}")
    overlay.save(overlay_path)

    result = {
        "frame": fn,
        "pixel_count": pixel_count,
        "overlay": overlay_path,
    }
    low, high = expected_range
    if pixel_count < low:
        status = "低於預期"
    elif pixel_count > high:
        status = "高於預期"
    else:
        status = "符合預期"
    result["status"] = status

    # record in json
    json_path = os.path.join(resource_dir, "pixel_counts_current.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(result, jf, ensure_ascii=False, indent=2)
    result["json"] = json_path

    return result

def past_feed_status(
    frames_dir: str,
    resource_dir: str,
    magic_mask_path: str = None
) -> dict:
    raw = os.path.join(frames_dir, "raw_image")
    fns = sorted([f for f in os.listdir(raw) if f.endswith(".png")])
    counts = []
    for fn in fns:
        img = Image.open(os.path.join(raw, fn)).convert("RGB")
        mask = _predict_mask(img)
        mask = _apply_magic(mask, magic_mask_path)
        counts.append({"time": fn, "pixel_number": int(np.sum(mask == 1))})

    os.makedirs(resource_dir, exist_ok=True)
    # record in json
    date = os.path.basename(frames_dir).split("-")[-1]
    json_path = os.path.join(resource_dir, f"pixel_counts_{date}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(counts, jf, ensure_ascii=False, indent=2)

    times = [datetime.strptime(item['time'], '%Y-%m-%d-%H-%M-%S.png') for item in counts]
    pixel_numbers = [item['pixel_number'] for item in counts]
    json_date = times[0].strftime('%Y-%m-%d')

    # ripple trend plot
    trend_path = os.path.join(resource_dir, f"trend_{date}.png")
    plt.figure()
    plt.plot(times, pixel_numbers, marker="o")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlabel("Time of Day")
    plt.ylabel("Pixel Count")
    plt.title(f"Ripple Trend {json_date}")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(trend_path)
    plt.close()

    # heat map
    heatmap_path = os.path.join(resource_dir, f"heatmap_{date}.png")

    pixel_array = np.array(pixel_numbers).reshape(-1, 1)
    times_num = mdates.date2num(times)

    fig, ax = plt.subplots(figsize=(2, 8))
    img = ax.imshow(
        pixel_array,
        aspect='auto',
        cmap='hot',
        extent=[0, 1, times_num.max(), times_num.min()]
    )
    ax.set_xticks([])
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    cbar = plt.colorbar(img, ax=ax, aspect=10)
    cbar.set_label('Pixel Number of Water Splashes')
    ax.set_xlabel(json_date)
    ax.set_ylabel('Time of Day')
    plt.title('Splash Activity')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    return {
        "counts": counts,
        "json": json_path,
        "trend_chart": trend_path,
        "heatmap": heatmap_path
    }