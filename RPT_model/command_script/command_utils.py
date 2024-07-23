import os
import subprocess
import torch
import cv2
import random
from einops import rearrange
import numpy as np
import psutil
import time
import h5py_cache


def break_text(text, break_points=(",", ".", ";", ":", "?", "!")):
    """
    Break the text by the specified break points (punctuation) and return as a list of lines.
    """
    # Check if the text contains any breakpoints
    if not any(bp in text for bp in break_points):
        # If not, replay the new lines with a comma
        text = text.replace("\n", ", ")

    lines = []
    current_line = ""

    # Replace new lines with a space
    text = text.replace("\n", " ")

    for char in text:
        current_line += char
        if char in break_points:
            # Trim the break_point characters
            for bp in break_points:
                current_line = current_line.rstrip(bp)
            lines.append(current_line.strip())
            current_line = ""
    if current_line:  # if any text remains that hasn't been appended
        lines.append(current_line.strip())
    # Filter out any empty lines
    lines = [line for line in lines if line]
    return lines


def modify_transcription(text):# 纠正一些容易语音识别错误的文字
    """Modify the transcription text."""
    text = text.lower()
    text = text.replace("the back", "the bag")
    text = text.replace("the paper", "the gripper")
    text = text.replace("creeper", "gripper")
    text = text.replace("the grip ", "the gripper ")
    text = text.replace("the bat", "the bag")
    text = text.replace("the sharky", "the sharpie")
    text = text.replace("peek up", "pick up")
    text = text.replace("cling", "clean")
    text = text.replace("love", "left")
    text = text.replace("rigiht", "right")
    text = text.replace("want last", "want less")
    text = text.replace("one last", "less")
    text = text.replace("one more", "more")

    return modify_real_time(text)


def modify_real_time(text): # 去掉一些感谢词
    text = text.lower()
    # Handle hallucinations
    text = text.replace("thanks for watching", "")
    text = text.replace("thank you for watching", "")
    text = text.replace("thank you", "")
    text = text.replace("for watching", "")
    text = text.replace("i'll see you next time", "")
    text = text.replace("i'll see you guys next time", "")
    text = text.replace("please subscribe and thumb up", "")
    text = text.replace("bye", "")
    text = text.replace("see you next time", "")

    # Strip away unwanted characters at the end of the string and any leading/trailing whitespace
    return text.rstrip(",.;:?!").strip()


def generate_transcription(
    dataset_dir,
    dataset_name,
    model_dir="whisper_models",
    model="large-v2",
    language="English",
):
    """Generate transcription using the whisper command."""
    input_path = os.path.join(dataset_dir, dataset_name + ".wav")
    output_dir = dataset_dir
    command = [
        "whisper",
        input_path,
        "--output_dir",
        output_dir,
        "--model_dir",
        model_dir,
        "--language",
        language,
        "--model",
        model,
        "--output_format",
        "txt",
    ]
    subprocess.run(command, check=True)


def initialize_model_and_tokenizer(encoder):
    from transformers import (
        DistilBertTokenizer,
        DistilBertModel,
        CLIPTextModel,
        CLIPTokenizer,
    )
    if encoder == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    elif encoder == "clip":
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    else:
        raise ValueError("Unknown encoder type. Please use 'distilbert' or 'clip'.")
    # print(f"initialize_model_and_tokenizer")
    return tokenizer, model


def encode_text(text, encoder, tokenizer, model):
    if encoder == "distilbert":
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the representation of the [CLS] token
        return outputs.last_hidden_state[:, 0, :].numpy().tolist()
    elif encoder == "clip":
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=77
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # Average the embeddings across tokens for CLIP to get a sentence representation
        return outputs.last_hidden_state.mean(dim=1).numpy().tolist()


def crop_resize(image, crop_h=240, crop_w=320, resize_h=480, resize_w=640, resize=True):
    """
    Helper function to crop the bottom middle (offset by 20 pixels) and resize
    """
    h, w, _ = image.shape
    y1 = h - crop_h - 20  # Subtracting 20 to start 20 pixels above the bottom
    x1 = (w - crop_w) // 2
    cropped = image[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (resize_w, resize_h)) if resize else cropped


def random_crop(image, crop_percentage=0.95):
    """
    Crop the given image by a random percentage without going out of boundary.
    """
    h, w, _ = image.shape
    new_h, new_w = int(h * crop_percentage), int(w * crop_percentage)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    cropped_image = image[top : top + new_h, left : left + new_w, :]
    return cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_LINEAR)
    # return cropped_image


def center_crop(image, crop_percentage=0.95):
    """
    Crop the center of the given image by a specified percentage.
    """
    h, w, _ = image.shape
    new_h, new_w = int(h * crop_percentage), int(w * crop_percentage)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped_image = image[top : top + new_h, left : left + new_w, :]
    return cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_LINEAR)
    # return cropped_image


def is_multi_gpu_checkpoint(state_dict):
    """
    Check if the given state_dict is from a model trained on multiple GPUs using DataParallel.
    """
    # Check if any key starts with 'module.'
    return any(k.startswith("model.module.") for k in state_dict.keys())


def get_auto_index(dataset_dir, dataset_name_prefix="", data_suffix="hdf5"):
    max_idx = 5000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.isfile(
            os.path.join(dataset_dir, f"{dataset_name_prefix}episode_{i}.{data_suffix}")
        ):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def visualize_language_correction(
    curr_image, predicted_instruction, command, save_dir, episode_idx
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3

    concat_images = [
        rearrange(img.cpu().numpy(), "c h w -> h w c") for img in curr_image.squeeze(0)
    ]
    concat_image = np.concatenate(concat_images, axis=1)
    concat_image = (concat_image * 255).astype(np.uint8)
    concat_image = np.ascontiguousarray(
        concat_image
    )  # Ensure it's contiguous in memory
    concat_image = cv2.cvtColor(
        concat_image, cv2.COLOR_RGB2BGR
    )  # Convert to BGR for OpenCV

    # Calculate the height needed for the padding based on text size and some margins
    (text_width_1, text_height_1), _ = cv2.getTextSize(
        "Prediction: " + predicted_instruction, font, font_scale, font_thickness
    )
    (text_width_2, text_height_2), _ = cv2.getTextSize(
        "User: " + command, font, font_scale, font_thickness
    )

    padding_height = (
        text_height_1 + text_height_2 + 5 * font_thickness
    )  # 5 times font_thickness for more space between lines
    padding = np.zeros((padding_height, concat_image.shape[1], 3), dtype=np.uint8)

    # Stack the padding on top of the image
    concat_image = np.vstack((padding, concat_image))

    # Add the text onto the padded region
    cv2.putText(
        concat_image,
        "Prediction: " + predicted_instruction,
        (10, text_height_1 + font_thickness),
        font,
        font_scale,
        (0, 0, 255),
        font_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        concat_image,
        "User: " + command,
        (10, text_height_1 + 4 * font_thickness + text_height_2),
        font,
        font_scale,
        (255, 255, 0),
        font_thickness,
        lineType=cv2.LINE_AA,
    )  # Darker shade of green

    save_path = os.path.join(save_dir, f"episode_{episode_idx}.png")
    cv2.imwrite(save_path, concat_image)


def create_dataset_path(dataset_dir):
    episode_idx = get_auto_index(dataset_dir)
    dataset_name = f"episode_{episode_idx}"
    print(f"Dataset name: {dataset_name}")
    dataset_path = os.path.join(dataset_dir, dataset_name)
    return dataset_path, episode_idx


def save_trajectory(
    dataset_path, timesteps, actions, camera_names, command, image_list=None
):
    # save trajectory
    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'
    - option                (1,)          'int'

    action                  (14,)         'float64'
    """

    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/effort": [],
        "/observations/option": [],
        "/action": [],
    }
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict["/observations/qpos"].append(ts.observation["qpos"])
        data_dict["/observations/qvel"].append(ts.observation["qvel"])
        data_dict["/observations/effort"].append(ts.observation["effort"])
        option_expanded = np.expand_dims(np.array(ts.observation["option"]), axis=0)
        data_dict["/observations/option"].append(option_expanded)
        data_dict["/action"].append(action)
        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(
                ts.observation["images"][cam_name]
            )

    COMPRESS = True

    if COMPRESS:
        # JPEG compression
        t0 = time.time()
        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            90,
        ]  # tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list_data = data_dict[f"/observations/images/{cam_name}"]
            compressed_list = []
            compressed_len.append([])
            for image in image_list_data:
                result, encoded_image = cv2.imencode(
                    ".jpg", image, encode_param
                )  # 0.02 sec # cv2.imdecode(encoded_image, 1)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f"/observations/images/{cam_name}"] = compressed_list
        # print(f'compression: {time.time() - t0:.2f}s')

        # pad so it has same length
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f"/observations/images/{cam_name}"]
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype="uint8")
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f"/observations/images/{cam_name}"] = padded_compressed_image_list
        # print(f'padding: {time.time() - t0:.2f}s')

    # HDF5
    t0 = time.time()
    max_timesteps = len(data_dict["/action"])
    with h5py_cache.File(
        dataset_path + ".hdf5", "w", chunk_cache_mem_size=1024**2 * 2
    ) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = COMPRESS
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in camera_names:
            if COMPRESS:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, padded_size),
                    dtype="uint8",
                    chunks=(1, padded_size),
                )
            else:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, 480, 640, 3),
                    dtype="uint8",
                    chunks=(1, 480, 640, 3),
                )
        _ = obs.create_dataset("qpos", (max_timesteps, 14))
        _ = obs.create_dataset("qvel", (max_timesteps, 14))
        _ = obs.create_dataset("effort", (max_timesteps, 14))
        _ = obs.create_dataset("option", (max_timesteps, 1))
        _ = root.create_dataset("action", (max_timesteps, 14))

        for name, array in data_dict.items():
            root[name][...] = array

        if COMPRESS:
            _ = root.create_dataset("compress_len", (len(camera_names), max_timesteps))
            root["/compress_len"][...] = compressed_len

    # save command in a txt file
    command_path = dataset_path + ".txt"
    with open(command_path, "w") as f:
        f.write(command)

    # print(f'Saving: {time.time() - t0:.1f} secs')
    return ts, image_list


# Automatically kill the job if it’s going to exceed the memory limit.
def memory_monitor():
    while True:
        available_memory = psutil.virtual_memory().available / (
            1024**2
        )  # Available memory in MB
        if (
            available_memory < 1000
        ):  # MEMORY_BUFFER_MB: The amount of memory to ensure remains free
            print(
                f"Available memory is too low! {available_memory:.2f}MB left. Terminating..."
            )
            os._exit(1)
        time.sleep(5)
