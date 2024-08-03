import argparse
import concurrent.futures
import re
import time
import unicodedata
from io import TextIOWrapper
from os import rename
from pathlib import Path
from shutil import rmtree
from threading import Lock
from typing import List, Tuple

import vapoursynth as vs
import vskernels
from requests import Session
from tqdm import tqdm
from vsmasktools import HardsubLine
from vspreview.api import is_preview
from vstools import depth, get_depth, set_output

core = vs.core

### Vapoursynth part

def get_scene_changes(clip: vs.VideoNode) -> List[Tuple[int, int]]:
    scene_changes = []
    current_start = None
    with tqdm(total=clip.num_frames, desc="Detecting scene changes...", unit="frame") as pbar:
        for n in range(clip.num_frames):
            f = clip.get_frame(n)
            pbar.update(1)
            if f.props["PlaneStatsAverage"] < 0.02:
                continue
            if f.props["_SceneChangePrev"] == 1:
                current_start = n
            elif f.props["_SceneChangeNext"] and current_start is not None:
                scene_changes.append((current_start, n))
                current_start = None

    return scene_changes

def to_timestamp(total_seconds) -> Tuple[str, str, str, str]:
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    
    return (
        f"{hours:02d}",
        f"{minutes:02d}",
        f"{seconds:02d}",
        f"{milliseconds:03d}"
    )

def format_frame_time(start_frame: int, end_frame: int, fpsnum: int, fpsden: int) -> str:
    start_time = to_timestamp(start_frame * fpsden / fpsnum)
    end_time = to_timestamp((end_frame + 1) * fpsden / fpsnum)
    
    start_formatted = f"{start_time[0]}_{start_time[1]}_{start_time[2]}_{start_time[3]}"
    end_formatted = f"{end_time[0]}_{end_time[1]}_{end_time[2]}_{end_time[3]}"
    return f"{start_formatted}__{end_formatted}"

def write_images(clip: vs.VideoNode, scene_changes: List[Tuple[int, int]]):
    # image = clip.fpng.Write(f'images/%d.png')
    for scene_change in scene_changes:
        crop = clip.acrop.AutoCrop(top=0, bottom=0, left=(clip.width / 4), right=(clip.width / 4))
        image = vskernels.Lanczos.scale(crop, format=vs.RGB24).fpng.Write(f'images/%d.png', compression=2)
        image.get_frame(scene_change[0])
        filename = format_frame_time(scene_change[0], scene_change[1], clip.fps_num, clip.fps_den)
        rename(f"images/{scene_change[0]}.png", f'images/{filename}.png')

def get_subtitles(clean: vs.VideoNode, sub: vs.VideoNode) -> vs.VideoNode:
    mask = HardsubLine().get_mask(sub, clean)
    blank = sub.std.BlankClip(format=sub.format.id, keep=True)
    merge = blank.std.MaskedMerge(sub, mask)
    # Use mask for better detection
    scd = mask.misc.SCDetect(0.02).std.PlaneStats()
    return merge.std.CopyFrameProps(scd)


def filter_videos(clean_path: str, clean_offset: int, sub_path: str, sub_offset: int):
    clean = core.lsmas.LWLibavSource(clean_path)[clean_offset:]
    sub = core.lsmas.LWLibavSource(sub_path)[sub_offset:]

    if clean.width != sub.width or clean.height != sub.height:
        clean = vskernels.Catrom.scale(clean, sub.width, sub.height)

    if get_depth(clean) != get_depth(sub):
        clean = depth(clean, get_depth(sub))

    bot_clean = clean.std.Crop(bottom=20, top=590)
    bot_sub = sub.std.Crop(bottom=20, top=590)

    top_clean = clean.std.Crop(bottom=590, top=20)
    top_sub = sub.std.Crop(bottom=590, top=20)

    bot_subtitles = get_subtitles(bot_clean, bot_sub)
    top_subtitles = get_subtitles(top_clean, top_sub)
    if is_preview():
        set_output(top_subtitles.text.FrameProps(props=["_SceneChangePrev", "_SceneChangeNext", "PlaneStatsAverage"]), "top")
        set_output(bot_subtitles.text.FrameProps(props=["_SceneChangePrev", "_SceneChangeNext", "PlaneStatsAverage"]), "bot")
        set_output(sub, "sub")
        set_output(clean, "clean")
        return

    scene_changes_bot = get_scene_changes(bot_subtitles)
    scene_changes_top = get_scene_changes(top_subtitles)

    print("Writing images...")
    write_images(bot_subtitles, scene_changes_bot)
    write_images(top_subtitles, scene_changes_top)

### OCR PART
THREADS = 30

class SRTSubtitle:
    def __init__(self, line_number, start_time, end_time, text_content):
        self.line_number = line_number
        self.start_time = start_time
        self.end_time = end_time
        self.text_content = text_content

    def __str__(self):
        return f"{self.line_number}\n{self.start_time} --> {self.end_time}\n{self.text_content}\n\n"

def ocr(images_dir: Path, srt_file: Path):
    image_extensions = ('*.jpeg', '*.jpg', '*.png', '*.bmp', '*.gif')
    images = []
    for extension in image_extensions:
        images.extend(list(images_dir.rglob(extension)))

    total_images = len(images)
    completed_scans = 0
    scan_lock = Lock()

    srt_dict = {}
    session = Session()

    with tqdm(total=total_images, desc="OCR images", unit="img") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
            future_to_image = {executor.submit(ocr_image, image, index+1, srt_dict, session): image for index, image in enumerate(images)}
            for future in concurrent.futures.as_completed(future_to_image):
                image = future_to_image[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"{image} generated an exception: {exc}")
                else:
                    with scan_lock:
                        completed_scans += 1
                        pbar.update(1)
    write_srt(srt_file, srt_dict)

def ocr_image(image: Path, line: int, srt_dict: List, session: Session):
    img_filename = str(image.absolute())
    img_name = str(image.name)

    try:
        text_content = google_lens_ocr(session, img_filename)
    except Exception:
        print(f"Error processing {img_name}")
        return
    
    text_content = remove_hieroglyphs_unicode(text_content)

    try:
        start_hour = img_name.split('_')[0][:2]
        start_min = img_name.split('_')[1][:2]
        start_sec = img_name.split('_')[2][:2]
        start_micro = img_name.split('_')[3][:3]

        end_hour = img_name.split('__')[1].split('_')[0][:2]
        end_min = img_name.split('__')[1].split('_')[1][:2]
        end_sec = img_name.split('__')[1].split('_')[2][:2]
        end_micro = img_name.split('__')[1].split('_')[3][:3]
    except IndexError:
        print(f"Error processing {img_name}: Filename format is incorrect. Please ensure the correct format is used.")
        return

    start_time = f'{start_hour}:{start_min}:{start_sec},{start_micro}'
    end_time = f'{end_hour}:{end_min}:{end_sec},{end_micro}'

    subtitle = SRTSubtitle(
        line,
        start_time,
        end_time,
        text_content
    )
    srt_dict[line] = subtitle


def google_lens_ocr(session: Session, img_path: str) -> str:
    regex = re.compile(r">AF_initDataCallback\(({key: 'ds:1'.*?)\);</script>")
    timestamp = int(time.time() * 1000)
    url = f"https://lens.google.com/v3/upload?hl=en-VN&re=df&stcs={timestamp}&vpw=1500&vph=1500"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    }
    cookies = {"SOCS": "CAESEwgDEgk0ODE3Nzk3MjQaAmVuIAEaBgiA_LyaBg"}

    files = {"encoded_image": ("screenshot.png", (open(img_path, 'rb')), "image/png")}
    res = session.post(
        url, files=files, headers=headers, cookies=cookies, timeout=40
    )
    match = regex.search(res.text)
    sideChannel = "sideChannel"
    null = None
    key = "key"
    data = "data"
    true = True
    false = False
    lens_object = eval(match.group(1))
    if "errorHasStatus" in lens_object:
        raise Exception(False, "Unknown Lens error!")
    
    text = lens_object["data"][3][4][0]
        
    
    return "\n".join(text[0])

def write_srt(srt_file: TextIOWrapper, srt_dict: List):
    cleaned_srt = []
    previous_subtitle = None
    for _, subtitle in sorted(srt_dict.items()):
        if not previous_subtitle:
            cleaned_srt.append(subtitle)
            previous_subtitle = subtitle
            continue
        if not subtitle.text_content or subtitle.text_content.isspace():
            continue
        if previous_subtitle.text_content == subtitle.text_content:
            merged_subtitle = SRTSubtitle(
                line_number=previous_subtitle.line_number,
                start_time=previous_subtitle.start_time,
                end_time=subtitle.end_time,
                text_content=previous_subtitle.text_content,
            )
            cleaned_srt.pop()
            cleaned_srt.append(merged_subtitle)
        else:
            subtitle.line = previous_subtitle.line_number + 1
            cleaned_srt.append(subtitle)
        previous_subtitle = cleaned_srt[-1]
    
    for subtitle in cleaned_srt:
        srt_file.write(str(subtitle))
    srt_file.close()

def remove_hieroglyphs_unicode(text):
    result = ""
    for char in text:
        if unicodedata.category(char) not in ('Lo'):
            result += char
    return result.lstrip()

### CLI Part
def create_arg_parser():
    parser = argparse.ArgumentParser(description='A simple tool to OCR Muse subtitles')

    parser.add_argument("clean", help="Path to the clean source.", metavar='<clean>')
    parser.add_argument("sub", help="Path to the hardsub source.", metavar='<hardsub>')
    parser.add_argument('-o', '--output', default="output_subtitles", dest='output_subtitles', metavar='<outputname>',
                        help='Output subtitles filename')
    parser.add_argument('--offset-clean', default=0, type=int, dest='offset_clean', metavar='<frame_offset>',
                        help='Offset frame of clean')
    parser.add_argument('--offset-sub', default=0, type=int, dest='offset_sub', metavar='<frame_offset>',
                        help='Offset frame of sub')
    return parser


if is_preview():
    filter_videos(r"clean.mp4", 0, r"sub.mp4", 0)
else:
    args = create_arg_parser().parse_args()
    current_directory = Path(Path.cwd())
    images_dir = Path(f'{current_directory}/images')
    srt_file = open(Path(f'{current_directory}/{args.output_subtitles}.srt'), 'w', encoding='utf-8')

    if images_dir.exists():
        rmtree(images_dir)
    images_dir.mkdir()

    filter_videos(args.clean, args.offset_clean, args.sub, args.offset_sub)
    ocr(images_dir, srt_file)

    print("Done")