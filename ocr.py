import argparse
import concurrent.futures
import os
import re
import unicodedata
from io import StringIO
from itertools import count
from os import rename
from pathlib import Path
from shutil import rmtree
from threading import Lock
from typing import Dict, List, Tuple

import json5
import lxml.html
import vapoursynth as vs
import vskernels
from requests import Session
from tqdm import tqdm
from vsmasktools import HardsubLine
from vspreview.api import is_preview
from vstools import depth, get_depth, iterate, set_output

core = vs.core

### Vapoursynth part
class Filter:
    def __init__(self, clean_path: str, clean_offset: int, sub_path: str, sub_offset: int):
        self.clean_path:str = clean_path
        self.clean_offset:int = clean_offset
        self.sub_path:str = sub_path
        self.sub_offset:int = sub_offset

    def filter_videos(self):
        clean = core.lsmas.LWLibavSource(self.clean_path)[self.clean_offset:]
        sub = core.lsmas.LWLibavSource(self.sub_path)[self.sub_offset:]

        if clean.width != sub.width or clean.height != sub.height:
            clean = vskernels.Catrom.scale(clean, sub.width, sub.height)

        if get_depth(clean) != get_depth(sub):
            clean = depth(clean, get_depth(sub))
        # 720p 590, 1080p 860
        sub_height = sub.height - (sub.height / 5)
        sub_vert = 20

        bot_clean = clean.std.Crop(bottom=sub_vert, top=sub_height)
        bot_sub = sub.std.Crop(bottom=sub_vert, top=sub_height)

        top_clean = clean.std.Crop(bottom=sub_height, top=sub_vert)
        top_sub = sub.std.Crop(bottom=sub_height, top=sub_vert)

        bot_subtitles = self._get_subtitles(bot_clean, bot_sub)
        top_subtitles = self._get_subtitles(top_clean, top_sub)
        if is_preview():
            # set_output(top_subtitles.text.FrameProps(props=["_SceneChangePrev", "_SceneChangeNext", "PlaneStatsAverage"]), "top")
            # set_output(bot_subtitles.text.FrameProps(props=["_SceneChangePrev", "_SceneChangeNext", "PlaneStatsAverage"]), "bot")
            set_output(top_subtitles.text.FrameProps(), "top")
            set_output(bot_subtitles.text.FrameProps(), "bot")
            set_output(sub.text.FrameProps(), "sub")
            set_output(clean.text.FrameProps(), "clean")
            return

        scene_changes_bot = self._get_scene_changes(bot_subtitles)
        scene_changes_top = self._get_scene_changes(top_subtitles)

        print("Writing images...")
        self._write_images(bot_subtitles, scene_changes_bot)
        self._write_images(top_subtitles, scene_changes_top)

    def _get_subtitles(self, clean: vs.VideoNode, sub: vs.VideoNode) -> vs.VideoNode:
        mask = HardsubLine().get_mask(sub, clean)
        scd = mask.misc.SCDetect(0.02).std.PlaneStats()
        mask = iterate(mask, core.std.Maximum, 15)
        blank = sub.std.BlankClip(format=sub.format.id, keep=True)
        merge = blank.std.MaskedMerge(sub, mask)
        # Use mask for better detection
        return merge.std.CopyFrameProps(scd)


    def _get_scene_changes(self, clip: vs.VideoNode) -> List[Tuple[int, int]]:
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
    
    def _write_images(self, clip: vs.VideoNode, scene_changes: List[Tuple[int, int]]):
        for scene_change in scene_changes:
            crop = clip.acrop.AutoCrop(top=0, bottom=0, left=(clip.width / 4), right=(clip.width / 4))
            image = crop.resize.Point(format=vs.RGB24, matrix_in_s='709').imwri.Write(imgformat="JPEG", filename=f'images/%d.png', quality=95)
            image.get_frame(scene_change[0])
            filename = self._format_frame_time(scene_change[0], scene_change[1], clip.fps_num, clip.fps_den)

            dst_path = Path(f"images/{filename}.png")
            for i in count(1):
                if not os.path.exists(dst_path):
                    break
                dst_path = Path(f"images/{filename}_{i}.png")

            rename(f"images/{scene_change[0]}.png", dst_path)

    def _format_frame_time(self, start_frame: int, end_frame: int, fpsnum: int, fpsden: int) -> str:
        start_time = self._to_timestamp(start_frame * fpsden / fpsnum)
        end_time = self._to_timestamp((end_frame + 1) * fpsden / fpsnum)
        
        start_formatted = f"{start_time[0]}_{start_time[1]}_{start_time[2]}_{start_time[3]}"
        end_formatted = f"{end_time[0]}_{end_time[1]}_{end_time[2]}_{end_time[3]}"
        return f"{start_formatted}__{end_formatted}"
    
    def _to_timestamp(self, total_seconds) -> Tuple[str, str, str, str]:
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

### OCR PART
class SRTSubtitle:
    def __init__(self, line_number, start_time, end_time, text_content):
        self.line_number = line_number
        self.start_time = start_time
        self.end_time = end_time
        self.text_content = text_content

    def __str__(self):
        return f"{self.line_number}\n{self.start_time} --> {self.end_time}\n{self.text_content}\n\n"
    
class Lens:
    LENS_ENDPOINT = f"https://lens.google.com/v3/upload"
    HEADERS = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': 'https://lens.google.com',
        'Referer': 'https://lens.google.com/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    COOKIES = {"SOCS": "CAESEwgDEgk0ODE3Nzk3MjQaAmVuIAEaBgiA_LyaBg"}

    DOUBLE_QUOTE_REGEX = re.compile("|".join([
        "«",
        "‹",
        "»",
        "›",
        "„",
        "“",
        "‟",
        "”",
        "❝",
        "❞",
        "❮",
        "❯",
        "〝",
        "〞",
        "〟",
        "＂",
    ]))

    SINGLE_QUOTE_REGEX = re.compile("|".join([
        "‘",
        "‛",
        "’",
        "❛", 
        "❜", 
        "`", 
        "´", 
        "‘", 
        "’"
    ]))

    def __init__(self):
        self.session = Session()

    def lens_ocr(self, img_path: str) -> str:
        files = {
            "encoded_image": ("screenshot.png", (open(img_path, 'rb')), "image/png")
        }

        res = self.session.post(
            self.LENS_ENDPOINT, files=files, headers=self.HEADERS, cookies=self.COOKIES, timeout=40
        )
        if res.status_code != 200:
            raise Exception(f"Failed to upload image. Status code: {res.status_code}")

        tree = lxml.html.parse(StringIO(res.text))
        r = tree.xpath("//script[@class='ds:1']")
        lens_object = json5.loads(r[0].text[len("AF_initDataCallback("):-2])
        data = lens_object['data']

        try:
            text = data[3][4][0][0]
            if isinstance(text, list):
                text =  "\n".join(text)
        except (IndexError, TypeError):
            text = ""

        text = self._fix_quotes(text)
        text = self._remove_hieroglyphs_unicode(text)
        text = self._apply_punctuation_and_spacing(text)
    
        return text
    
    def _remove_hieroglyphs_unicode(self, text: str) -> str:
        result = ""
        for char in text:
            if unicodedata.category(char) not in ('Lo'):
                result += char
        return result.strip()

    def _apply_punctuation_and_spacing(self, text: str) -> str:
        # Remove extra spaces before punctuation
        text = re.sub(r'\s+([,.!?…])', r'\1', text)
        
        # Ensure single space after punctuation, except for multiple punctuation marks
        text = re.sub(r'([,.!?…])(?!\s)(?![,.!?…])', r'\1 ', text)
        
        # Remove space between multiple punctuation marks
        text = re.sub(r'([,.!?…])\s+([,.!?…])', r'\1\2', text)
        
        return text.strip()
    
    def _fix_quotes(self, text: str) -> str:
        text = self.SINGLE_QUOTE_REGEX.sub("'", text)
        text = self.DOUBLE_QUOTE_REGEX.sub('"', text)
        return text

class OCR_Subtitles:
    THREADS = 30
    IMAGE_EXTENSIONS = ('*.jpeg', '*.jpg', '*.png', '*.bmp', '*.gif')
    def __init__(self, images_dir: Path, srt_file: Path) -> None:
        self.images: List[str] = []
        self.srt_dict: Dict[str, SRTSubtitle] = {}
        self.scan_lock:Lock = Lock()
        self.lens:Lens = Lens()
        self.images_dir:Path = images_dir
        self.srt_file:Path = srt_file
        self.completed_scans:int = 0

    def ocr(self):
        self.completed_scans = 0

        for extension in self.IMAGE_EXTENSIONS:
            self.images.extend(list(self.images_dir.rglob(extension)))

        total_images = len(self.images)
        
        with tqdm(total=total_images, desc="OCR images", unit="img") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.THREADS) as executor:
                future_to_image = {executor.submit(self._process_image, image, index+1): image for index, image in enumerate(self.images)}
                for future in concurrent.futures.as_completed(future_to_image):
                    image = future_to_image[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"{image} generated an exception: {exc}")
                    else:
                        with self.scan_lock:
                            self.completed_scans += 1
                            pbar.update(1)
        self._write_srt()

    def _process_image(self, image: Path, line: int):
        img_filename = str(image.absolute())
        img_name = str(image.name)

        try:
            text_content = self.lens.lens_ocr(img_filename)
        except Exception:
            print(f"Error processing {img_name}: {Exception}")

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
        self.srt_dict[line] = subtitle

    def _write_srt(self):
        cleaned_srt = []
        previous_subtitle = None
        for _, subtitle in sorted(self.srt_dict.items()):
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
            self.srt_file.write(str(subtitle))
        self.srt_file.close()

### CLI Part
def create_arg_parser():
    parser = argparse.ArgumentParser(description='A simple tool to OCR Muse subtitles')

    parser.add_argument('-o', '--output', default="output_subtitles", dest='output_subtitles', metavar='<outputname>',
                        help='Output subtitles filename')
    parser.add_argument('--filter', action=argparse.BooleanOptionalAction, default=True,
                        help='Skip filter and generate image step')
    filter_group = parser.add_argument_group('Filter options (required if --filter is enable)')
    filter_group.add_argument("clean", nargs='?', help="Path to the clean source.", metavar='<clean>')
    filter_group.add_argument("sub", nargs='?', help="Path to the hardsub source.", metavar='<hardsub>')
    filter_group.add_argument('--offset-clean', default=0, type=int, dest='offset_clean', metavar='<frame_offset>',
                        help='Offset frame of clean')
    filter_group.add_argument('--offset-sub', default=0, type=int, dest='offset_sub', metavar='<frame_offset>',
                        help='Offset frame of sub')
    
    return parser

if is_preview():
    filter = Filter(r"clean.mkv", 0, r"sub.mkv", 0)
    filter.filter_videos()
elif __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    if args.filter and (not args.clean or not args.sub):
        parser.error("The 'clean' and 'sub' arguments are required when '--filter' is enable.")
    current_directory = Path(Path.cwd())
    images_dir = Path(f'{current_directory}/images')
    srt_file = open(Path(f'{current_directory}/{args.output_subtitles}.srt'), 'w', encoding='utf-8')


    if args.filter:
        if images_dir.exists():
            rmtree(images_dir)
        images_dir.mkdir()
        filter = Filter(args.clean, args.offset_clean, args.sub, args.offset_sub)
        filter.filter_videos()

    engine = OCR_Subtitles(images_dir, srt_file)
    engine.ocr()

    print("Done")