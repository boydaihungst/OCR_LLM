import argparse
import concurrent.futures
import glob
import re
import unicodedata
from io import StringIO
from os import rename
from pathlib import Path
from shutil import rmtree
from threading import Lock
from typing import Dict, List, Optional, Tuple

import json5
import lxml.html
import vapoursynth as vs
import vskernels
from requests import Session
from rich.console import Console
from rich.progress import (BarColumn, Progress, ProgressColumn, Task, TaskID,
                           Text, TextColumn, TimeRemainingColumn)
from vsmasktools import HardsubLine
from vspreview.api import is_preview
from vstools import (clip_async_render, depth, get_depth, get_w, iterate,
                     set_output)

core = vs.core

### Vapoursynth part
class Filter:
    def __init__(self, clean_path: str, clean_offset: int, hardsub_path: str, sub_offset: int, images_dir: Path):
        self.clean_path:str = clean_path
        self.clean_offset:int = clean_offset
        self.hardsub_path:str = hardsub_path
        self.sub_offset:int = sub_offset
        self.images_dir:Path = images_dir

    def filter_videos(self):
        clean = core.lsmas.LWLibavSource(self.clean_path)[self.clean_offset:]
        hardsub = core.lsmas.LWLibavSource(self.hardsub_path)[self.sub_offset:]

        if hardsub.height > 720:
            hardsub = vskernels.Bilinear.scale(hardsub, width=get_w(720, hardsub), height=720)
        if clean.width != hardsub.width or clean.height != hardsub.height:
            clean = vskernels.Bilinear.scale(clean, hardsub.width, hardsub.height)

        if get_depth(clean) != get_depth(hardsub):
            clean = depth(clean, get_depth(hardsub))

        sub_height = hardsub.height - (hardsub.height / 5)
        sub_vert = 20

        bot_clean = clean.std.Crop(bottom=sub_vert, top=sub_height)
        bot_hardsub = hardsub.std.Crop(bottom=sub_vert, top=sub_height)

        top_clean = clean.std.Crop(bottom=sub_height, top=sub_vert)
        top_hardsub = hardsub.std.Crop(bottom=sub_height, top=sub_vert)

        bot_subtitles = self._get_subtitles(bot_clean, bot_hardsub)
        top_subtitles = self._get_subtitles(top_clean, top_hardsub)
        
        def merge_props(n, f):
            top_f = f[0]
            bot_f = f[1]
            sub_f = f[2].copy()
            for prop in top_f.props:
                sub_f.props[f"top{prop}"] = top_f.props[prop]
            for prop in bot_f.props:
                sub_f.props[f"bot{prop}"] = bot_f.props[prop]
            return sub_f

        blank = hardsub.std.BlankClip(format=hardsub.format.id, keep=True)
        mergeProps = blank.std.ModifyFrame(merge_props, clips=[top_subtitles, bot_subtitles, blank])

        if is_preview():
            set_output(top_subtitles.text.FrameProps(props=["_SceneChangePrev", "_SceneChangeNext", "PlaneStatsAverage"]), "top")
            set_output(bot_subtitles.text.FrameProps(props=["_SceneChangePrev", "_SceneChangeNext", "PlaneStatsAverage"]), "bot")
            set_output(hardsub.text.FrameProps(), "sub")
            set_output(clean.text.FrameProps(), "clean")
            return
        
        scene_changes = self._get_scene_changes(mergeProps, bot_subtitles, top_subtitles)
        self._rename_images(scene_changes, hardsub.fps_num, hardsub.fps_den)

    def _get_subtitles(self, clean: vs.VideoNode, hardsub: vs.VideoNode) -> vs.VideoNode:
        mask = HardsubLine().get_mask(hardsub, clean)
         # Use mask for better detection
        scd = mask.misc.SCDetect(0.02).std.PlaneStats()
        mask = iterate(mask, core.std.Maximum, 15)
        blank = hardsub.std.BlankClip(format=hardsub.format.id, keep=True)
        merge = blank.std.MaskedMerge(hardsub, mask)
        return merge.std.CopyFrameProps(scd)
    
    def _get_scene_changes(self, clip: vs.VideoNode, bot_clip: vs.VideoNode, top_clip: vs.VideoNode) -> List[Tuple[int, int, str]]:
        scene_changes = []
        current_start = {'top': None, 'bot': None}
        
        def process_frame(n: int, f: vs.VideoFrame):
            nonlocal scene_changes, current_start
            for location in ['top', 'bot']:
                if f.props[f"{location}PlaneStatsAverage"] < 0.02:
                    continue
                if f.props[f"{location}_SceneChangePrev"] == 1:
                    current_start[location] = n
                elif f.props[f"{location}_SceneChangeNext"] == 1 and current_start[location] is not None:
                    source_clip = bot_clip if location == 'bot' else top_clip

                    crop_value = int(source_clip.width / 3)
                    crop_value = crop_value if crop_value % 2 == 0 else crop_value - 1
                    crop = source_clip.acrop.AutoCrop(top=0, bottom=0, left=crop_value, right=crop_value)
                    crop = vskernels.Point.scale(crop, format=vs.RGB24, matrix_in_s='709')

                    images = crop.imwri.Write(imgformat="JPEG", filename=f'{self.images_dir}/{location}_%d.jpg', quality=95)
                    images.get_frame(current_start[location])

                    scene_changes.append((current_start[location], n, location))
                    current_start[location] = None

        clip_async_render(clip, callback=process_frame, progress="Detecting scene changes...", async_requests=False)

        return scene_changes

    def _rename_images(self, scene_changes: List[Tuple[int, int, str]], fpsnum: int, fpsden: int):
        for scene_change in scene_changes:
            filename = self._format_frame_time(scene_change[0], scene_change[1], fpsnum, fpsden)
            dst_path = Path(f"{self.images_dir}/{filename}.jpg")
            i = 1
            while dst_path.exists():
                dst_path = Path(f"{self.images_dir}/{filename}_{i}.jpg")
                i += 1
            if Path(f"{self.images_dir}/{scene_change[2]}_{scene_change[0]}.jpg").exists():
                rename(f"{self.images_dir}/{scene_change[2]}_{scene_change[0]}.jpg", dst_path)
            else:
                print(f"Image {scene_change[2]}_{scene_change[0]}.jpg not found")
        
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
class OCRSpeedColumn(ProgressColumn):
    """Progress rendering."""

    def render(self, task: Task) -> Text:
        """Render bar."""

        return Text(f"{task.speed or 0:.02f} images/s")
    
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
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    COOKIES = {"SOCS": "CAESHAgBEhJnd3NfMjAyMjA5MjktMF9SQzEaAnJvIAEaBgiAkvOZBg"}

    DOUBLE_QUOTE_REGEX = re.compile("|".join([
        "«", "‹", "»", "›", "„", "“", "‟", "”", "❝", "❞", "❮", "❯", "〝", "〞", "〟", "＂", "＂"
    ]))

    SINGLE_QUOTE_REGEX = re.compile("|".join([
        "‘", "‛", "’", "❛", "❜", "`", "´", "‘", "’"
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
        # text = self._apply_punctuation_and_spacing(text)
    
        return text
    
    def _remove_hieroglyphs_unicode(self, text: str) -> str:
        allowed_categories = {
            'Lu',  # Uppercase letter
            'Ll',  # Lowercase letter
            'Lt',  # Titlecase letter
            'Nd',  # Decimal number
            'Nl',  # Letter number
            'No',  # Other number
            'Pc',  # Connector punctuation
            'Pd',  # Dash punctuation
            'Ps',  # Open punctuation
            'Pe',  # Close punctuation
            'Pi',  # Initial punctuation
            'Pf',  # Final punctuation
            'Po',  # Other punctuation
            'Sm',  # Math symbol
            'Sc',  # Currency symbol
            'Zs',  # Space separator
        }
        
        result = ""
        for char in text:
            if unicodedata.category(char) in allowed_categories:
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
        
        console = Console()
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("{task.percentage:>3.0f}%"),
            OCRSpeedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task: TaskID = progress.add_task("OCR images", total=total_images)
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
                            progress.update(task, advance=1)
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

def find_matching_files(directory: str, hardsub_pattern: str, clean_pattern: str) -> Dict[str, Dict[str, str]]:
    all_files = glob.glob(f"{directory}/*")
    
    episodes: Dict[str, Dict[str, str]] = {}
    
    for file in all_files:
        hardsub_match = re.search(hardsub_pattern, file)
        if hardsub_match:
            episode = hardsub_match.group(1)
            if episode not in episodes:
                episodes[episode] = {'hardsub': file}
            continue
        
        clean_match = re.search(clean_pattern, file)
        if clean_match:
            episode = clean_match.group(1)
            if episode not in episodes:
                episodes[episode] = {'clean': file}
            elif 'clean' not in episodes[episode]:
                episodes[episode]['clean'] = file
    
    return episodes

def process_episode(clean_path: Optional[str], sub_path: Optional[str], images_dir: Optional[str], output_subtitles: str, offset_clean: int, offset_sub: int, do_filter: bool = True) -> None:
    current_directory = Path(Path.cwd())
    if not images_dir:
        images_dir = Path(f'{current_directory}/images')
    else:
        images_dir = Path(images_dir)
    
    output_file = Path(f'{current_directory}/{output_subtitles}.srt')
    counter = 1
    while output_file.exists():
        output_file = Path(f'{current_directory}/{output_subtitles}_{counter}.srt')
        counter += 1
    
    srt_file = open(output_file, 'w', encoding='utf-8')

    if do_filter:
        if images_dir.exists():
            rmtree(images_dir)
        images_dir.mkdir()
        
        filter = Filter(clean_path, offset_clean, sub_path, offset_sub, images_dir)
        filter.filter_videos()

    engine = OCR_Subtitles(images_dir, srt_file)
    engine.ocr()

def batch_process(directory: str, hardsub_pattern: str, clean_pattern: str, offset_clean: int, offset_sub: int) -> None:
    episodes = find_matching_files(directory, hardsub_pattern, clean_pattern)
    
    for episode, files in sorted(episodes.items()):
        if 'clean' in files and 'hardsub' in files:
            print(f"Processing episode {episode}")
            output_subtitles = f"output_subtitles_ep{episode}"
            images_dir = f"{directory}/images_{episode}"
            process_episode(clean_path=files['clean'],
                            sub_path=files['hardsub'],
                            images_dir=images_dir,
                            output_subtitles=output_subtitles,
                            offset_clean=offset_clean,
                            offset_sub=offset_sub,
                            do_filter=True)
        else:
            print(f"Skipping episode {episode} - missing clean or hardsub file")

def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='A tool to OCR subtitles from video files, supporting both single-file and batch processing modes.')
    parser.add_argument('-o', '--output', default="output_subtitles", dest='output_subtitles', metavar='<outputname>',
                        help='Base name for output subtitle files. In batch mode, episode numbers will be appended.')
    parser.add_argument('--filter', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable or disable the filtering and image generation step. Default: enabled')
    
    parser.add_argument('--batch', action='store_true',
                        help='Enable batch processing mode to handle multiple episodes')
    
    parser.add_argument('--directory', type=str, metavar='<dir>', default=".",
                        help='Directory containing video files for batch processing')
    
    parser.add_argument('--clean-offset', default=0, type=int, dest='offset_clean', metavar='<frames>',
                        help='Frame offset for clean video. Default: 0')
    
    parser.add_argument('--hardsub-offset', default=0, type=int, dest='offset_sub', metavar='<frames>',
                        help='Frame offset for hardsub video. Default: 0')
    
    parser.add_argument("clean", nargs='?', metavar='<clean>',
                        help='In single-file mode: path to the clean source video. '
                             'In batch mode: regex pattern for matching clean files, with episode number as group 1.')
    
    parser.add_argument("hardsub", nargs='?', metavar='<hardsub>',
                        help='In single-file mode: path to the hardsub source video. '
                             'In batch mode: regex pattern for matching hardsub files, with episode number as group 1.')
    
    return parser

if is_preview():
    filter = Filter(r"clean.mkv", 0, r"sub.mkv", 0)
    filter.filter_videos()
elif __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    if args.batch:
        if not all([args.directory, args.clean, args.hardsub]):
            parser.error("The --directory, clean (pattern), and sub (pattern) arguments are required when using batch mode")
        batch_process(args.directory, args.hardsub, args.clean, args.offset_clean, args.offset_sub)
    else:
        if args.filter:
            if not args.clean or not args.hardsub:
                parser.error("The 'clean' and 'sub' arguments are required when '--filter' is enabled.")
            
            process_episode(clean_path=args.clean,
                            sub_path=args.hardsub,
                            images_dir=None,
                            output_subtitles=args.output_subtitles,
                            offset_clean=args.offset_clean,
                            offset_sub=args.offset_sub,
                            do_filter=True)
        else:
            process_episode(clean_path=None,
                            sub_path=None,
                            images_dir=None,
                            output_subtitles=args.output_subtitles,
                            offset_clean=args.offset_clean,
                            offset_sub=args.offset_sub,
                            do_filter=False)

    print("Done")

