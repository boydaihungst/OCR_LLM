import argparse
import concurrent.futures
import glob
import re
import time
import unicodedata
from io import StringIO
from pathlib import Path
from shutil import rmtree
from threading import Lock
from typing import Dict, List, Optional

import json5
import lxml.html
from httpx import Client
from rich.console import Console
from rich.progress import (BarColumn, Progress, ProgressColumn, Task, TaskID,
                           Text, TextColumn, TimeRemainingColumn)


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
    fake_chromium_config = {
            "viewport": (1920, 1080),
            "major_version": "109",
            "version": "109.0.5414.87",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.5414.87 Safari/537.36",
        }
    
    LENS_ENDPOINT = f"https://lens.google.com/v3/upload"
    COOKIES = {"SOCS": "CAESEwgDEgk0ODE3Nzk3MjQaAmVuIAEaBgiA_LyaBg"}
    PARAMS = {
            "ep": "ccm",  # EntryPoint
            "re": "dcsp",  # RenderingEnvironment - DesktopChromeSurfaceProto
            "s": "4",  # SurfaceProtoValue - Surface.CHROMIUM
            "st": str(int(time.time() * 1000)),
            "sideimagesearch": "1",
            "vpw": str(fake_chromium_config["viewport"][0]),
            "vph": str(fake_chromium_config["viewport"][1]),
        }
    HEADERS = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.5",
            "Cache-Control": "max-age=0",
            "Origin": "https://lens.google.com",
            "Referer": "https://lens.google.com/",
            "Sec-Ch-Ua": f'"Not A(Brand";v="99", "Google Chrome";v="{fake_chromium_config["major_version"]}", "Chromium";v="{fake_chromium_config["major_version"]}"',
            "Sec-Ch-Ua-Arch": '"x86"',
            "Sec-Ch-Ua-Bitness": '"64"',
            "Sec-Ch-Ua-Full-Version": f'"{fake_chromium_config["version"]}"',
            "Sec-Ch-Ua-Full-Version-List": f'"Not A(Brand";v="99.0.0.0", "Google Chrome";v="{fake_chromium_config["major_version"]}", "Chromium";v="{fake_chromium_config["major_version"]}"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Model": '""',
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Ch-Ua-Platform-Version": '"15.0.0"',
            "Sec-Ch-Ua-Wow64": "?0",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            # "User-Agent": fake_chromium_config["user_agent"],
            "X-Client-Data": "CIW2yQEIorbJAQipncoBCIH+ygEIkqHLAQiKo8sBCPWYzQEIhaDNAQji0M4BCLPTzgEI19TOAQjy1c4BCJLYzgEIwNjOAQjM2M4BGM7VzgE=",
        }

    DOUBLE_QUOTE_REGEX = re.compile("|".join([
        "«", "‹", "»", "›", "„", "“", "‟", "”", "❝", "❞", "❮", "❯", "〝", "〞", "〟", "＂", "＂"
    ]))

    SINGLE_QUOTE_REGEX = re.compile("|".join([
        "‘", "‛", "’", "❛", "❜", "`", "´", "‘", "’"
    ]))

    def __init__(self):
        self.client = Client()

    def __del__(self):
        self.client.close()

    def lens_ocr(self, img_path: str) -> str:
        files = {
            "encoded_image": ("screenshot.png", (open(img_path, 'rb')), "image/png")
        }
        
        max_retries = 3
        last_exception = None
        for attempt in range(max_retries):
            try:
                res = self.client.post(
                    self.LENS_ENDPOINT,
                    files=files,
                    headers=self.HEADERS,
                    params=self.PARAMS,
                    cookies=self.COOKIES,
                    timeout=40
                )
                
                if res.status_code == 200:
                    break
                    
                raise Exception(f"Request failed with status code: {res.status_code}")
                
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt + 1} failed. Retrying...")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to upload image after {max_retries} attempts. Last error: {str(last_exception)}")
                continue

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
    THREADS = 10
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
                        console.print(f"[red]{image} generated an exception: {exc}[/red]")
                    else:
                        with self.scan_lock:
                            self.completed_scans += 1
                            progress.update(task, advance=1)
        self._write_srt()

    def _process_image(self, image: Path, line: int):
        img_filename = str(image.absolute())
        img_name = str(image.name)

        try:
            text = self.lens.lens_ocr(img_filename)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            text = ""

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
            text
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
        file_name = Path(file).name
        hardsub_match = re.search(hardsub_pattern, file_name)
        if hardsub_match:
            episode = hardsub_match.group(1)
            if episode not in episodes:
                episodes[episode] = {'hardsub': file}
            elif 'hardsub' not in episodes[episode]:
                episodes[episode]['hardsub'] = file
        
        clean_match = re.search(clean_pattern, file_name)
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
        from filter import Filter
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

if __name__ == "__main__":
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

