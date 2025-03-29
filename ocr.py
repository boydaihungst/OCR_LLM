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

from httpx import Client
from rich.console import Console
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskID, TextColumn, TimeRemainingColumn
from rich.text import Text

from lens import (AppliedFilter, LensOverlayFilterType, LensOverlayRoutingInfo, LensOverlayServerRequest,
                  LensOverlayServerResponse, Platform, Surface,)
from utils import Utils


### OCR PART
class OCRSpeedColumn(ProgressColumn):
    """Progress rendering."""

    def render(self, task: Task) -> Text:
        """Render bar."""

        return Text(f"{task.speed or 0:.02f} images/s")


class AssSubtitle:
    def __init__(self, start_time, end_time, text_content, is_top=False):
        self.start_time = start_time
        self.end_time = end_time
        self.text_content = text_content
        self.style_name = "Top" if is_top else "Default"
        self.is_top = is_top

    def convert_timestamp(self, s: str):
        h, m, rest = s.split(":")
        s, ms = rest.split(",")
        return f"{h}:{m}:{s}.{ms}"

    def __str__(self):
        return f"Dialogue: 0,{self.convert_timestamp(self.start_time)},{self.convert_timestamp(self.end_time)},{self.style_name},,0,0,0,,{self.text_content}\n"

class GoogleLens:
    LENS_ENDPOINT: str = "https://lensfrontend-pa.googleapis.com/v1/crupload"

    HEADERS: dict[str, str]  = {
        'Host': 'lensfrontend-pa.googleapis.com',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-protobuf',
        'X-Goog-Api-Key': 'AIzaSyDr2UxVnv_U85AbhhY8XSHSIavUW0DC-sY',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-Mode': 'no-cors',
        'Sec-Fetch-Dest': 'empty',
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        'Accept-Encoding': 'gzip, deflate, br, zstd',
    }

    def __init__(self):
        self.client: Client = Client()

    def __del__(self):
        self.client.close()
    
    def __call__(self, img_path: str):
        
        request = LensOverlayServerRequest()

        request.objects_request.request_context.request_id.uuid = random.randint(0, 2**64 - 1)
        request.objects_request.request_context.request_id.sequence_id = 0
        request.objects_request.request_context.request_id.image_sequence_id = 0
        request.objects_request.request_context.request_id.analytics_id = random.randbytes(n=16)
        request.objects_request.request_context.request_id.routing_info = LensOverlayRoutingInfo()

        request.objects_request.request_context.client_context.platform = Platform.WEB
        request.objects_request.request_context.client_context.surface = Surface.CHROMIUM

        # request.objects_request.request_context.client_context.locale_context.language = 'vi'
        # request.objects_request.request_context.client_context.locale_context.region = 'Asia/Ho_Chi_Minh'
        request.objects_request.request_context.client_context.locale_context.time_zone = '' # not set by chromium

        request.objects_request.request_context.client_context.app_id = '' # not set by chromium

        filter = AppliedFilter()
        filter.filter_type = LensOverlayFilterType.AUTO_FILTER
        request.objects_request.request_context.client_context.client_filters.filter.append(filter)

        image_data = Utils.get_image_raw_bytes_and_dims(img_path)
        if image_data is not None:
            raw_bytes, width, height = image_data

            request.objects_request.image_data.payload.image_bytes = raw_bytes
            request.objects_request.image_data.image_metadata.width = width
            request.objects_request.image_data.image_metadata.height = height
        else:
            print(f"Error: Could not process image file '{img_path}'. Cannot populate image data in request.")

        payload = request.SerializeToString()

        res = None
        max_retries = 3
        last_exception = None
        for attempt in range(max_retries):
            try:
                res = self.client.post(
                    self.LENS_ENDPOINT,
                    content=payload,
                    headers=self.HEADERS,
                    timeout=40,
                )

                if res.status_code == 200:
                    break
                
                raise Exception(f"Request failed with status code: {res.status_code}")

            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt + 1} failed. Retrying...")
                if attempt == max_retries - 1:
                    raise Exception(
                        f"Failed to upload image after {max_retries} attempts. Last error: {str(last_exception)}"
                    )
                continue

        if res != None:
            response_proto = LensOverlayServerResponse().FromString(res.content)
            response_dict: dict[str, Any] = response_proto.to_dict()

            result: str = ''
            text = response_dict['objectsResponse']['text']['textLayout']['paragraphs'] 
            for paragraph in text:
                for line in paragraph['lines']:
                    for word in line['words']:
                        result += word['plainText'] + word['textSeparator']
                result += '\n'

            return result
        


class OCR_Subtitles:
    THREADS = 10
    IMAGE_EXTENSIONS = ("*.jpeg", "*.jpg", "*.png", "*.bmp", "*.gif")

    def __init__(self, images_dir: Path, ass_file: Path) -> None:
        self.images: List[str] = []
        self.ass_dict: Dict[int, AssSubtitle] = {}
        self.scan_lock: Lock = Lock()
        self.lens: Lens = Lens()
        self.images_dir: Path = images_dir
        self.ass_file: Path = ass_file
        self.completed_scans: int = 0

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
            console=console,
        ) as progress:
            task: TaskID = progress.add_task("OCR images", total=total_images)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.THREADS) as executor:
                future_to_image = {
                    executor.submit(self._process_image, Path(image), index + 1): image
                    for index, image in enumerate(self.images)
                }
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
        self._write_ass()

    def _process_image(self, image: Path, line: int):
        img_filename = str(image.absolute())
        img_name = str(image.name)

        try:
            text = self.lens.lens_ocr(img_filename)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            text = ""
            is_top = False

        try:
            # Case filename = top/bot_time
            if img_name.split("_")[0] == "top" or img_name.split("_")[0] == "bot":
                is_top = img_name.split("_")[0] == "top"
                start_hour = img_name.split("_")[1][:2]
                start_min = img_name.split("_")[2][:2]
                start_sec = img_name.split("_")[3][:2]
                start_micro = img_name.split("_")[4][:3]

                end_hour = img_name.split("__")[1].split("_")[0][:2]
                end_min = img_name.split("__")[1].split("_")[1][:2]
                end_sec = img_name.split("__")[1].split("_")[2][:2]
                end_micro = img_name.split("__")[1].split("_")[3][:3]
            # Case filename = time. Backward compatibility
            else:
                is_top = False
                start_hour = img_name.split("_")[0][:2]
                start_min = img_name.split("_")[1][:2]
                start_sec = img_name.split("_")[2][:2]
                start_micro = img_name.split("_")[3][:3]

                end_hour = img_name.split("__")[1].split("_")[0][:2]
                end_min = img_name.split("__")[1].split("_")[1][:2]
                end_sec = img_name.split("__")[1].split("_")[2][:2]
                end_micro = img_name.split("__")[1].split("_")[3][:3]
        except IndexError:
            print(
                f"Error processing {img_name}: Filename format is incorrect. Please ensure the correct format is used."
            )
            return

        start_time = f"{start_hour}:{start_min}:{start_sec},{start_micro}"
        end_time = f"{end_hour}:{end_min}:{end_sec},{end_micro}"

        subtitle = AssSubtitle(start_time, end_time, text, is_top)
        self.ass_dict[line] = subtitle

    def _write_ass(self):
        cleaned_ass_bot = []
        cleaned_ass_top = []
        previous_subtitle_bot = None
        previous_subtitle_top = None
        for _, subtitle in sorted(self.ass_dict.items()):
            previous_subtitle = previous_subtitle_top if subtitle.is_top else previous_subtitle_bot
            cleaned_ass = cleaned_ass_top if subtitle.is_top else cleaned_ass_bot
            if not subtitle.text_content or subtitle.text_content.isspace():
                continue
            if not previous_subtitle:
                cleaned_ass.append(subtitle)
                if subtitle.is_top:
                    previous_subtitle_top = subtitle
                else:
                    previous_subtitle_bot = subtitle
                continue
            if previous_subtitle.text_content.lower() == subtitle.text_content.lower():
                merged_subtitle = AssSubtitle(
                    start_time=previous_subtitle.start_time,
                    end_time=subtitle.end_time,
                    text_content=previous_subtitle.text_content,
                    is_top=subtitle.is_top,
                )
                cleaned_ass.pop()
                cleaned_ass.append(merged_subtitle)
            else:
                cleaned_ass.append(subtitle)

            if subtitle.is_top:
                previous_subtitle_top = cleaned_ass[-1]
            else:
                previous_subtitle_bot = cleaned_ass[-1]
        self.ass_file.write(
            f"""[Script Info]
ScriptType: v4.00+
PlayDepth: 0
ScaledBorderAndShadow: Yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,60,&H00FFFFFF,&H00000000,&H4D000000,&H81000000,-1,0,0,0,100,100,0,0,1,3,0,2,60,60,40,1
Style: Top,Arial,60,&H00FFFFFF,&H00000000,&H4D000000,&H81000000,-1,0,0,0,100,100,0,0,1,3,0,8,60,60,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        )
        for subtitle in cleaned_ass_bot + cleaned_ass_top:
            self.ass_file.write(str(subtitle))
        self.ass_file.close()


def find_matching_files(directory: str, hardsub_pattern: str, clean_pattern: str) -> Dict[str, Dict[str, str]]:
    all_files = glob.glob(f"{directory}/*")

    episodes: Dict[str, Dict[str, str]] = {}

    for file in all_files:
        file_name = Path(file).name
        hardsub_match = re.search(hardsub_pattern, file_name)
        if hardsub_match:
            episode = hardsub_match.group(1)
            if episode not in episodes:
                episodes[episode] = {"hardsub": file}
            elif "hardsub" not in episodes[episode]:
                episodes[episode]["hardsub"] = file

        clean_match = re.search(clean_pattern, file_name)
        if clean_match:
            episode = clean_match.group(1)
            if episode not in episodes:
                episodes[episode] = {"clean": file}
            elif "clean" not in episodes[episode]:
                episodes[episode]["clean"] = file

    return episodes


def process_episode(
    clean_path: Optional[str],
    sub_path: Optional[str],
    images_dir: Optional[str],
    output_subtitles: str,
    offset_clean: int,
    offset_sub: int,
    do_filter: bool = True,
) -> None:
    current_directory = Path(Path.cwd())
    if not images_dir:
        images_dir = Path(f"{current_directory}/images")
    else:
        images_dir = Path(images_dir)

    output_file = Path(f"{current_directory}/{output_subtitles}.ass")
    counter = 1
    while output_file.exists():
        output_file = Path(f"{current_directory}/{output_subtitles}_{counter}.ass")
        counter += 1

    ass_file = open(output_file, "w", encoding="utf-8")

    if do_filter:
        from filter import Filter

        if images_dir.exists():
            rmtree(images_dir)
        images_dir.mkdir()

        filter = Filter(clean_path, offset_clean, sub_path, offset_sub, images_dir)
        filter.filter_videos()

    engine = OCR_Subtitles(images_dir, ass_file)
    engine.ocr()


def batch_process(directory: str, hardsub_pattern: str, clean_pattern: str, offset_clean: int, offset_sub: int) -> None:
    episodes = find_matching_files(directory, hardsub_pattern, clean_pattern)

    for episode, files in sorted(episodes.items()):
        if "clean" in files and "hardsub" in files:
            print(f"Processing episode {episode}")
            output_subtitles = f"output_subtitles_ep{episode}"
            images_dir = f"{directory}/images_{episode}"
            process_episode(
                clean_path=files["clean"],
                sub_path=files["hardsub"],
                images_dir=images_dir,
                output_subtitles=output_subtitles,
                offset_clean=offset_clean,
                offset_sub=offset_sub,
                do_filter=True,
            )
        else:
            print(f"Skipping episode {episode} - missing clean or hardsub file")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A tool to OCR subtitles from video files, supporting both single-file and batch processing modes."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output_subtitles",
        dest="output_subtitles",
        metavar="<outputname>",
        help="Base name for output subtitle files. In batch mode, episode numbers will be appended.",
    )
    parser.add_argument(
        "--filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the filtering and image generation step. Default: enabled",
    )

    parser.add_argument("--batch", action="store_true", help="Enable batch processing mode to handle multiple episodes")

    parser.add_argument(
        "--directory",
        type=str,
        metavar="<dir>",
        default=".",
        help="Directory containing video files for batch processing",
    )

    parser.add_argument(
        "--clean-offset",
        default=0,
        type=int,
        dest="offset_clean",
        metavar="<frames>",
        help="Frame offset for clean video. Default: 0",
    )

    parser.add_argument(
        "--hardsub-offset",
        default=0,
        type=int,
        dest="offset_sub",
        metavar="<frames>",
        help="Frame offset for hardsub video. Default: 0",
    )

    parser.add_argument(
        "clean",
        nargs="?",
        metavar="<clean>",
        help="In single-file mode: path to the clean source video. "
        "In batch mode: regex pattern for matching clean files, with episode number as group 1.",
    )

    parser.add_argument(
        "hardsub",
        nargs="?",
        metavar="<hardsub>",
        help="In single-file mode: path to the hardsub source video. "
        "In batch mode: regex pattern for matching hardsub files, with episode number as group 1.",
    )

    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    if args.batch:
        if not all([args.directory, args.clean, args.hardsub]):
            parser.error(
                "The --directory, clean (pattern), and sub (pattern) arguments are required when using batch mode"
            )
        batch_process(args.directory, args.hardsub, args.clean, args.offset_clean, args.offset_sub)
    else:
        if args.filter:
            if not args.clean or not args.hardsub:
                parser.error("The 'clean' and 'sub' arguments are required when '--filter' is enabled.")

            process_episode(
                clean_path=args.clean,
                sub_path=args.hardsub,
                images_dir=None,
                output_subtitles=args.output_subtitles,
                offset_clean=args.offset_clean,
                offset_sub=args.offset_sub,
                do_filter=True,
            )
        else:
            process_episode(
                clean_path=None,
                sub_path=None,
                images_dir=None,
                output_subtitles=args.output_subtitles,
                offset_clean=args.offset_clean,
                offset_sub=args.offset_sub,
                do_filter=False,
            )

    print("Done")
