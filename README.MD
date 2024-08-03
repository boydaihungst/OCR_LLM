## How it work?
Diff a hardsubbed video with video that is not hardsubbed to detect subtitle frames.
Extract these frames as images and OCR them using Google Lens.
Integrate the resulting text into an SRT subtitle file.

## Accuracy
Guess this :)

## Setup

Step 1: Install Python

Step 2: Install VapourSynth

Step 3: `pip install -r requirements.txt`

Step 4: `vsrepo.py install acrop fpng hysteresis lsmas misc tcanny tedgemask`


## Usage
Prepare two source.
- Muse HardSubed from YouTube, should choose 720p AVC format.
- Non HardSubed, should be the same resoluion with HardSubed source. Higher resolution will take a longer time to process.
  
Two sources must be synchronized. If not, adjust offset arguments.

```sh
python ocr.py clean.mp4 sub.mp4
```

For more.
```sh
python ocr.py --help
```

For non-Muse sources, it is necessary to adjust the crop parameters to an subtitles area, also may need to adjust SceneDetect threshold.

## TODO

Rewrite this spaghetti.

## Acknowledgement
- [VapourSynth](https://www.vapoursynth.com/doc/index.html)
- [JET](https://github.com/Jaded-Encoding-Thaumaturgy)
- [image-ocr-google-docs-srt](https://github.com/Abu3safeer/image-ocr-google-docs-srt)
- [LunaTranslator](https://github.com/HIllya51/LunaTranslator/blob/main/LunaTranslator/LunaTranslator/ocrengines/googlelens.py)