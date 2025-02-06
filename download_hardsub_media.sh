#!/usr/bin/env bash

# Download hardsub media from https://phim.nguonc.com
# URL should be: https://embed12.streamc.xyz/embed.php?hash=b3f41e4623c49554396f352e4049b4e4
# Usage: ./download_hardsub_media.sh "https://embed12.streamc.xyz/embed.php?hash=b3f41e4623c49554396f352e4049b4e4"
set -x

m3u8_file="a.m3u8"

curl "$(echo "$1" | sed 's|embed\.php|get\.php|g')" --compressed -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:135.0) Gecko/20100101 Firefox/135.0' -H 'Accept: */*' -H 'Accept-Language: vi,en-US;q=0.7,en;q=0.3' -H 'Accept-Encoding: gzip, deflate, br, zstd' -H 'DNT: 1' -H 'Sec-GPC: 1' -H 'Connection: keep-alive' -H "Referer: $1" -H 'Sec-Fetch-Dest: empty' -H 'Sec-Fetch-Mode: cors' -H 'Sec-Fetch-Site: same-origin' -H 'Pragma: no-cache' -H 'Cache-Control: no-cache' -H 'TE: trailers' -o "$m3u8_file"

i=0
rm "./hls_downloaded" -rf
mkdir "./hls_downloaded" -p
while read -r line; do
  if [[ $line == https://* ]]; then
    curl "$line" -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:135.0) Gecko/20100101 Firefox/135.0' -H 'Accept: */*' -H 'Accept-Language: vi,en-US;q=0.7,en;q=0.3' -H 'Accept-Encoding: gzip, deflate, br, zstd' -H 'Origin: https://embed1.streamc.xyz' -H 'DNT: 1' -H 'Sec-GPC: 1' -H 'Connection: keep-alive' -H 'Referer: https://embed1.streamc.xyz/' -H 'Sec-Fetch-Dest: empty' -H 'Sec-Fetch-Mode: cors' -H 'Sec-Fetch-Site: cross-site' -H 'Pragma: no-cache' -H 'Cache-Control: no-cache' -H 'TE: trailers' --output "./hls_downloaded/segments_$i.ts"
    printf "file 'segments_%s.ts'\n" "$i" >>"./hls_downloaded/temp.txt"
    i=$((i + 1))
  fi
done <"$m3u8_file"
ffmpeg -y -nostdin -loglevel error -f concat -safe 0 -i "./hls_downloaded/temp.txt" -c copy "./sub.ts"
