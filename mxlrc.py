from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
import argparse
import sys
from pathlib import Path
from dataclasses import asdict, dataclass, field
import json
import logging
import math
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from rich import print
from rich.progress import track
from tinytag import TinyTag


class Musixmatch:
    base_url = 'https://apic-desktop.musixmatch.com/ws/1.1/macro.subtitles.get'
    queries = {
        'format': 'json',
        'namespace': 'lyrics_richsynched',
        'subtitle_format': 'mxm',
        'app_id': 'web-desktop-app-v1.0',
    }
    headers = {
        'authority': 'apic-desktop.musixmatch.com',
        'cookie': 'x-mxm-token-guid=',
    }

    def __init__(self, token: str):
        self.set_token(token)

    def set_token(self, token: str):
        self.token = token

    def find_lyrics(self, song: Song):
        durr = song.duration / 1000 if song.duration else ''
        query = dict(self.queries, **{
            'q_album': song.album,
            'q_artist': song.artist,
            'q_artists': song.artist,
            'q_track': song.title,
            'track_spotify_id': song.uri,
            'q_duration': durr,
            'f_subtitle_length': math.floor(durr) if durr else '',
            'usertoken': self.token,
        })
        url_encoded = urllib.parse.urlencode(query, quote_via=urllib.parse.quote)
        url = self.base_url + '?' + url_encoded

        req = urllib.request.Request(url, headers=self.headers)
        try:
            response = urllib.request.urlopen(req).read()
        except (urllib.error.HTTPError, urllib.error.URLError, ConnectionResetError) as e:
            logging.error(repr(e))
            return

        r = json.loads(response.decode())
        if r['message']['header']['status_code'] != 200 and r['message']['header'].get('hint') == 'renew':
            logging.error('Invalid token')
            return
        body = r['message']['body']['macro_calls']
        if body['matcher.track.get']['message']['header']['status_code'] != 200:
            if body['matcher.track.get']['message']['header']['status_code'] == 404:
                logging.info('Song not found.')
            elif body['matcher.track.get']['message']['header']['status_code'] == 401:
                logging.warning('Timed out. Change the token or wait a few minutes before trying again.')
            else:
                header = body['matcher.track.get']['message']['header']
                logging.error(f'Requested error: {header}')
            return
        elif isinstance(body['track.lyrics.get']['message'].get('body'), dict):
            if body['track.lyrics.get']['message']['body']['lyrics']['restricted']:
                logging.info('Restricted lyrics.')
                return
        return body

    @staticmethod
    def get_unsynced(song: Song, body):
        if song.is_instrumental:
            lines = [LyricLine('♪ Instrumental ♪')]
        elif song.has_unsynced:
            lyrics_body = body['track.lyrics.get']['message'].get('body')
            if lyrics_body is None:
                return False
            lyrics = lyrics_body['lyrics']['lyrics_body']
            if lyrics:
                lines = [LyricLine(line) for line in list(filter(None, lyrics.split('\n')))]
            else:
                lines = [LyricLine('')]
        else:
            lines = []
        song.lyrics = lines
        return True

    @staticmethod
    def get_synced(song: Song, body):
        if song.is_instrumental:
            lines = [LyricLine('♪ Instrumental ♪')]
        elif song.has_synced:
            subtitle_body = body['track.subtitles.get']['message'].get('body')
            if subtitle_body is None:
                return False
            subtitle = subtitle_body['subtitle_list'][0]['subtitle']
            if subtitle:
                lines = [
                    LyricLine(
                        text=str(line['text'] or '♪'),
                        minutes=int(line['time']['minutes']),
                        seconds=int(line['time']['seconds']),
                        hundredths=int(line['time']['hundredths']),
                    )
                    for line in json.loads(subtitle['subtitle_body'])
                ]
            else:
                lines = [LyricLine('')]
        else:
            lines = []
        song.subtitles = lines
        return True

    @staticmethod
    def gen_lrc(song: Song):
        lyrics = song.subtitles
        if lyrics is None:
            logging.warning('Synced lyrics not found, using unsynced lyrics...')
            lyrics = song.lyrics
            if lyrics is None:
                logging.warning('Unsynced lyrics not found')
                return False
        logging.info('Formatting lyrics')
        tags = [
            '[by:fashni]',
            f'[ar:{song.artist}]',
            f'[ti:{song.title}]',
        ]
        if song.album:
            tags.append(f'[al:{song.album}]')
        if song.duration:
            tags.append(f'[length:{int((song.duration/1000)//60):02d}:{int((song.duration/1000)%60):02d}]')

        out_path = song.outfile
        with out_path.open('w', encoding='utf-8') as f:
            f.write('\n'.join(str(s) for s in (tags + lyrics)))
        print(f'Lyrics saved: {out_path}')
        return True


@dataclass
class LyricLine:
    text: str
    minutes: int = 0
    seconds: int = 0
    hundredths: int = 0

    def __str__(self) -> str:
        return f'[{self.minutes:02d}:{self.seconds:02d}.{self.hundredths:02d}]{self.text}'


@dataclass
class Song(object):
    artist: str
    title: str
    filepath: Path

    album: str = ''
    uri: str = ''
    duration: int = 0
    has_synced: bool = False
    has_unsynced: bool = False
    is_instrumental: bool = False
    lyrics: List[LyricLine] = field(default_factory=list)
    subtitles: List[LyricLine] = field(default_factory=list)
    coverart_url: Optional[str] = None

    @staticmethod
    def slugify(value: str):
        'https://github.com/django/django/blob/main/django/utils/text.py'
        value = unicodedata.normalize('NFKC', value)
        value = re.sub(r"[^\w\s()'-]", '', value)
        return re.sub(r'[-]+', '-', value).strip('-_')

    @property
    def outfile(self):
        return self.filepath.with_suffix('.lrc')

    @classmethod
    def create_file_name(cls, out_dir: Path, artist: str, title: str):
        return out_dir / cls.slugify(artist + ' - ' + title)

    def __str__(self) -> str:
        return f'{self.artist},{self.title}'

    @property
    def info(self):
        return self.__dict__

    def update_info(self, body: Dict[str, Any]):
        meta = body['matcher.track.get']['message']['body']
        if not meta:
            return
        coverart_sizes = ['100x100', '350x350', '500x500', '800x800']
        coverart_urls = list(filter(None, [meta['track'][f'album_coverart_{size}'] for size in coverart_sizes]))
        self.coverart_url = coverart_urls[-1] if coverart_urls else None
        self.title = meta['track']['track_name']
        self.artist = meta['track']['artist_name']
        self.album = meta['track']['album_name']
        self.duration = meta['track']['track_length'] * 1000
        self.has_synced = meta['track']['has_subtitles']
        self.has_unsynced = meta['track']['has_lyrics']  # or meta['track']['has_lyrics_crowd']
        self.is_instrumental = meta['track']['instrumental']

    @classmethod
    def parse_strs(cls, songs: List[str], cfg: Config):
        to_process: List[Song] = []
        for song in songs:
            maybe_path = Path(song).expanduser()
            if maybe_path.is_dir():
                to_process.extend(cls.get_song_dir(maybe_path, cfg))
            elif maybe_path.is_file():
                to_process.extend(cls.get_song_txt(maybe_path, cfg))
            else:
                to_process.extend(cls.get_song_multi([song], cfg))
        return to_process

    @classmethod
    def get_song_dir(cls, directory: Path, cfg: Config, current_depth=0):
        logging.info(f'Scanning directory: {directory}')
        logging.debug(f'Max depth: {cfg.depth} - Current depth: {current_depth}')
        songs = []
        for f in sorted([f for f in directory.iterdir()], key=lambda x: x.is_dir() if cfg.bfs else x.is_file()):
            if f.is_dir():
                if current_depth < cfg.depth:
                    songs.extend(cls.get_song_dir(f, cfg, current_depth + 1))
                continue
            if f.suffix not in cfg.ext:
                continue
            if not TinyTag.is_supported(str(f.absolute())):
                logging.debug(f'TinyTag not supporting {f.suffix}. Skipping "{f}". File not supported.')
                continue
            song_file = TinyTag.get(str(f.absolute()))
            if not isinstance(song_file, TinyTag) or not (song_file.artist and song_file.title):
                logging.warning(f'Skipping "{f}". Cannot parse song info')
                continue
            if f.with_suffix('.lrc').exists() and not cfg.update:
                logging.info(f'Skipping "{f}". Lyrics file exists')
                continue
            logging.info(f'Adding "{f}"')
            songs.append(Song(song_file.artist, song_file.title, f))
        return songs

    @classmethod
    def get_song_txt(cls, txt: Path, cfg: Config):
        with txt.open('r') as f:
            song_list = f.read().split('\n')
        return cls.get_song_multi(song_list, cfg)

    @classmethod
    def get_song_multi(cls, song_list: List[str], cfg: Config):
        songs = []
        for song in song_list:
            artist, title = cls.validate_input(song)
            if artist is None or title is None:
                continue
            songs.append(Song(artist, title, Song.create_file_name(cfg.out, artist, title)))
        return songs

    @staticmethod
    def validate_input(inp: str):
        parsed = [s.strip() for s in inp.split(',')]
        if len(parsed) > 1:
            return parsed[:2]
        if len(parsed) == 1:
            return parsed[0], parsed[0]
        return None, None


@dataclass
class Config:
    outdir: str = 'lyrics'
    ext: Set[str] = field(default_factory=set)
    sleep: int = 30
    depth: int = 100
    update: bool = False
    bfs: bool = False
    quiet: bool = False
    token: Optional[str] = None
    debug: bool = False

    @property
    def out(self):
        o = Path(self.outdir)
        o.mkdir(exist_ok=True, parents=True)
        return o

    @classmethod
    def parse_args(cls, _args: List[str]):
        parser = argparse.ArgumentParser(
            description=f'Fetch synced lyrics (*.lrc) from Musixmatch: {Musixmatch.base_url}')
        parser.add_argument('-s', '--song', dest='song', nargs='+', required=True,
                            help='song information in the format [(artist,)title],'
                            + 'a text file containing list of songs,'
                            + 'or a directory containing the song files')
        parser.add_argument('-e', '--ext', nargs='*',
                            help='List of extensions of files to process in dir mode. Default: `.mp3`')
        parser.add_argument('-o', '--out', dest='outdir', default='lyrics', action='store', type=str,
                            help='output directory to save the .lrc file(s), default: lyrics')
        parser.add_argument('-t', '--sleep', dest='sleep', default=30, action='store', type=int,
                            help='sleep time (seconds) in between request, default: 30')
        parser.add_argument('-d', '--depth', dest='depth', default=100, type=int,
                            help='(directory mode) maximum recursion depth, default: 100')
        parser.add_argument('-u', '--update', dest='update', action='store_true',
                            help='(directory mode) rewrite existing .lrc files inside the output directory')
        parser.add_argument('--bfs', dest='bfs', action='store_true',
                            help='(directory mode) use breadth first search for scanning directory')
        parser.add_argument('-q', '--quiet', dest='quiet', help='suppress logging output', action='store_true')
        parser.add_argument('--token', dest='token', help='musixmatch token', type=str)
        parser.add_argument('--debug', dest='debug', help=argparse.SUPPRESS, action='store_true')
        args = parser.parse_args(_args)
        songs: List[str] = args.song
        self = cls()
        for k, v in asdict(self).items():
            arg = getattr(args, k)
            if k == 'ext':
                self.ext.update(['.mp3'] + (arg or []))
            elif isinstance(v, bool):
                setattr(self, k, bool(arg))
            elif isinstance(v, str):
                setattr(self, k, str(arg))
            elif isinstance(v, int):
                setattr(self, k, int(arg))
            elif isinstance(v, list):
                setattr(self, k, list(arg))
            elif arg is None:
                pass
            else:
                raise RuntimeError(f'Unknown type on {k}: {arg} (default: {v})')
        return self, songs


def get_lrc(mx: Musixmatch, song: Song):
    logging.info(f'Searching song: {song}')
    body = mx.find_lyrics(song)
    if body is None:
        return False
    song.update_info(body)
    logging.info(f'Song found: {song}')
    logging.info(f'Searching lyrics: {song}')
    mx.get_synced(song, body)
    mx.get_unsynced(song, body)
    status = mx.gen_lrc(song)
    return status


def main(songs: List[Song], cfg: Config):
    run_time = time.strftime('%Y%m%d_%H%M%S')
    MX_TOKEN = cfg.token if cfg.token else '200501593b603a3fdc5c9b4a696389f6589dd988e5a1cf02dfdce1'
    mx = Musixmatch(MX_TOKEN)

    failed: List[Song] = []
    for idx, song in enumerate(songs):
        try:
            if idx != 0:
                for _ in track(range(cfg.sleep), description='Sleeping...', total=cfg.sleep):
                    time.sleep(1)
            success = get_lrc(mx, song)
            if not success:
                failed.append(song)
        except KeyboardInterrupt as e:
            logging.warning(repr(e))
            failed += songs[idx:]
            break
    logging.info(f'Succesfully fetch {len(songs)-len(failed)} out of {len(songs)} lyrics.')
    if failed:
        logging.warning(f'Failed to fetch {len(failed)} lyrics.')
        failed_path = Path().cwd() / f'{run_time}_failed.txt'
        logging.warning(f'Saving list of failed items in {failed_path}. You can try again using this file as the input')
        with failed_path.open('w') as f:
            f.write('\n'.join(str(s) for s in failed))


def rename_logging_level_names():
    for level in list(logging._levelToName):
        if level == logging.NOTSET:
            name = '[-]'
        elif level == logging.DEBUG:
            name = '[/]'
        elif level == logging.INFO:
            name = '[+]'
        elif level == logging.WARNING:
            name = '[o]'
        elif level == logging.ERROR:
            name = '[X]'
        else:
            name = logging.getLevelName(level).lower()
        logging.addLevelName(level, name)


def run_cli():
    rename_logging_level_names()
    cfg, song_strs = Config.parse_args(sys.argv[1:])
    logging_level = logging.WARNING if cfg.quiet else logging.INFO
    logging.basicConfig(format='%(levelname)s %(message)s', level=logging.DEBUG if cfg.debug else logging_level)
    logging.info(f'{cfg}')
    songs = Song.parse_strs(song_strs, cfg)
    if cfg is not None:
        logging.info(f'{len(songs)} lyrics to fetch')
        main(songs, cfg)


if __name__ == '__main__':
    run_cli()
