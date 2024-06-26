import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, make_dirs


@click.command()
@click.argument("source", type=click.Choice(["file", "dir"], case_sensitive=False))
@click.argument("input_dir", type=click.Path(exists=False, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=False, file_okay=False))
@click.option("--wav-source", help="Where are the audio files saved, file list or folder", type=str)
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite existing files")
@click.option(
    "--peak",
    help="Peak normalize audio to -1 dB",
    default=-1.0,
    show_default=True,
    type=float,
)
@click.option(
    "--loudness",
    help="Loudness normalize audio to -23 dB LUFS",
    default=-23.0,
    show_default=True,
    type=float,
)
@click.option(
    "--block-size",
    help="Block size for loudness measurement, unit is second",
    default=0.400,
    show_default=True,
    type=float,
)
@click.option(
    "--num-workers",
    help="Number of workers to use for processing, defaults to number of CPU cores",
    default=os.cpu_count(),
    show_default=True,
    type=int,
)
def loudness_norm(
    source: str,
    wav_source: str,
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    peak: float,
    loudness: float,
    block_size: float,
    num_workers: int,
):
    """Perform loudness normalization (ITU-R BS.1770-4) on audio files."""

    from fish_audio_preprocess.utils.loudness_norm import loudness_norm_file

    # input_dir, output_dir = Path(input_dir), Path(output_dir)

    # if input_dir == output_dir and clean:
    #     logger.error("You are trying to clean the input directory, aborting")
    #     return

    make_dirs(output_dir)

    logger.info(f"normalizing audio files in wav dir '{input_dir}'")
    if source == "file":
        # 从文件中读取音频文件列表
        files = [
            Path(line.strip().split("|")[0]) for line in open(wav_source, "r", encoding="UTF8").readlines()
        ]
    elif source == "dir":
        # 扫描出所有的音频文件
        files = list_files(wav_source, recursive=recursive)
        files = [str(file) for file in files if file.suffix in AUDIO_EXTENSIONS]

    logger.info(f"Found {len(files)} files, normalizing loudness")

    skipped = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = []

        for file in tqdm(files, desc="Preparing tasks"):
            # Get relative path to input_dir
            relative_path = file.relative_to(input_dir)
            new_file = output_dir / relative_path

            if new_file.parent.exists() is False:
                new_file.parent.mkdir(parents=True)

            if new_file.exists() and not overwrite:
                skipped += 1
                continue

            tasks.append(executor.submit(loudness_norm_file, file, new_file, peak, loudness, block_size))

        for i in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            assert i.exception() is None, i.exception()

    logger.info("Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    loudness_norm()
