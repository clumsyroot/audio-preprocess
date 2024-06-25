import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import click
import torch
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, split_list, split_list_m
from fish_audio_preprocess.utils.transcribe import ASRModelType, batch_transcribe


def replace_lastest(string, old, new):
    return string[::-1].replace(old[::-1], new[::-1], 1)[::-1]


@click.command()
@click.argument("source", type=click.Choice(["file", "dir"], case_sensitive=False))
# @click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--wav_source",
    help="Where are the audio files saved, file list or folder",
    type=str,
)
@click.option(
    "--chunk-size",
    help="chunk size, defaults to 200000",
    default=200000,
    show_default=True,
    type=int,
)
@click.option(
    "--num-workers",
    help="Number of workers to use for processing, defaults to 2",
    default=2,
    show_default=True,
    type=int,
)
@click.option(
    "--lang",
    help="language",
    default="zh",
    show_default=True,
    type=str,
)
@click.option(
    "--model-size",
    # whisper 默认 medium, funasr 默认 paraformer-zh
    help="asr model size(default medium for whisper, paraformer-zh for funasr)",
    default="medium",
    show_default=True,
    type=str,
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Search recursively",
)
@click.option(
    "--model-type",
    help="ASR model type (funasr or whisper)",
    default="whisper",
    show_default=True,
)
def transcribe(
    source: str,
    wav_source: str,
    num_workers: int,
    chunk_size: int,
    lang: str,
    model_size: str,
    recursive: bool,
    model_type: ASRModelType,
):
    """
    Transcribe audio files in a directory.
    """
    ctx = click.get_current_context()
    provided_options = {
        key: value
        for key, value in ctx.params.items()
        if ctx.get_parameter_source(key) == click.core.ParameterSource.COMMANDLINE
    }

    # 如果是 funasr 且没有提供 model_size, 则默认为 paraformer-zh
    if model_type == "funasr" and "model_size" not in provided_options:
        logger.info("Using paraformer-zh model for funasr as default")
        model_size = "paraformer-zh"

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available, using CPU. This will be slow and even this script can not work. "
            "To speed up, use a GPU enabled machine or install torch with cuda builtin."
        )
    logger.info(f"Using {num_workers} workers for processing")
    if source == "file":
        logger.info(f"Transcribing audio files in file list '{wav_source}', total: {len(audio_files)}")
        # 扫描出所有的音频文件
        audio_files = [line.strip() for line in open(wav_source, "r", encoding="UTF8").readlines()]
    elif source == "dir":
        logger.info(f"Transcribing audio files in wav dir '{wav_source}'")
        # 扫描出所有的音频文件
        audio_files = list_files(wav_source, recursive=recursive)
        audio_files = [str(file) for file in audio_files if file.suffix in AUDIO_EXTENSIONS]

    if len(audio_files) == 0:
        logger.error(f"No audio files found in source.")
        return

    # 按照 num workers 切块
    chunks = split_list_m(audio_files, chunk_size)
    logger.info(f"splited to '{len(chunks)}'")

    for idx, i_chunk in enumerate(chunks):
        logger.info(f"starting processing '{idx}-rd i_chunk', total: {len(i_chunk)}.")
        chunk_threads = split_list(i_chunk, num_workers)
        with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as executor:
            tasks = []
            for _chunk in chunk_threads:
                tasks.append(
                    executor.submit(
                        batch_transcribe,
                        files=_chunk,
                        model_size=model_size,
                        model_type=model_type,
                        lang=lang,
                        pos=len(tasks),
                    )
                )
            results = {}
            for task in tasks:
                ret = task.result()
                for res in ret.keys():
                    results[res] = ret[res]

            logger.info("Output to .lab file")
            for file in tqdm(results.keys()):
                path = replace_lastest(file, ".wav", ".lab")
                # logger.info(path)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(results[file])
