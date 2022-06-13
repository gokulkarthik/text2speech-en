import argparse
import os

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def str2bool(v):
    """
    src: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse 
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Traning and evaluation script for hateful meme classification')

    # dataset parameters
    parser.add_argument('--dataset_name', default='ljspeech', choices=['ljspeech'])
    parser.add_argument('--dataset_path', default='../../data/tts/LJSpeech-1.1', type=str)
    parser.add_argument('--language', default='en', choices=['en'])

    # model parameters
    parser.add_argument('--model', default='glowtts', choices=['glowtts'])

    # training parameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--batch_size_eval', default=64, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--num_workers_eval', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--use_phonemes', default=True, type=str2bool)
    parser.add_argument('--phoneme_language', default='en-us', choices=['en-us'])
    parser.add_argument('--print_step', default=25, type=int)
    parser.add_argument('--print_eval', default=False, type=str2bool)
    parser.add_argument('--mixed_precision', default=False, type=str2bool)
    parser.add_argument('--output_path', default='output', type=str)
    parser.add_argument('--save_step', default=1000, type=int)

    return parser

def main(args):
    dataset_config = BaseDatasetConfig(
        name=args.dataset_name, meta_file_train="metadata.csv", path=args.dataset_path
    )

    if args.model == 'glowtts':
        config = GlowTTSConfig(
            batch_size=args.batch_size,
            eval_batch_size=args.batch_size_eval,
            num_loader_workers=args.num_workers,
            num_eval_loader_workers=args.num_workers_eval,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=args.epochs,
            text_cleaner="phoneme_cleaners",
            use_phonemes=args.use_phonemes,
            phoneme_language=args.phoneme_language,
            phoneme_cache_path=os.path.join(args.output_path, "phoneme_cache"),
            print_step=args.print_step,
            print_eval=args.print_eval,
            mixed_precision=args.mixed_precision,
            output_path=args.output_path,
            datasets=[dataset_config],
            save_step=args.save_step,
        )

    # load preprocessors
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # load data
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # load model
    if args.model == 'glowtts':
        model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

    # set trainer
    trainer = Trainer(
        TrainerArgs(), config, args.output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    # run training
    trainer.fit()


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
