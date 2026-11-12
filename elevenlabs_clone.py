"""ElevenLabs speaks the text, OpenVoice swaps in the voice from your reference clip.

Usage:
    python elevenlabs_clone.py "Text to speak" my_voice.mp3 outputs/out.wav [--voice-id ID]

ELEVENLABS_API_KEY must be set (in a cloud session the network proxy supplies it).
"""
import argparse
import os
import time

import torch
from elevenlabs.client import ElevenLabs

from openvoice import se_extractor
from openvoice.api import ToneColorConverter

CKPT = "checkpoints_v2/converter"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # ElevenLabs premade voice "George"


def elevenlabs_tts(text, voice_id, out_path):
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", "placeholder"))
    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        text=text,
        output_format="mp3_44100_128",
    )
    with open(out_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)


def source_embedding(converter, voice_id, text_audio):
    """Embedding of the ElevenLabs voice, cached per voice id.

    Extracting from one short line is noisy (and fails under ~5s of speech),
    so the first run for a voice generates a longer calibration sample.
    """
    cache = f"processed/elevenlabs_{voice_id}_se.pth"
    if os.path.exists(cache):
        return torch.load(cache).to(converter.device)
    sample = f"processed/elevenlabs_{voice_id}_calibration.mp3"
    elevenlabs_tts(
        "The quick brown fox jumps over the lazy dog. I'm reading this passage so the "
        "system can learn how my voice sounds. It covers a range of sounds, from soft "
        "whispers to sharp consonants, and from low tones to high ones. Thanks for "
        "listening, and have a wonderful day wherever you happen to be.",
        voice_id,
        sample,
    )
    se, _ = se_extractor.get_se(sample, converter, target_dir="processed", vad=True)
    torch.save(se.cpu(), cache)
    return se


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("reference", help="audio clip of the voice to clone (10s+ recommended)")
    p.add_argument("output")
    p.add_argument("--voice-id", default=DEFAULT_VOICE_ID)
    p.add_argument("--tau", type=float, default=0.3)
    args = p.parse_args()

    os.makedirs("processed", exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    t = time.time()
    tts_path = "processed/elevenlabs_tts.mp3"
    elevenlabs_tts(args.text, args.voice_id, tts_path)
    print(f"ElevenLabs TTS: {time.time()-t:.1f}s")

    converter = ToneColorConverter(f"{CKPT}/config.json", device=device)
    converter.load_ckpt(f"{CKPT}/checkpoint.pth")

    t = time.time()
    src_se = source_embedding(converter, args.voice_id, tts_path)
    tgt_se, _ = se_extractor.get_se(args.reference, converter, target_dir="processed", vad=True)
    converter.convert(tts_path, src_se, tgt_se, output_path=args.output, tau=args.tau)
    print(f"OpenVoice conversion: {time.time()-t:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
