"""Speak text directly in an ElevenLabs voice (e.g. your own cloned voice), no OpenVoice step.

Usage:
    python elevenlabs_speak.py VOICE_ID "Text to speak" outputs/out.mp3

ELEVENLABS_API_KEY must be set (in a cloud session the network proxy supplies it).
"""
import argparse
import os

from elevenlabs.client import ElevenLabs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("voice_id")
    p.add_argument("text")
    p.add_argument("output")
    p.add_argument("--model", default="eleven_multilingual_v2")
    args = p.parse_args()

    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", "placeholder"))
    audio = client.text_to_speech.convert(
        voice_id=args.voice_id,
        model_id=args.model,
        text=args.text,
        output_format="mp3_44100_128",
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
