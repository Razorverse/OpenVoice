"""Clone a voice onto existing speech: python clone_test.py SOURCE_AUDIO REFERENCE_VOICE OUTPUT.wav"""
import sys, time, torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

src_audio, ref_audio, out_path = sys.argv[1:4]
device = "cuda:0" if torch.cuda.is_available() else "cpu"

t = time.time()
converter = ToneColorConverter("checkpoints_v2/converter/config.json", device=device)
converter.load_ckpt("checkpoints_v2/converter/checkpoint.pth")
print(f"loaded model in {time.time()-t:.1f}s")

t = time.time()
src_se, _ = se_extractor.get_se(src_audio, converter, target_dir="processed", vad=True)
tgt_se, _ = se_extractor.get_se(ref_audio, converter, target_dir="processed", vad=True)
print(f"extracted voices in {time.time()-t:.1f}s")

t = time.time()
converter.convert(src_audio, src_se, tgt_se, output_path=out_path, tau=0.3)
print(f"converted in {time.time()-t:.1f}s -> {out_path}")
