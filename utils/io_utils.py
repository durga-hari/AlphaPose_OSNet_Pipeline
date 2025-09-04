# pipeline/io_utils.py
import json, os, threading, time
from typing import List, Dict, Any

class PoseWriter:
    """
    Writes one record per line (.jsonl) for incremental inspection.
    Optionally checkpoints a 'fc1_pose.json' array every N records.
    """
    def __init__(self, out_dir: str, base_name: str = "fc1_pose"):
        os.makedirs(out_dir, exist_ok=True)
        self.base = base_name
        self.dir = out_dir
        self.jsonl_path = os.path.join(out_dir, f"{base_name}.jsonl")
        self.checkpoint_path = os.path.join(out_dir, f"{base_name}.json")
        self._lock = threading.Lock()
        # create file if not exists
        if not os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, "w", encoding="utf-8") as f:
                pass

    def write_record(self, rec: Dict[str, Any]) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock, open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def checkpoint_json_array(self, max_lines: int = 200000) -> None:
        """Compact jsonl -> json array periodically, safe to call anytime."""
        with self._lock:
            arr = []
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        arr.append(json.loads(line))
                    except Exception:
                        # keep going; corrupt line will be ignored
                        continue
                    if i > max_lines:
                        break
            tmp = self.checkpoint_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as g:
                json.dump(arr, g, ensure_ascii=False)
            os.replace(tmp, self.checkpoint_path)

    def finalize_to_json(self) -> None:
        """Call at the very end if you want a single JSON array."""
        self.checkpoint_json_array()

def human_readable_fps(start_ts: float, frames_done: int) -> str:
    elapsed = max(1e-6, time.time() - start_ts)
    fps = frames_done / elapsed
    return f"{fps:.2f} fps ({frames_done} frames in {elapsed:.1f}s)"
