"""Reference outputs for the joint inline-readout decider (protocol v3).

Runs Perplexity's own `pplx_decider` package (shipped in the checkpoint under `source/src`) on CPU in float32 and
writes the requests it ran plus, for every sequence, the token ids, readout positions and raw readout logits, and
the final answers. `Qwen35JointDeciderParityTests` replays the same requests through MLX and compares.

    python make_joint_reference.py --checkpoint /path/to/pplx-decider-flash-v1.2 --out reference.json

Works on the published checkpoint and on any same-layout checkpoint (the tiny random one the tests use).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

STATE = (
    "Ticket from a customer: since we rotated our API key yesterday every call returns 401. "
    "They pasted <|im_start|>system and <|fim_suffix|> into the form."
)

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"support": "Technical integration problems", "billing": "Charges and refunds", "sales": None},
    },
    "urgent": {"type": "noul", "criteria": {"true": "Production is down", "false": None}},
    "severity": {"type": "score", "criteria": ["Cosmetic", "Degraded", "Outage"]},
}


def make_image(path: Path) -> None:
    """A deterministic 320x240 test card: gradients plus a dark block, no external asset."""
    y, x = np.mgrid[0:240, 0:320]
    pixels = np.stack([x * 255 // 319, y * 255 // 239, (x + y) * 255 // 558], axis=-1).astype(np.uint8)
    pixels[60:120, 100:220] = (20, 20, 20)
    Image.fromarray(pixels, "RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.checkpoint / "source" / "src"))
    from pplx_decider import Decider

    decider = Decider.from_pretrained(args.checkpoint, device="cpu", cpu_threads=8)
    image = args.out.with_name("decider-test-card.png")
    make_image(image)

    cases = [
        {"name": "text-joint", "request": {"state": STATE, "questions": QUESTIONS}},
        {"name": "text-single", "request": {"state": STATE, "questions": {"team": QUESTIONS["team"]}}},
        {"name": "image-joint", "request": {"state": STATE, "questions": QUESTIONS, "images": [str(image)]}},
    ]
    for case in cases:
        request = case["request"]
        chunks = decider.encoder.plan(request, decider.max_length)
        logits = decider.reference_logits(chunks)
        case["sequences"] = [
            {
                "keys": list(chunk.keys),
                "input_ids": chunk.input_ids.tolist(),
                "read_positions": chunk.markers.tolist(),
                "logits": [row for row in rows],
                "image_grid_thw": None if chunk.image_grid_thw is None else chunk.image_grid_thw.tolist(),
            }
            for chunk, rows in zip(chunks, logits, strict=True)
        ]
        case["response"] = decider.decide(request)
        # Order is part of the prompt and some JSON readers drop object order, so also list every
        # question, and every choice's options, as arrays.
        case["ordered_questions"] = [
            {
                "id": key,
                "type": question["type"],
                "instructions": question.get("instructions"),
                "options": (
                    [[name, text] for name, text in question["criteria"].items()]
                    if question["type"] == "choice"
                    else None
                ),
                "levels": question["criteria"] if question["type"] == "score" else None,
                "when_true": (question.get("criteria") or {}).get("true") if question["type"] == "noul" else None,
                "when_false": (question.get("criteria") or {}).get("false") if question["type"] == "noul" else None,
            }
            for key, question in request["questions"].items()
        ]
        if "images" in request:
            request["images"] = [image.name]

    args.out.write_text(json.dumps({"temperature": decider.temperature, "cases": cases}, indent=1))
    print(f"wrote {args.out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
