"""End-to-end tests of the command-line entry points and attention plotting,
using a checkpoint trained on synthetic data (``trained_checkpoint`` fixture)."""

import json
import math
import os

import matplotlib
import pytest
import torch
from PIL import Image

from src import evaluate, predict, train
from src.attention_viz import attention_entropy, plot_attention
from src.utils import load_checkpoint


# ------------------------------------------------------------------ attention viz
def test_attention_entropy_bounds():
    uniform = torch.full((49,), 1 / 49)
    one_hot = torch.zeros(49)
    one_hot[3] = 1.0
    assert attention_entropy(uniform) == pytest.approx(math.log(49), rel=1e-5)
    assert attention_entropy(one_hot) == pytest.approx(0.0, abs=1e-6)


def test_plot_attention_one_panel_per_word():
    matplotlib.use("Agg")
    words = ["<start>", "a", "dog", "runs", "<end>"]
    attn = [torch.softmax(torch.randn(49), 0) for _ in range(len(words) - 1)]
    fig = plot_attention(Image.new("RGB", (120, 80)), words, attn)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert [t.split()[0] for t in titles] == ["a", "dog", "runs"]  # no <start>/<end> panels
    empty = plot_attention(Image.new("RGB", (32, 32)), ["<start>", "<end>"], [torch.zeros(49)])
    assert len(empty.axes) == 1


# ------------------------------------------------------------------------ predict
def test_predict_cli_json_and_unreadable(trained_checkpoint, tmp_path, capsys):
    good = os.path.join(trained_checkpoint["img_dir"], "img_3.jpg")
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(SystemExit) as exc:
        predict.main(
            [
                good,
                str(bad),
                "--checkpoint",
                trained_checkpoint["checkpoint"],
                "--device",
                "cpu",
                "--json",
                "--beam",
                "2",
            ]
        )
    assert exc.value.code == 2  # partial failure is signalled to the shell
    out = json.loads(capsys.readouterr().out)
    assert isinstance(out[good], str) and out[str(bad)] is None


def test_predict_attention_figure(trained_checkpoint, tmp_path):
    good = os.path.join(trained_checkpoint["img_dir"], "img_5.jpg")
    fig_path = tmp_path / "attn.png"
    predict.main(
        [
            good,
            "--checkpoint",
            trained_checkpoint["checkpoint"],
            "--device",
            "cpu",
            "--attention-out",
            str(fig_path),
        ]
    )
    assert fig_path.stat().st_size > 0


def test_caption_files_batches(trained_checkpoint):
    model, vocab, _ = load_checkpoint(trained_checkpoint["checkpoint"])
    paths = [os.path.join(trained_checkpoint["img_dir"], f"img_{i}.jpg") for i in range(5)]
    caps = predict.caption_files(model, vocab, paths, beam=1, batch_size=2)
    assert list(caps) == paths and all(isinstance(c, str) for c in caps.values())


# ----------------------------------------------------------------------- evaluate
def test_evaluate_cli_writes_metrics(trained_checkpoint, tmp_path):
    out = tmp_path / "eval.json"
    evaluate.main(
        [
            "--checkpoint",
            trained_checkpoint["checkpoint"],
            "--split",
            "val",
            "--beam",
            "2",
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--out",
            str(out),
        ]
    )
    result = json.loads(out.read_text())
    assert result["split"] == "val" and result["beam"] == 2
    for key in ("BLEU-1", "BLEU-4", "CIDEr", "ROUGE-L", "ms_per_image"):
        assert key in result["metrics"] and math.isfinite(result["metrics"][key])
    assert result["samples"] and {"image", "prediction", "references"} <= set(result["samples"][0])


def test_evaluate_cli_missing_checkpoint_exits_1(tmp_path):
    with pytest.raises(SystemExit) as exc:
        evaluate.main(
            ["--checkpoint", str(tmp_path / "nope.pth"), "--out", str(tmp_path / "e.json")]
        )
    assert exc.value.code == 1


# -------------------------------------------------------------------------- train
def test_train_cli_rejects_bad_values(capsys):
    with pytest.raises(SystemExit) as exc:
        train.main(["--lr", "abc"])
    assert exc.value.code == 2
    assert "expects a number" in capsys.readouterr().err


def test_train_cli_missing_data_exits_1(tmp_path):
    with pytest.raises(SystemExit) as exc:
        train.main(["--img_dir", str(tmp_path / "none"), "--out_dir", str(tmp_path / "o")])
    assert exc.value.code == 1


def test_train_cli_resume_finished_run_is_noop(trained_checkpoint):
    last = os.path.join(trained_checkpoint["out_dir"], "last.pth")
    hist = train.run(
        train.load_config(
            os.path.join(trained_checkpoint["out_dir"], "config.yaml"), [f"train.resume={last}"]
        )
    )
    assert len(hist["train_loss"]) == 2  # history restored, no extra epochs


# ------------------------------------------------------------ predict: folders etc.
def test_collect_images_from_folders(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("b.jpg", "a.PNG", "notes.txt", "sub/c.jpeg"):
        (tmp_path / name).write_bytes(b"x")
    flat = predict.collect_images([str(tmp_path)])
    assert [os.path.basename(p) for p in flat] == ["a.PNG", "b.jpg"]  # sorted, images only
    deep = predict.collect_images([str(tmp_path)], recursive=True)
    assert {os.path.basename(p) for p in deep} == {"a.PNG", "b.jpg", "c.jpeg"}
    explicit = predict.collect_images([str(tmp_path / "notes.txt")])  # explicit files are kept
    assert explicit == [str(tmp_path / "notes.txt")]
    with pytest.raises(FileNotFoundError):
        predict.collect_images([str(tmp_path / "missing")])


@pytest.mark.parametrize("fmt", ["text", "json", "jsonl", "csv"])
def test_predict_folder_all_formats(trained_checkpoint, tmp_path, fmt):
    out = tmp_path / f"captions.{fmt}"
    predict.main(
        [trained_checkpoint["img_dir"], "--checkpoint", trained_checkpoint["checkpoint"],
         "--device", "cpu", "--beam", "1", "--format", fmt, "--output", str(out)]
    )  # fmt: skip
    text = out.read_text()
    if fmt == "json":
        rows = json.loads(text)
        assert len(rows) == 20 and all(r["status"] == "ok" for r in rows)
    elif fmt == "jsonl":
        assert len([json.loads(line) for line in text.splitlines()]) == 20
    elif fmt == "csv":
        lines = text.splitlines()
        assert lines[0] == "image,caption,status" and len(lines) == 21
    else:
        assert len(text.splitlines()) == 20 and all("\t" in line for line in text.splitlines())


def test_predict_save_attention_for_folder(trained_checkpoint, tmp_path):
    folder = tmp_path / "imgs"
    folder.mkdir()
    for i in (1, 2):
        src = os.path.join(trained_checkpoint["img_dir"], f"img_{i}.jpg")
        (folder / f"img_{i}.jpg").write_bytes(open(src, "rb").read())
    viz = tmp_path / "viz"
    predict.main(
        [str(folder), "--checkpoint", trained_checkpoint["checkpoint"], "--device", "cpu",
         "--save-attention", str(viz), "--output", str(tmp_path / "out.txt")]
    )  # fmt: skip
    assert sorted(os.listdir(viz)) == ["img_1_attention.png", "img_2_attention.png"]


def test_predict_no_images_exits_1(trained_checkpoint, tmp_path):
    with pytest.raises(SystemExit) as exc:
        predict.main([str(tmp_path), "--checkpoint", trained_checkpoint["checkpoint"]])
    assert exc.value.code == 1
