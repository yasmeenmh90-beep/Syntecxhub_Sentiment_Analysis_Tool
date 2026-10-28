"""End-to-end tests: train.py on a CSV, then predict.py on the saved model."""

import csv
import json
import subprocess
import sys

import pytest

from conftest import ROOT


@pytest.fixture
def trained_model(tmp_path, toy_csv):
    import train
    summary = train.main(["--csv", str(toy_csv), "--text-col", "review", "--label-col", "sentiment",
                          "--name", "toy", "--folds", "3",
                          "--models-dir", str(tmp_path / "models"), "--results-dir", str(tmp_path / "results")])
    return tmp_path, summary


def test_train_saves_model_metrics_and_charts(trained_model):
    tmp, summary = trained_model
    assert (tmp / "models" / "toy.joblib").exists()
    for f in ["metrics.json", "model_comparison.png", "confusion_matrix.png", "top_words.png"]:
        assert (tmp / "results" / "toy" / f).exists(), f
    metrics = json.loads((tmp / "results" / "toy" / "metrics.json").read_text())
    stages = [c["stage"] for c in metrics["comparison"]]
    assert stages.count("Stage 1") == 8 and stages.count("Stage 2") == 5
    assert metrics["test_accuracy"] > 0.9
    cleaning = metrics["data_cleaning"]
    assert cleaning["loaded"] == 240
    assert metrics["n_train"] + metrics["n_test"] == cleaning["kept"] == 240 - cleaning["near_duplicates"]
    check = metrics["calibration_check"]
    assert metrics["calibrated"] == (check["cv_log_loss_calibrated"] < check["cv_log_loss_raw"])
    assert set(metrics["uncalibrated"]) >= {"accuracy", "calibration_error"}
    assert metrics["per_source"] == {} and metrics["benchmark"] is None


def test_best_model_is_chosen_by_cross_validation(trained_model):
    _, summary = trained_model
    best_cv = max(c["cv_f1_mean"] for c in summary["comparison"])
    chosen = next(c for c in summary["comparison"] if c["model"] == summary["config"])
    assert chosen["cv_f1_mean"] == best_cv


def run_predict(*args):
    return subprocess.run([sys.executable, "predict.py", *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=120, env={"PATH": "", "COLUMNS": "100", "PYTHONIOENCODING": "utf-8"})


def test_predict_json(trained_model):
    tmp, _ = trained_model
    r = run_predict("--model", str(tmp / "models" / "toy.joblib"), "--json",
                    "--text", "I love this, amazing", "--text", "terrible and boring")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert [o["label"] for o in out] == ["positive", "negative"]


def test_predict_batch_file_to_csv(trained_model):
    tmp, _ = trained_model
    (tmp / "in.txt").write_text("great trip\n\nawful food\n")
    r = run_predict("--model", str(tmp / "models" / "toy.joblib"), "--file", str(tmp / "in.txt"),
                    "--out", str(tmp / "out.csv"))
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(tmp / "out.csv")))
    assert [r["sentiment"] for r in rows] == ["positive", "negative"]   # blank line skipped
    assert {"text", "confidence", "low_confidence", "p_positive", "p_negative"} <= set(rows[0])


def test_predict_panels_show_explanation(trained_model):
    tmp, _ = trained_model
    r = run_predict("--model", str(tmp / "models" / "toy.joblib"), "--text", "I love this amazing song")
    assert "POSITIVE" in r.stdout and "Why positive?" in r.stdout and "supports" in r.stdout


def test_interactive_session(trained_model):
    tmp, _ = trained_model
    p = subprocess.run([sys.executable, "predict.py", "--model", str(tmp / "models" / "toy.joblib")],
                       input="/help\nI love it\n/details\nawful day\n/history\n/bogus\n/quit\n",
                       cwd=ROOT, capture_output=True, text=True, timeout=120,
                       env={"PATH": "", "COLUMNS": "100", "PYTHONIOENCODING": "utf-8"})
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "/examples" in out                     # help text
    assert "Explanations off" in out
    assert "Unknown command /bogus" in out
    assert "Analysed 2 text(s)" in out


def test_missing_model_message():
    r = run_predict("--model", "does_not_exist", "--text", "hi")
    assert r.returncode == 1
    assert "No model named" in r.stdout


def test_no_tune_flag(tmp_path, toy_csv):
    import train
    summary = train.main(["--csv", str(toy_csv), "--text-col", "review", "--label-col", "sentiment",
                          "--name", "quick", "--folds", "3", "--no-tune", "--jobs", "1",
                          "--models-dir", str(tmp_path / "m"), "--results-dir", str(tmp_path / "r")])
    assert len(summary["comparison"]) == 8
    assert summary["cv_folds"] == 3


def test_no_calibrate_flag(tmp_path, toy_csv):
    import train
    summary = train.main(["--csv", str(toy_csv), "--text-col", "review", "--label-col", "sentiment",
                          "--name", "raw", "--folds", "3", "--no-tune", "--no-calibrate", "--jobs", "1",
                          "--models-dir", str(tmp_path / "m"), "--results-dir", str(tmp_path / "r")])
    assert summary["calibrated"] is False and summary["calibration_check"] is None
    assert summary["metrics"]["accuracy"] == summary["uncalibrated"]["accuracy"]
