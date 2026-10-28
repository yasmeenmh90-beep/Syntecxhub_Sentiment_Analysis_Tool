"""
Train and evaluate sentiment classifiers.

    python train.py --dataset general    # positive / negative / neutral on everyday text (CLI default)
    python train.py --dataset imdb       # IMDb 50k reviews: the most accurate model (~92%)
    python train.py --dataset airline    # airline tweets, positive vs negative (~92%)
    python train.py --dataset reviews    # movie reviews
    python train.py --csv my_data.csv --text-col review --label-col sentiment

Steps:
  1. load labelled data and drop duplicate texts
  2. stratified 80/20 train/test split
  3. stage 1: compare 8 pipelines with cross-validation on the training set
     (CountVectorizer / TF-IDF  x  Naive Bayes / Logistic Regression  x  uni / uni+bigrams)
     stage 2: tune the best one (C or alpha, stopwords removed or kept, and on
              imbalanced data also balanced class weights)
  4. refit the best pipeline on the full training set, with probability calibration
     if it lowers the cross-validated log loss
  5. evaluate once on the untouched test set (accuracy, precision, recall, F1,
     calibration error; per source for the mixed "general" dataset)
  6. save the model, metrics and charts
"""

import argparse
import json
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split

from sentiment import plots
from sentiment.data import DATASETS, EMOTICONS_ARE_SIGNAL, load_dataset, load_tweeteval, prepare
from sentiment.preprocess import TextCleaner
from sentiment.model import (ALL_CONFIGS, build_pipeline, cross_validate, cv_log_loss, evaluate,
                             is_imbalanced, save_model, top_features, tuning_grid)

ROOT = Path(__file__).resolve().parent
VIOLET, PINK, CYAN, DIM = "#a78bfa", "#f472b6", "#22d3ee", "grey58"
console = Console()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train sentiment classifiers and pick the best one.")
    p.add_argument("--dataset", choices=DATASETS, default="general", help="built-in dataset (default: general)")
    p.add_argument("--csv", help="train on your own CSV instead")
    p.add_argument("--text-col", default="text", help="text column in the CSV (default: text)")
    p.add_argument("--label-col", default="label", help="label column in the CSV (default: label)")
    p.add_argument("--name", help="name for the saved model and results (default: dataset or CSV name)")
    p.add_argument("--test-size", type=float, default=0.2, help="share held out for testing (default: 0.2)")
    p.add_argument("--folds", type=int, help="cross-validation folds (default: 5, or 3 above 20,000 training texts)")
    p.add_argument("--no-tune", action="store_true", help="skip stage 2 (regularisation and stopword tuning)")
    p.add_argument("--no-calibrate", action="store_true",
                   help="skip the calibration check and keep the classifier's raw probabilities")
    p.add_argument("--jobs", type=int, default=-1, help="parallel CV jobs (default: all CPU cores)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-emoticons", action="store_true",
                   help="keep emoticons in the text (shows the label-leakage effect on the tweet data)")
    p.add_argument("--models-dir", default=str(ROOT / "models"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    name = args.name or (Path(args.csv).stem if args.csv else args.dataset)
    emoticons_are_signal = not args.csv and args.dataset in EMOTICONS_ARE_SIGNAL
    if args.keep_emoticons and not emoticons_are_signal:
        name += "_with_emoticons"
    remove_emoticons = not (args.keep_emoticons or emoticons_are_signal)

    console.print(Panel.fit(
        f"[bold {VIOLET}]Sentiment Analysis Tool[/]  [{DIM}]training[/]\n"
        f"[{DIM}]Syntecxhub AI Internship · Project 2[/]", border_style=VIOLET))

    # 1. Data -----------------------------------------------------------------
    with console.status(f"[{VIOLET}]Loading and cleaning data..."):
        df = load_dataset(args.dataset, args.csv, args.text_col, args.label_col)
        df, data_stats = prepare(df, TextCleaner(remove_emoticons=remove_emoticons))
    console.print(f"[{DIM}]Loaded {data_stats['loaded']:,} texts · removed {data_stats['empty_after_cleaning']:,} empty "
                  f"after cleaning, {data_stats['near_duplicates']:,} near-duplicates, "
                  f"{data_stats['conflicting_labels']:,} with conflicting labels[/]")
    counts = df["label"].value_counts()
    t = Table(title="Dataset", box=box.SIMPLE_HEAVY, title_justify="left", header_style=f"bold {CYAN}")
    t.add_column("Label"); t.add_column("Texts", justify="right"); t.add_column("Share", justify="right")
    for label, n in counts.items():
        t.add_row(label, f"{n:,}", f"{n / len(df):.0%}")
    t.add_row("[bold]used[/]", f"[bold]{len(df):,}[/]", "")
    console.print(t)
    if emoticons_are_signal:
        console.print(f"[{DIM}]Emoticons are kept: these labels were made by people, so emoticons are real signal.[/]")
    elif not remove_emoticons:
        console.print(f"[yellow]Emoticons are kept. On the tweet data the labels were created from "
                      f"emoticons, so scores will be inflated.[/]")

    # 2. Split ----------------------------------------------------------------
    # Mixed data is stratified by source and label, so every source is represented in the test set
    has_source = "source" in df.columns
    strata = df["source"] + "/" + df["label"] if has_source else df["label"]
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=strata)
    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()
    console.print(f"[{DIM}]Train {len(X_train):,} · Test {len(X_test):,} (stratified, seed {args.seed})[/]\n")

    # 3. Compare models with cross-validation ---------------------------------
    folds = args.folds or (5 if len(X_train) <= 20000 else 3)

    def run_stage(configs, label):
        out = []
        with Progress(SpinnerColumn(style=VIOLET), TextColumn("{task.description}"),
                      BarColumn(complete_style=VIOLET), TextColumn("{task.completed}/{task.total}"),
                      TimeElapsedColumn(), console=console, transient=True) as prog:
            task = prog.add_task(label, total=len(configs))
            for cfg in configs:
                prog.update(task, description=f"{label}: [bold]{cfg.name}[/]")
                res = cross_validate(cfg, X_train, y_train, folds, args.seed, remove_emoticons, args.jobs)
                # Also report each model on the test set for transparency (selection uses CV only)
                pipe = build_pipeline(cfg, remove_emoticons, args.seed).fit(X_train, y_train)
                test = evaluate(pipe, X_test, y_test)
                res.update(test_accuracy=test["accuracy"], test_f1=test["f1_macro"], stage=label)
                out.append(res)
                prog.advance(task)
        return out

    def show(rows, title, best):
        t = Table(title=title, box=box.SIMPLE_HEAVY, title_justify="left", header_style=f"bold {CYAN}")
        for col, just in [("Model", "left"), ("CV F1", "right"), ("Test acc.", "right"), ("Test F1", "right")]:
            t.add_column(col, justify=just)
        for r in sorted(rows, key=lambda r: r["cv_f1_mean"], reverse=True):
            t.add_row(r["config"].name, f"{r['cv_f1_mean']:.3f} ± {r['cv_f1_std']:.3f}",
                      f"{r['test_accuracy']:.3f}", f"{r['test_f1']:.3f}",
                      style=f"bold {VIOLET}" if r is best else None)
        console.print(t)

    results = run_stage(ALL_CONFIGS, "Stage 1")
    stage1_best = max(results, key=lambda r: r["cv_f1_mean"])
    show(results, f"Stage 1: model comparison ({folds}-fold CV on training set)", stage1_best)

    if not args.no_tune:
        imbalanced = is_imbalanced(y_train)
        if imbalanced:
            console.print(f"[{DIM}]Classes are imbalanced, so stage 2 also tries balanced class weights.[/]")
        tuned = run_stage(tuning_grid(stage1_best["config"], imbalanced), "Stage 2")
        show([stage1_best] + tuned, f"Stage 2: tuning {stage1_best['config'].name} ({folds}-fold CV)",
             max([stage1_best] + tuned, key=lambda r: r["cv_f1_mean"]))
        results += tuned
    best = max(results, key=lambda r: r["cv_f1_mean"])

    # 4-5. Refit best, final evaluation -----------------------------------------
    cfg = best["config"]
    raw = build_pipeline(cfg, remove_emoticons, args.seed).fit(X_train, y_train)
    raw_metrics = evaluate(raw, X_test, y_test)
    # Calibrate only if it improves the cross-validated log loss on the training set:
    # it fixes over-confident models (Naive Bayes) but can make well-calibrated ones worse.
    calibration_check = None
    if not args.no_calibrate:
        with console.status(f"[{VIOLET}]Checking whether probability calibration helps..."):
            ll_raw = cv_log_loss(cfg, X_train, y_train, folds, args.seed, remove_emoticons, args.jobs)
            ll_cal = cv_log_loss(cfg, X_train, y_train, folds, args.seed, remove_emoticons, args.jobs, calibrate=True)
        calibration_check = {"cv_log_loss_raw": ll_raw, "cv_log_loss_calibrated": ll_cal}
        console.print(f"[{DIM}]Calibration: CV log loss {ll_raw:.3f} raw vs {ll_cal:.3f} calibrated -> "
                      f"{'calibrated' if ll_cal < ll_raw else 'raw'} probabilities kept[/]")
    calibrated = bool(calibration_check and ll_cal < ll_raw)
    start = time.perf_counter()
    if calibrated:
        pipeline = build_pipeline(cfg, remove_emoticons, args.seed, calibrate=folds).fit(X_train, y_train)
    else:
        pipeline = raw
    fit_seconds = time.perf_counter() - start
    metrics = evaluate(pipeline, X_test, y_test)

    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column(style=DIM); t.add_column(style=f"bold {VIOLET}", justify="right")
    t.add_row("Accuracy", f"{metrics['accuracy']:.3f}")
    t.add_row("Majority-class baseline", f"{metrics['baseline_accuracy']:.3f}")
    t.add_row("Balanced accuracy", f"{metrics['balanced_accuracy']:.3f}")
    t.add_row("Precision (macro)", f"{metrics['precision_macro']:.3f}")
    t.add_row("Recall (macro)", f"{metrics['recall_macro']:.3f}")
    t.add_row("F1 (macro)", f"{metrics['f1_macro']:.3f}")
    t.add_row("Calibration error", f"{metrics['calibration_error']:.3f}"
              + (f"  [{DIM}](uncalibrated {raw_metrics['calibration_error']:.3f})[/]" if calibrated else ""))
    console.print(Panel(t, title=f"[bold]Best model on the test set[/]  [{DIM}]{cfg.name}[/]",
                        title_align="left", border_style=VIOLET, width=min(80, console.width)))

    cm = metrics["confusion_matrix"]
    t = Table(title="Confusion matrix (rows = actual)", box=box.SIMPLE, title_justify="left",
              header_style=f"bold {CYAN}")
    t.add_column("")
    for label in metrics["labels"]:
        t.add_column(f"pred {label}", justify="right")
    for label, row in zip(metrics["labels"], cm):
        t.add_row(f"actual {label}", *[str(v) for v in row])
    console.print(t)

    by_source = {}
    if has_source:
        t = Table(title="Test results per source", box=box.SIMPLE, title_justify="left", header_style=f"bold {CYAN}")
        for col in ("Source", "Texts", "Accuracy", "F1 (macro)"):
            t.add_column(col, justify="left" if col == "Source" else "right")
        for src, part in test_df.groupby("source"):
            present = sorted(part["label"].unique())   # e.g. IMDb has no neutral reviews
            pred = pipeline.predict(part["text"].tolist())
            acc = float((pred == part["label"].to_numpy()).mean())
            f1 = f1_score(part["label"], pred, labels=present, average="macro", zero_division=0)
            by_source[src] = {"n": len(part), "accuracy": acc, "f1_macro": float(f1), "labels": present}
            t.add_row(src, f"{len(part):,}", f"{acc:.3f}", f"{f1:.3f}")
        console.print(t)

    benchmark = None
    if not args.csv and args.dataset == "general":
        # TweetEval's official test set: never used for training or model selection
        bench = load_tweeteval(("test",))
        m = evaluate(pipeline, bench["text"].tolist(), bench["label"].tolist())
        pred = pipeline.predict(bench["text"].tolist())
        benchmark = {"name": "TweetEval sentiment, official test set", "n": len(bench), "accuracy": m["accuracy"],
                     "f1_macro": m["f1_macro"], "recall_macro": float(recall_score(bench["label"], pred, average="macro"))}
        console.print(f"[{DIM}]TweetEval official test set ({len(bench):,} tweets): accuracy {m['accuracy']:.3f}, "
                      f"macro-recall {benchmark['recall_macro']:.3f} (the benchmark's official metric)[/]")

    # 6. Save -------------------------------------------------------------------
    out_dir = Path(args.results_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.models_dir) / f"{name}.joblib"
    metadata = {
        "name": name, "dataset": args.csv or args.dataset, "config": cfg.name,
        "remove_emoticons": remove_emoticons, "n_train": len(X_train), "n_test": len(X_test),
        "labels": metrics["labels"], "test_accuracy": metrics["accuracy"], "test_f1": metrics["f1_macro"],
        "baseline_accuracy": metrics["baseline_accuracy"], "emoticons_are_signal": emoticons_are_signal,
        "cv_f1": best["cv_f1_mean"], "cv_folds": folds, "seed": args.seed, "data_cleaning": data_stats,
        "calibrated": calibrated, "calibration_check": calibration_check,
    }
    save_model(pipeline, model_path, metadata)

    features = top_features(pipeline, 15)
    summary = {
        **metadata,
        "fit_seconds": round(fit_seconds, 3),
        "metrics": {k: v for k, v in metrics.items() if k != "report"},
        "per_class": metrics["report"],
        "uncalibrated": {k: raw_metrics[k] for k in ("accuracy", "f1_macro", "log_loss", "calibration_error")},
        "per_source": by_source,
        "benchmark": benchmark,
        "comparison": [{
            "model": r["config"].name, "vectorizer": r["config"].vectorizer, "classifier": r["config"].classifier,
            "ngram": list(r["config"].ngram), "strength": r["config"].strength,
            "keep_stopwords": r["config"].keep_stopwords, "balanced": r["config"].balanced, "stage": r["stage"], "cv_f1_mean": round(r["cv_f1_mean"], 4),
            "cv_f1_std": round(r["cv_f1_std"], 4), "test_accuracy": round(r["test_accuracy"], 4),
            "test_f1": round(r["test_f1"], 4)} for r in results],
        "top_features": {k: [[w, round(s, 4)] for w, s in v] for k, v in features.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    plots.model_comparison(results, cfg.name, out_dir / "model_comparison.png", folds)
    plots.confusion(cm, metrics["labels"], out_dir / "confusion_matrix.png")
    plots.top_words(features, out_dir / "top_words.png")

    rel = lambda p: p.relative_to(ROOT) if p.is_relative_to(ROOT) else p  # noqa: E731
    console.print(f"\n[{CYAN}]✔[/] Model saved to [bold]{rel(model_path)}[/]")
    console.print(f"[{CYAN}]✔[/] Metrics and charts saved to [bold]{rel(out_dir)}/[/]")
    console.print(f"\nTry it: [bold {VIOLET}]python predict.py --model {name}[/]")
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
