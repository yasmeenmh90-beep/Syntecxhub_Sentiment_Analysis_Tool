"""
Predict sentiment from the command line.

    python predict.py                                  # interactive mode (general model: also knows neutral)
    python predict.py --model imdb                     # most accurate on reviews (positive / negative)
    python predict.py --text "I love this!" --text "Worst day ever"
    python predict.py --file inputs.txt --out predictions.csv   # one text per line
    python predict.py --text "great job" --json        # machine-readable output
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sentiment.model import LOW_CONFIDENCE, load_model, predict

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
VIOLET, PINK, CYAN, DIM, AMBER = "#a78bfa", "#f472b6", "#22d3ee", "grey58", "#fbbf24"
LABEL_STYLE = {"positive": VIOLET, "negative": PINK, "neutral": CYAN}

EXAMPLES = [
    "I absolutely love this new phone, the camera is amazing!",
    "Worst customer service ever. I waited two hours and nobody helped.",
    "The movie was not bad at all, actually pretty fun",
    "I'm so tired of this rain",
    "Thanks for the follow, have a great weekend",
    "The meeting is at 3pm in room 204",
    "Is there wifi on the plane?",
]

console = Console()


def resolve_model(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".joblib" and p.exists():
        return p
    return MODELS_DIR / f"{name_or_path}.joblib"


def available_models() -> list:
    return sorted(p.stem for p in MODELS_DIR.glob("*.joblib"))


# general handles any everyday text and has a neutral class; imdb/airline are more accurate in their domain
PREFERRED = ("general", "imdb", "airline", "airline3", "tweets", "reviews")


def default_model() -> str:
    trained = available_models()
    return next((m for m in PREFERRED if m in trained), trained[0] if trained else "imdb")


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
def bar(value: float, width: int = 28, color: str = VIOLET) -> Text:
    filled = round(value * width)
    t = Text("█" * filled, style=color)
    t.append("░" * (width - filled), style="grey30")
    t.append(f"  {value:6.1%}", style="bold")
    return t


def render(result: dict, show_details: bool = True) -> Panel:
    label = result["label"]
    color = LABEL_STYLE.get(label, CYAN)
    head = Text()
    head.append(f" {label.upper()} ", style=f"bold black on {color}")
    head.append("   confidence ", style=DIM)
    head.append(f"{result['confidence']:.0%}", style=f"bold {color}")
    parts = [head, Text("")]

    probs = Table.grid(padding=(0, 2))
    probs.add_column(style=DIM, min_width=9)
    probs.add_column()
    for cls, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        probs.add_row(cls, bar(p, color=LABEL_STYLE.get(cls, CYAN)))
    parts.append(probs)

    if result["low_confidence"]:
        reason = ("No recognisable words after cleaning." if not result["tokens"] else
                  f"Below {LOW_CONFIDENCE:.0%}: the text may be neutral, mixed or unlike the training data.")
        parts += [Text(""), Text(f"⚠  Low confidence. {reason}", style=AMBER)]

    if show_details:
        parts.append(Text(""))
        words = Text("Words used  ", style=DIM)
        words.append(" ".join(result["tokens"]) or "(none)", style="italic")
        parts.append(words)
        if result["explanation"]:
            why = Table(box=None, show_header=False, padding=(0, 1))
            why.add_column(style=DIM)
            why.add_column()
            why.add_column(justify="right")
            for word, weight in result["explanation"]:
                towards = weight > 0
                arrow = Text("▲ supports" if towards else "▼ against ", style=color if towards else "grey62")
                why.add_row(word, arrow, f"{weight:+.2f}")
            parts += [Text(""), Text(f"Why {label}?", style="bold"), why]

    return Panel(Group(*parts), border_style=color, box=box.ROUNDED,
                 title=f"[{DIM}]“{shorten(result['text'], 60)}”[/]", title_align="left",
                 width=min(84, console.width))


def shorten(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def model_info(meta: dict) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style=DIM)
    t.add_column()
    t.add_row("Model", f"[bold]{meta['name']}[/]")
    t.add_row("Pipeline", meta["config"])
    t.add_row("Trained on", f"{meta['n_train']:,} texts ({meta['dataset']})")
    t.add_row("Test accuracy", f"[bold {VIOLET}]{meta['test_accuracy']:.1%}[/]"
              + (f"  [{DIM}](always-majority baseline {meta['baseline_accuracy']:.1%})[/]" if "baseline_accuracy" in meta else ""))
    t.add_row("Test F1 (macro)", f"[bold {VIOLET}]{meta['test_f1']:.3f}[/]")
    t.add_row("Classes", ", ".join(meta["labels"]))
    t.add_row("Confidence", "calibrated (sigmoid, cross-validated)" if meta.get("calibrated") else "raw classifier probabilities")
    if meta.get("remove_emoticons", True):
        emo = "removed"
    elif meta.get("emoticons_are_signal"):
        emo = "kept (labels made by people, so emoticons are real signal)"
    else:
        emo = "[yellow]kept (labels came from emoticons: inflated scores)[/]"
    t.add_row("Emoticons", emo)
    return Panel(t, title="[bold]Model[/]", title_align="left", border_style=CYAN, width=min(84, console.width))


HELP = """[bold]Type any sentence[/] to analyse it, or use a command:

  [bold]/examples[/]        run a few sample sentences
  [bold]/model NAME[/]      switch model (e.g. /model imdb)
  [bold]/models[/]          list trained models
  [bold]/info[/]            show the current model's test scores
  [bold]/details[/]         turn the word-level explanation on or off
  [bold]/history[/]         show this session's predictions
  [bold]/quit[/]            exit"""


# --------------------------------------------------------------------------- #
#  Modes
# --------------------------------------------------------------------------- #
def interactive(pipeline, meta):
    console.print(Panel.fit(
        f"[bold {VIOLET}]Sentiment Analysis Tool[/]\n[{DIM}]Type a sentence and press Enter · /help for commands[/]",
        border_style=VIOLET))
    console.print(model_info(meta))
    history, details = [], True
    while True:
        try:
            text = console.input(f"\n[bold {PINK}]Text ❯ [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            continue
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/help":
            console.print(Panel(HELP, border_style=CYAN, width=min(84, console.width)))
        elif cmd == "/models":
            console.print("Trained models: " + (", ".join(available_models()) or "none, run python train.py"))
        elif cmd == "/model":
            try:
                pipeline, meta = load_model(resolve_model(arg.strip()))
                console.print(model_info(meta))
            except (FileNotFoundError, KeyError) as e:
                console.print(f"[red]{e}[/]  Available: {', '.join(available_models())}")
        elif cmd == "/info":
            console.print(model_info(meta))
        elif cmd == "/details":
            details = not details
            console.print(f"Explanations {'on' if details else 'off'}.")
        elif cmd == "/history":
            t = Table(box=box.SIMPLE, header_style=f"bold {CYAN}")
            t.add_column("Text", max_width=50, overflow="ellipsis")
            t.add_column("Sentiment")
            t.add_column("Confidence", justify="right")
            for r in history:
                t.add_row(shorten(r["text"], 50), Text(r["label"], style=LABEL_STYLE.get(r["label"], CYAN)),
                          f"{r['confidence']:.0%}")
            console.print(t if history else "No predictions yet.")
        elif cmd == "/examples":
            for ex in EXAMPLES:
                r = predict(pipeline, ex)
                history.append(r)
                console.print(render(r, show_details=False))
        elif cmd.startswith("/"):
            console.print(f"Unknown command {cmd}. Type /help.")
        else:
            r = predict(pipeline, text)
            history.append(r)
            console.print(render(r, details))
    console.print(f"[{DIM}]Analysed {len(history)} text(s). Goodbye![/]")


def batch(pipeline, texts, out=None, as_json=False):
    results = [predict(pipeline, t) for t in texts]
    if out:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            classes = list(results[0]["probabilities"]) if results else []
            w.writerow(["text", "sentiment", "confidence", "low_confidence"] + [f"p_{c}" for c in classes])
            for r in results:
                w.writerow([r["text"], r["label"], f"{r['confidence']:.4f}", r["low_confidence"]]
                           + [f"{r['probabilities'][c]:.4f}" for c in classes])
        console.print(f"[{CYAN}]✔[/] Wrote {len(results)} predictions to [bold]{out}[/]")
    elif as_json:
        print(json.dumps([{k: r[k] for k in ("text", "label", "confidence", "probabilities", "low_confidence")}
                          for r in results], indent=2))
    else:
        for r in results:
            console.print(render(r))
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="Predict the sentiment of text.")
    p.add_argument("--model", help="model name in models/ or a .joblib path (default: general)")
    p.add_argument("--text", action="append", help="text to analyse (repeat for several)")
    p.add_argument("--file", help="text file, one input per line")
    p.add_argument("--out", help="write batch predictions to this CSV")
    p.add_argument("--json", action="store_true", help="print JSON instead of panels")
    args = p.parse_args(argv)
    args.model = args.model or default_model()

    try:
        pipeline, meta = load_model(resolve_model(args.model))
    except FileNotFoundError:
        trained = available_models()
        console.print(f"[red]No model named '{args.model}'.[/] "
                      + (f"Available: {', '.join(trained)}" if trained else "Train one first: python train.py"))
        return 1

    texts = list(args.text or [])
    if args.file:
        texts += [line.strip() for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if texts:
        batch(pipeline, texts, args.out, args.json)
    else:
        interactive(pipeline, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
