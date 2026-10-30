"""
abstract_classifier_gui.py
Desktop tool to classify research abstracts (Science / Business / Social Science)
with the best model saved by Project1_26566019_structured.ipynb.

You can give it text in four ways:
  Tab 1 (Single Abstract)
    - paste text into the box
    - upload a .txt file (its content is loaded into the box)
  Tab 2 (Batch Upload)
    - upload a .csv file  (choose which column holds the abstracts)
    - upload a .txt file  (one abstract per line, per blank-line-separated block, or the whole file)

Batch results are saved to  outputs/predictions_<input file name>.csv

Requirements: pandas, scikit-learn, joblib, nltk (nltk is optional but strongly recommended).
"""
import os
import re
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

# ------------------------- Settings -------------------------
MODEL_CANDIDATES = [
    "outputs/saved_models/best_model_pipeline.pkl",
    "outputs/saved_models/best_model.pkl",       # name used by the notebook (section 8.1)
]
OUTPUT_DIR = "outputs"
DEFAULT_COLUMN = "Abstract"

# ------------------------- Text preprocessing -------------------------
# The notebook trained the models on the *preprocessed* column `Cleaned_Abstract`
# (lemmatised, stopwords removed), not on raw abstracts. The same steps are repeated here
# so that new text looks like the training data. Without NLTK, a simplified version is used.
try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    _NLTK_IMPORTED = True
except ImportError:
    _NLTK_IMPORTED = False

NLTK_READY = False
STOP_WORDS = set()
LEMMATIZER = None
NEGATION_WORDS = {"no", "nor", "not", "never", "neither", "cannot", "without"}

CONTRACTIONS_DICT = {
    "can't": "cannot", "won't": "will not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "shouldn't": "should not", "couldn't": "could not", "wouldn't": "would not",
    "shan't": "shall not", "ain't": "is not",
}
CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTIONS_DICT, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)
GENERIC_NT_PATTERN = re.compile(r"\b(\w+)n't\b", flags=re.IGNORECASE)
CHAR_MAP = str.maketrans({
    "\u2019": "'", "\u2018": "'", "\u02bc": "'",
    "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
})


def setup_nltk() -> bool:
    """Make sure NLTK resources exist (downloads them once if needed)."""
    global NLTK_READY, STOP_WORDS, LEMMATIZER
    if not _NLTK_IMPORTED:
        return False
    needed = {
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    }
    try:
        for name, path in needed.items():
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(name, quiet=True)
                nltk.data.find(path)
        STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS
        LEMMATIZER = WordNetLemmatizer()
        NLTK_READY = True
    except Exception:
        NLTK_READY = False
    return NLTK_READY


def expand_contractions(text):
    text = CONTRACTION_PATTERN.sub(lambda m: CONTRACTIONS_DICT[m.group(0).lower()], text)
    return GENERIC_NT_PATTERN.sub(r"\1 not", text)


def clean_unicode_and_noise(text):
    text = text.translate(CHAR_MAP)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return text


def _prepare(text):
    return expand_contractions(clean_unicode_and_noise(text))


def _get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


@lru_cache(maxsize=100_000)
def _cached_lemma(word, pos):
    return LEMMATIZER.lemmatize(word, pos=pos)


def _clean_tagged_tokens(pos_tags):
    out = []
    for word, tag in pos_tags:
        wn_pos = _get_wordnet_pos(tag)
        for w in re.findall(r"[a-z0-9]+", word.lower()):
            if w in STOP_WORDS:
                continue
            if len(w) == 1 and not w.isdigit():
                continue
            out.append(_cached_lemma(w, wn_pos))
    return " ".join(out)


def _basic_clean(text):
    """Fallback when NLTK is unavailable: no lemmatisation or stopword removal."""
    return " ".join(re.findall(r"[a-z0-9]+", _prepare(text).lower()))


def preprocess_corpus(texts):
    """Same pipeline as the notebook (section 3). Returns one cleaned string per input."""
    if not NLTK_READY:
        return [_basic_clean(t) if isinstance(t, str) and t.strip() else "" for t in texts]
    results = [""] * len(texts)
    prepared, valid_idx = [], []
    for i, t in enumerate(texts):
        if isinstance(t, str) and t.strip():
            prepared.append(word_tokenize(_prepare(t)))
            valid_idx.append(i)
    for i, tags in zip(valid_idx, nltk.pos_tag_sents(prepared)):
        results[i] = _clean_tagged_tokens(tags)
    return results


# ------------------------- Model + prediction -------------------------
def load_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return joblib.load(path), path
    return None, None


def predict_texts(model, texts):
    """Returns (labels, confidences). Blank / unusable texts get None."""
    cleaned = preprocess_corpus(texts)
    idx = [i for i, c in enumerate(cleaned) if c.strip()]
    labels = [None] * len(texts)
    conf = [None] * len(texts)
    if not idx:
        return labels, conf
    batch = [cleaned[i] for i in idx]
    preds = model.predict(batch)
    for i, p in zip(idx, preds):
        labels[i] = p
    if hasattr(model, "predict_proba"):
        top = model.predict_proba(batch).max(axis=1)
        for i, p in zip(idx, top):
            conf[i] = float(p)
    return labels, conf


def predict_single(model, text):
    text = str(text).strip()
    if not text:
        return "Please enter an abstract or upload a .txt file."
    cleaned = preprocess_corpus([text])[0]
    if not cleaned.strip():
        return "The text has no usable words after cleaning."
    label = model.predict([cleaned])[0]
    res = f"Predicted Class: {label}\n"
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([cleaned])[0]
        res += "\nProbabilities:\n"
        for c, p in sorted(zip(classes, probs), key=lambda x: -x[1]):
            res += f"  - {c}: {p:.2%}\n"
    elif hasattr(model, "decision_function"):
        scores = model.decision_function([cleaned])[0]
        res += "\nDecision scores (higher = more likely; not probabilities):\n"
        for c, s in sorted(zip(classes, scores), key=lambda x: -x[1]):
            res += f"  - {c}: {s:+.3f}\n"
    if not NLTK_READY:
        res += "\nNote: NLTK not available, so a simplified cleaning was used (accuracy may be lower)."
    return res


# ------------------------- File reading -------------------------
def read_text_file(path):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the text file.")


def split_text(content, mode):
    """mode: 'line' (one abstract per line), 'blank' (separated by blank lines), 'whole'."""
    if mode == "line":
        parts = content.splitlines()
    elif mode == "blank":
        parts = re.split(r"\n\s*\n", content.replace("\r\n", "\n"))
    else:
        parts = [content]
    return [p.strip() for p in parts if p.strip()]


def run_batch(model, path, column, txt_mode):
    """Load a CSV or TXT file, predict, save results. Returns (dataframe, saved_path, n_predicted)."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        matched = next((c for c in df.columns if str(c).strip().lower() == column.lower()), None)
        if matched is None:
            raise KeyError(f"Column '{column}' not found. Available columns: {list(df.columns)}")
        texts = df[matched].fillna("").astype(str).str.strip().tolist()
    elif ext == ".txt":
        items = split_text(read_text_file(path), txt_mode)
        if not items:
            raise ValueError("The text file is empty.")
        df = pd.DataFrame({"Abstract": items})
        texts = items
    else:
        raise ValueError("Unsupported file type. Please choose a .csv or .txt file.")

    labels, conf = predict_texts(model, texts)
    df["Predicted_Category"] = labels
    if any(c is not None for c in conf):
        df["Confidence"] = conf

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"predictions_{Path(path).stem}.csv")
    df.to_csv(out_path, index=False)
    return df, out_path, sum(l is not None for l in labels)


# ------------------------- GUI -------------------------
def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Abstract Classifier Tool")
    root.geometry("680x640")
    ttk.Style().theme_use("clam")

    model, model_path = load_model()
    if model is None:
        messagebox.showerror(
            "Error",
            "Model file not found. Looked for:\n" + "\n".join(MODEL_CANDIDATES) +
            "\n\nRun the notebook (section 8) to save your pipeline first.",
        )
        root.destroy()
        return

    status = tk.StringVar(value="Loading text-processing resources...")
    ttk.Label(root, textvariable=status, foreground="#555").pack(side="bottom", anchor="w", padx=12, pady=(0, 6))

    def init_nltk():
        ok = setup_nltk()
        msg = f"Model: {os.path.basename(model_path)}  |  " + (
            "NLTK ready" if ok else "NLTK unavailable: using simplified cleaning (pip install nltk)")
        root.after(0, lambda: status.set(msg))

    threading.Thread(target=init_nltk, daemon=True).start()

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    def set_text(widget, content):
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.config(state="disabled")

    # --- Tab 1: single abstract ---
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Single Abstract")

    ttk.Label(tab1, text="Paste abstract text or upload a .txt file:",
              font=("Helvetica", 11, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
    text_entry = tk.Text(tab1, height=10, width=75, wrap="word")
    text_entry.pack(padx=10, pady=5, fill="x")

    def upload_txt_single():
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not path:
            return
        try:
            content = read_text_file(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        text_entry.delete("1.0", tk.END)
        text_entry.insert(tk.END, content.strip())
        status.set(f"Loaded {os.path.basename(path)}")

    single_out = tk.Text(tab1, height=9, width=75, state="disabled", bg="#f4f4f4")

    def on_predict_single():
        set_text(single_out, predict_single(model, text_entry.get("1.0", tk.END)))

    btns = ttk.Frame(tab1)
    btns.pack(pady=5)
    ttk.Button(btns, text="Upload .txt...", command=upload_txt_single).pack(side="left", padx=5)
    ttk.Button(btns, text="Classify Abstract", command=on_predict_single).pack(side="left", padx=5)
    ttk.Button(btns, text="Clear", command=lambda: (text_entry.delete("1.0", tk.END),
                                                    set_text(single_out, ""))).pack(side="left", padx=5)
    ttk.Label(tab1, text="Result:", font=("Helvetica", 10, "bold")).pack(anchor="w", padx=10)
    single_out.pack(padx=10, pady=5, fill="x")

    # --- Tab 2: batch upload ---
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="Batch Upload (CSV / TXT)")

    selected_file = tk.StringVar(value="")
    txt_mode = tk.StringVar(value="line")

    def browse_file():
        path = filedialog.askopenfilename(
            filetypes=[("CSV or Text", "*.csv *.txt"), ("CSV Files", "*.csv"), ("Text Files", "*.txt")])
        if path:
            selected_file.set(path)

    ttk.Label(tab2, text="Select a .csv or .txt file:", font=("Helvetica", 10, "bold")).pack(
        anchor="w", padx=10, pady=(12, 2))
    file_frame = ttk.Frame(tab2)
    file_frame.pack(fill="x", padx=10)
    ttk.Entry(file_frame, textvariable=selected_file, width=58).pack(side="left", padx=(0, 5))
    ttk.Button(file_frame, text="Browse...", command=browse_file).pack(side="left")

    csv_opts = ttk.LabelFrame(tab2, text="CSV options")
    csv_opts.pack(fill="x", padx=10, pady=(12, 4))
    ttk.Label(csv_opts, text="Text column name:").pack(side="left", padx=8, pady=6)
    col_entry = ttk.Entry(csv_opts, width=25)
    col_entry.insert(0, DEFAULT_COLUMN)
    col_entry.pack(side="left", padx=4)

    txt_opts = ttk.LabelFrame(tab2, text="TXT options (how to split the file into abstracts)")
    txt_opts.pack(fill="x", padx=10, pady=4)
    for label, val in [("One abstract per line", "line"),
                       ("Blank-line separated blocks", "blank"),
                       ("Whole file = one abstract", "whole")]:
        ttk.Radiobutton(txt_opts, text=label, variable=txt_mode, value=val).pack(
            side="left", padx=8, pady=6)

    progress = ttk.Progressbar(tab2, mode="indeterminate")
    csv_out = tk.Text(tab2, height=12, width=75, state="disabled", bg="#f4f4f4")
    run_btn = ttk.Button(tab2, text="Run Predictions")

    def on_predict_batch():
        path = selected_file.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Warning", "Please select a valid .csv or .txt file.")
            return
        run_btn.config(state="disabled")
        progress.pack(fill="x", padx=10, pady=(0, 4), before=csv_out)
        progress.start(12)
        set_text(csv_out, "Processing... large files can take a minute.")

        col, mode = col_entry.get().strip(), txt_mode.get()

        def worker():
            try:
                df, out_path, n = run_batch(model, path, col, mode)
                counts = df["Predicted_Category"].value_counts().to_string()
                skipped = len(df) - n
                summary = (f"Predictions generated for {n} of {len(df)} rows"
                           + (f" ({skipped} blank/unusable skipped)" if skipped else "") + ".\n"
                           f"Saved to: {out_path}\n\nClass counts:\n{counts}")
                if not NLTK_READY:
                    summary += "\n\nNote: NLTK not available, simplified cleaning was used."
                root.after(0, lambda: finish(summary, None))
            except Exception as e:
                root.after(0, lambda: finish(None, str(e)))

        def finish(summary, error):
            progress.stop()
            progress.pack_forget()
            run_btn.config(state="normal")
            if error:
                set_text(csv_out, "")
                messagebox.showerror("Error", error)
            else:
                set_text(csv_out, summary)

        threading.Thread(target=worker, daemon=True).start()

    run_btn.config(command=on_predict_batch)
    run_btn.pack(pady=10)
    csv_out.pack(padx=10, pady=5, fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":
    run_gui()
