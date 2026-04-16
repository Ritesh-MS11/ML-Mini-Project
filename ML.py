import os
import re
import threading
import warnings
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

import easyocr
from PIL import Image

warnings.filterwarnings("ignore")

# ── Colour palette ──────────────────────────────────────────
BG        = "#0f0f1a"
CARD      = "#1a1a2e"
ACCENT    = "#7c3aed"
ACCENT2   = "#06b6d4"
REAL_CLR  = "#10b981"
FAKE_CLR  = "#ef4444"
TEXT      = "#e2e8f0"
SUBTEXT   = "#94a3b8"
BORDER    = "#2d2d4e"
WHITE     = "#ffffff"

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_HEAD   = ("Segoe UI", 13, "bold")
FONT_BODY   = ("Segoe UI", 11)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 10)

DATASET_PATH = "WELFake_Dataset.csv"


# ═══════════════════════════════════════════════════════════
#  TEXT CLEANING
# ═══════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════
#  MODEL TRAINING  (runs in background thread)
# ═══════════════════════════════════════════════════════════

class ModelTrainer:
    def __init__(self):
        self.tfidf  = None
        self.model  = None
        self.metrics = {}
        self.cm      = None
        self.bow_acc = None
        self.tfidf_acc = None
        self.feature_names = None
        self.log_prob_diff  = None
        self.label_names    = {0: "Real", 1: "Fake"}
        self.ready = False

    def train(self, progress_cb):
        progress_cb("Loading dataset…", 5)
        df = pd.read_csv(DATASET_PATH)

        progress_cb("Combining title + body…", 15)
        df['text_combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

        progress_cb("Cleaning text…", 25)
        df['clean_text'] = df['text_combined'].apply(clean_text)

        X = df['clean_text']
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── BoW baseline ─────────────────────────────────
        progress_cb("Training Bag-of-Words baseline…", 40)
        bow_pipe = Pipeline([
            ('vec', CountVectorizer(max_features=50000, stop_words='english')),
            ('clf', MultinomialNB())
        ])
        bow_pipe.fit(X_train, y_train)
        self.bow_acc = accuracy_score(y_test, bow_pipe.predict(X_test)) * 100

        # ── TF-IDF + Naive Bayes ─────────────────────────
        progress_cb("Training TF-IDF + Naïve Bayes model…", 65)
        self.tfidf = TfidfVectorizer(
            max_features=50000,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.model = MultinomialNB(alpha=0.1)
        X_tr_vec = self.tfidf.fit_transform(X_train)
        X_te_vec = self.tfidf.transform(X_test)
        self.model.fit(X_tr_vec, y_train)
        y_pred = self.model.predict(X_te_vec)

        # ── Metrics ───────────────────────────────────────
        progress_cb("Computing evaluation metrics…", 80)
        self.tfidf_acc = accuracy_score(y_test, y_pred) * 100
        self.metrics = {
            "Accuracy" : accuracy_score(y_test, y_pred)  * 100,
            "Precision": precision_score(y_test, y_pred) * 100,
            "Recall"   : recall_score(y_test, y_pred)    * 100,
            "F1 Score" : f1_score(y_test, y_pred)        * 100,
        }
        self.cm = confusion_matrix(y_test, y_pred)

        # ── Feature importance ────────────────────────────
        self.feature_names = np.array(self.tfidf.get_feature_names_out())
        self.log_prob_diff = self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0]

        progress_cb("Ready!", 100)
        self.ready = True

    def predict(self, text: str):
        cleaned = clean_text(text)
        vec     = self.tfidf.transform([cleaned])
        pred    = self.model.predict(vec)[0]
        prob    = self.model.predict_proba(vec)[0]
        return self.label_names[pred], prob[0] * 100, prob[1] * 100


# ═══════════════════════════════════════════════════════════
#  MAIN GUI
# ═══════════════════════════════════════════════════════════

class FakeNewsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fake News Detector  |  ML Mini Project")
        self.geometry("1200x780")
        self.minsize(1000, 680)
        self.configure(bg=BG)

        self.trainer = ModelTrainer()
        self.ocr_reader = None
        self._build_ui()
        self._start_training()

    # ── UI BUILD ────────────────────────────────────────────

    def _build_ui(self):
        # ── Header bar ──────────────────────────────────────
        hdr = tk.Frame(self, bg=ACCENT, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🛡  Fake News Detector",
                 font=FONT_TITLE, bg=ACCENT, fg=WHITE).pack(side="left", padx=24)
        self.status_lbl = tk.Label(hdr, text="⏳  Training model…",
                                   font=FONT_BODY, bg=ACCENT, fg="#ddd6fe")
        self.status_lbl.pack(side="right", padx=24)

        # ── Progress bar ────────────────────────────────────
        self.prog_frame = tk.Frame(self, bg=BG)
        self.prog_frame.pack(fill="x", padx=0, pady=0)
        self.progress = ttk.Progressbar(self.prog_frame, mode="determinate",
                                        maximum=100, length=400)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=CARD,
                         background=ACCENT2, thickness=6)
        self.progress.pack(fill="x")

        # ── Notebook tabs ────────────────────────────────────
        nb_style = ttk.Style()
        nb_style.configure("Dark.TNotebook",         background=BG, borderwidth=0)
        nb_style.configure("Dark.TNotebook.Tab",     background=CARD, foreground=SUBTEXT,
                            padding=[18, 8], font=FONT_BODY)
        nb_style.map("Dark.TNotebook.Tab",
                     background=[("selected", ACCENT)],
                     foreground=[("selected", WHITE)])

        self.nb = ttk.Notebook(self, style="Dark.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=0, pady=0)

        self.tab_predict = tk.Frame(self.nb, bg=BG)
        self.tab_metrics = tk.Frame(self.nb, bg=BG)
        self.tab_features= tk.Frame(self.nb, bg=BG)

        self.nb.add(self.tab_predict,  text="  🔍  Predict  ")
        self.nb.add(self.tab_metrics,  text="  📊  Model Metrics  ")
        self.nb.add(self.tab_features, text="  🧠  Feature Insights  ")

        self._build_predict_tab()
        self._build_metrics_tab()
        self._build_features_tab()

    # ── TAB 1 : PREDICT ─────────────────────────────────────

    def _build_predict_tab(self):
        tab = self.tab_predict
        pad = {"padx": 20, "pady": 10}

        tk.Label(tab, text="Paste a news article below and click Analyze",
                 font=FONT_HEAD, bg=BG, fg=SUBTEXT).pack(anchor="w", **pad)

        # text input area
        input_frame = tk.Frame(tab, bg=BORDER, bd=1)
        input_frame.pack(fill="both", expand=True, padx=20, pady=(0, 6))

        self.article_box = scrolledtext.ScrolledText(
            input_frame, font=FONT_MONO, bg=CARD, fg=TEXT,
            insertbackground=WHITE, relief="flat",
            wrap="word", height=12,
            padx=14, pady=12,
            selectbackground=ACCENT
        )
        self.article_box.pack(fill="both", expand=True)
        self.article_box.insert("end", "Paste your article here…")
        self.article_box.bind("<FocusIn>",  self._clear_placeholder)
        self.article_box.bind("<FocusOut>", self._restore_placeholder)

        # buttons row
        btn_row = tk.Frame(tab, bg=BG)
        btn_row.pack(fill="x", padx=20, pady=6)

        self.analyze_btn = tk.Button(
            btn_row, text="  🔍  Analyze Article",
            font=FONT_HEAD, bg=ACCENT, fg=WHITE,
            activebackground="#6d28d9", activeforeground=WHITE,
            relief="flat", padx=20, pady=10, cursor="hand2",
            command=self._analyze, state="disabled"
        )
        self.analyze_btn.pack(side="left")

        self.scan_btn = tk.Button(
            btn_row, text="  📷  Scan Image",
            font=FONT_HEAD, bg=ACCENT2, fg=WHITE,
            activebackground="#0891b2", activeforeground=WHITE,
            relief="flat", padx=20, pady=10, cursor="hand2",
            command=self._scan_image, state="disabled"
        )
        self.scan_btn.pack(side="left", padx=10)

        tk.Button(
            btn_row, text="  ✕  Clear",
            font=FONT_BODY, bg=CARD, fg=SUBTEXT,
            activebackground=BORDER, activeforeground=TEXT,
            relief="flat", padx=14, pady=10, cursor="hand2",
            command=self._clear_input
        ).pack(side="left")

        # result card
        self.result_frame = tk.Frame(tab, bg=CARD, bd=0, relief="flat")
        self.result_frame.pack(fill="x", padx=20, pady=(4, 16))

        self.verdict_icon = tk.Label(self.result_frame, text="", font=("Segoe UI", 36),
                                     bg=CARD, fg=TEXT)
        self.verdict_icon.pack(side="left", padx=20, pady=16)

        info = tk.Frame(self.result_frame, bg=CARD)
        info.pack(side="left", fill="both", expand=True, pady=12)

        self.verdict_lbl = tk.Label(info, text="Awaiting input…",
                                    font=("Segoe UI", 18, "bold"), bg=CARD, fg=SUBTEXT)
        self.verdict_lbl.pack(anchor="w")

        self.conf_lbl = tk.Label(info, text="", font=FONT_BODY, bg=CARD, fg=SUBTEXT)
        self.conf_lbl.pack(anchor="w", pady=(2, 6))

        # confidence bar (canvas)
        bar_outer = tk.Frame(info, bg=BORDER, height=10)
        bar_outer.pack(fill="x", pady=(0, 4))
        bar_outer.pack_propagate(False)
        self.conf_bar = tk.Frame(bar_outer, bg=ACCENT2, height=10)
        self.conf_bar.place(x=0, y=0, relheight=1, relwidth=0)

        self.breakdown_lbl = tk.Label(info, text="", font=FONT_SMALL, bg=CARD, fg=SUBTEXT)
        self.breakdown_lbl.pack(anchor="w")

    def _clear_placeholder(self, event):
        if self.article_box.get("1.0", "end-1c") == "Paste your article here…":
            self.article_box.delete("1.0", "end")
            self.article_box.config(fg=TEXT)

    def _restore_placeholder(self, event):
        if not self.article_box.get("1.0", "end-1c").strip():
            self.article_box.insert("1.0", "Paste your article here…")
            self.article_box.config(fg=SUBTEXT)

    def _clear_input(self):
        self.article_box.delete("1.0", "end")
        self.article_box.insert("1.0", "Paste your article here…")
        self.article_box.config(fg=SUBTEXT)
        self.verdict_lbl.config(text="Awaiting input…", fg=SUBTEXT)
        self.verdict_icon.config(text="")
        self.conf_lbl.config(text="")
        self.breakdown_lbl.config(text="")
        self.conf_bar.place(relwidth=0)
        self.result_frame.config(bg=CARD)
        self.verdict_icon.config(bg=CARD)
        for w in self.result_frame.winfo_children():
            w.config(bg=CARD)

    def _analyze(self):
        try:
            print("[DEBUG] Analyze button clicked")
            text = self.article_box.get("1.0", "end-1c").strip()
            if not text or text == "Paste your article here…":
                messagebox.showwarning("Empty Input", "Please paste an article first.")
                return
            if len(text.split()) < 10:
                messagebox.showwarning("Too Short", "Please paste a longer article (at least 10 words).")
                return

            print(f"[DEBUG] Predicting for text length: {len(text)}")
            label, real_pct, fake_pct = self.trainer.predict(text)
            conf = max(real_pct, fake_pct)
            print(f"[DEBUG] Prediction: {label} ({conf:.1f}%)")

            if label == "Fake":
                clr  = FAKE_CLR
                icon = "❌"
                msg  = "FAKE NEWS DETECTED"
            else:
                clr  = REAL_CLR
                icon = "✅"
                msg  = "REAL NEWS"

            # Use solid colors to ensure compatibility with all Tkinter versions
            self.result_frame.config(bg=CARD)
            for w in self.result_frame.winfo_children():
                w.config(bg=CARD)

            self.verdict_icon.config(text=icon, fg=clr)
            self.verdict_lbl.config(text=msg, fg=clr)
            self.conf_lbl.config(
                text=f"Confidence: {conf:.1f}%",
                fg=TEXT
            )
            self.conf_bar.place(relwidth=conf / 100)
            self.conf_bar.config(bg=clr)
            self.breakdown_lbl.config(
                text=f"Real probability: {real_pct:.1f}%    |    Fake probability: {fake_pct:.1f}%",
                fg=SUBTEXT
            )
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            messagebox.showerror("Analysis Error", f"An unexpected error occurred:\n{str(e)}")

    # ── OCR SCANNING ──────────────────────────────────────────

    def _scan_image(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select Article Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
            if not file_path:
                return

            self.status_lbl.config(text="⏳  Initializing OCR engine…")
            self.update_idletasks()

            def ocr_worker():
                try:
                    if self.ocr_reader is None:
                        # gpu=False is used for better compatibility on various machines
                        self.ocr_reader = easyocr.Reader(['en'], gpu=False)

                    self.after(0, lambda: self.status_lbl.config(text="⏳  Scanning image contents…"))

                    # Read text from the image
                    results = self.ocr_reader.readtext(file_path, detail=0)
                    full_text = " ".join(results)

                    if not full_text:
                        self.after(0, lambda: messagebox.showwarning("No Text Found", "Could not extract any text from the image."))
                        return

                    self.after(0, lambda: self._populate_ocr_text(full_text))
                except Exception as e:
                    print(f"[ERROR] OCR failed in thread: {e}")
                    self.after(0, lambda: messagebox.showerror("OCR Error", f"An error occurred during scanning:\n{str(e)}"))
                finally:
                    status = "✅  Model ready" if self.trainer.ready else "⏳  Training model…"
                    self.after(0, lambda: self.status_lbl.config(text=status))

            threading.Thread(target=ocr_worker, daemon=True).start()
        except Exception as e:
            print(f"[ERROR] Scan image setup failed: {e}")
            messagebox.showerror("Error", f"Failed to start scanner:\n{str(e)}")

    def _populate_ocr_text(self, text):
        self.article_box.delete("1.0", "end")
        self.article_box.insert("1.0", text)
        self.article_box.config(fg=TEXT)
        self._analyze()

    # ── TAB 2 : METRICS ─────────────────────────────────────

    def _build_metrics_tab(self):
        tab = self.tab_metrics
        self.metrics_placeholder = tk.Label(
            tab, text="⏳  Training model — metrics will appear here shortly…",
            font=FONT_HEAD, bg=BG, fg=SUBTEXT
        )
        self.metrics_placeholder.pack(expand=True)

    def _populate_metrics_tab(self):
        self.metrics_placeholder.destroy()
        tab = self.tab_metrics

        fig = Figure(figsize=(11, 4.5), facecolor=BG)

        # ── Confusion Matrix ─────────────────────────────
        ax1 = fig.add_subplot(131)
        cm  = self.trainer.cm
        im  = ax1.imshow(cm, cmap="Blues")
        ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
        ax1.set_xticklabels(["Real", "Fake"], color=TEXT)
        ax1.set_yticklabels(["Real", "Fake"], color=TEXT)
        ax1.set_xlabel("Predicted", color=SUBTEXT)
        ax1.set_ylabel("Actual",    color=SUBTEXT)
        ax1.set_title("Confusion Matrix", color=TEXT, fontsize=12, pad=10)
        ax1.set_facecolor(CARD)
        for (r, c), val in np.ndenumerate(cm):
            ax1.text(c, r, f"{val:,}", ha="center", va="center",
                     color=WHITE, fontsize=11, fontweight="bold")
        ax1.tick_params(colors=SUBTEXT)
        for spine in ax1.spines.values():
            spine.set_edgecolor(BORDER)

        # ── Metrics bar ──────────────────────────────────
        ax2  = fig.add_subplot(132)
        keys = list(self.trainer.metrics.keys())
        vals = list(self.trainer.metrics.values())
        clrs = [ACCENT2, REAL_CLR, "#f59e0b", ACCENT]
        bars = ax2.bar(keys, vals, color=clrs, edgecolor=BG, width=0.5)
        ax2.set_ylim(80, 100)
        ax2.set_facecolor(CARD)
        ax2.set_title("Performance Metrics (%)", color=TEXT, fontsize=12, pad=10)
        ax2.tick_params(colors=SUBTEXT)
        for spine in ax2.spines.values():
            spine.set_edgecolor(BORDER)
        ax2.yaxis.label.set_color(SUBTEXT)
        ax2.xaxis.label.set_color(SUBTEXT)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.15,
                     f"{v:.2f}%", ha="center", color=TEXT,
                     fontsize=9, fontweight="bold")

        # ── BoW vs TF-IDF ────────────────────────────────
        ax3  = fig.add_subplot(133)
        methods = ["Bag-of-Words", "TF-IDF"]
        accs    = [self.trainer.bow_acc, self.trainer.tfidf_acc]
        clrs2   = [ACCENT, ACCENT2]
        bars2   = ax3.bar(methods, accs, color=clrs2, edgecolor=BG, width=0.4)
        ax3.set_ylim(80, 100)
        ax3.set_facecolor(CARD)
        ax3.set_title("BoW vs TF-IDF Accuracy", color=TEXT, fontsize=12, pad=10)
        ax3.tick_params(colors=SUBTEXT)
        for spine in ax3.spines.values():
            spine.set_edgecolor(BORDER)
        for bar, v in zip(bars2, accs):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.15,
                     f"{v:.2f}%", ha="center", color=TEXT,
                     fontsize=10, fontweight="bold")

        fig.tight_layout(pad=2.5)

        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # ── TAB 3 : FEATURES ────────────────────────────────────

    def _build_features_tab(self):
        tab = self.tab_features
        self.features_placeholder = tk.Label(
            tab, text="⏳  Training model — feature insights will appear here shortly…",
            font=FONT_HEAD, bg=BG, fg=SUBTEXT
        )
        self.features_placeholder.pack(expand=True)

    def _populate_features_tab(self):
        self.features_placeholder.destroy()
        tab = self.tab_features

        fig = Figure(figsize=(11, 4.8), facecolor=BG)

        diff = self.trainer.log_prob_diff
        feats = self.trainer.feature_names

        top_fake_idx = np.argsort(diff)[-15:][::-1]
        top_real_idx = np.argsort(diff)[:15]

        ax1 = fig.add_subplot(121)
        ax1.barh(feats[top_fake_idx][::-1], diff[top_fake_idx][::-1],
                 color=FAKE_CLR, edgecolor=BG)
        ax1.set_title("Top 15 Words → Fake News", color=TEXT, fontsize=12, pad=10)
        ax1.set_xlabel("Log-Prob Difference", color=SUBTEXT)
        ax1.set_facecolor(CARD)
        ax1.tick_params(colors=TEXT)
        for spine in ax1.spines.values():
            spine.set_edgecolor(BORDER)

        ax2 = fig.add_subplot(122)
        ax2.barh(feats[top_real_idx][::-1], np.abs(diff[top_real_idx][::-1]),
                 color=REAL_CLR, edgecolor=BG)
        ax2.set_title("Top 15 Words → Real News", color=TEXT, fontsize=12, pad=10)
        ax2.set_xlabel("Log-Prob Difference", color=SUBTEXT)
        ax2.set_facecolor(CARD)
        ax2.tick_params(colors=TEXT)
        for spine in ax2.spines.values():
            spine.set_edgecolor(BORDER)

        fig.tight_layout(pad=2.5)

        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # ── TRAINING THREAD ─────────────────────────────────────

    def _start_training(self):
        if not os.path.exists(DATASET_PATH):
            messagebox.showerror(
                "Dataset Not Found",
                f"'{DATASET_PATH}' not found.\n\n"
                "Download from:\n"
                "https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification\n\n"
                "Place it in the same folder as this script."
            )
            self.destroy()
            return
        t = threading.Thread(target=self._train_worker, daemon=True)
        t.start()

    def _train_worker(self):
        def update(msg, pct):
            self.after(0, lambda: self.status_lbl.config(text=f"⏳  {msg}"))
            self.after(0, lambda: self.progress.config(value=pct))
        try:
            self.trainer.train(update)
            self.after(0, self._on_training_done)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Training Error", str(e)))

    def _on_training_done(self):
        self.status_lbl.config(text="✅  Model ready")
        self.progress.config(value=100)
        self.prog_frame.after(800, self.prog_frame.pack_forget)
        self.analyze_btn.config(state="normal")
        self.scan_btn.config(state="normal")
        self._populate_metrics_tab()
        self._populate_features_tab()


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = FakeNewsApp()
    app.mainloop()