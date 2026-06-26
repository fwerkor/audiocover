from __future__ import annotations

import queue
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from audiocover.config import RenderConfig, TrainingConfig, default_config_path
from audiocover.pipeline import render_cover
from audiocover.training import train_model


def _load_tk() -> None:
    global BooleanVar, ScrolledText, StringVar, Tk, filedialog, messagebox, ttk

    from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText


class Worker:
    def __init__(self, log_queue: queue.Queue[str]) -> None:
        self.log_queue = log_queue

    def run(self, title: str, target, *args, **kwargs) -> None:
        def wrapper() -> None:
            self.log_queue.put(f"[{title}] started")
            try:
                result = target(*args, **kwargs)
                self.log_queue.put(f"[{title}] finished")
                self.log_queue.put(str(result))
            except Exception:
                self.log_queue.put(f"[{title}] failed")
                self.log_queue.put(traceback.format_exc())

        threading.Thread(target=wrapper, daemon=True).start()


class AudioCoverGui:
    def __init__(self) -> None:
        _load_tk()
        self.root = Tk()
        self.root.title("AudioCover")
        self.root.geometry("920x720")
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker = Worker(self.log_queue)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        self.train_frame = ttk.Frame(notebook, padding=10)
        self.render_frame = ttk.Frame(notebook, padding=10)
        self.log_frame = ttk.Frame(notebook, padding=10)
        notebook.add(self.train_frame, text="Train")
        notebook.add(self.render_frame, text="Generate cover")
        notebook.add(self.log_frame, text="Logs")

        self._build_train_tab()
        self._build_render_tab()
        self._build_log_tab()
        self.root.after(200, self._poll_logs)

    def _entry_row(self, parent, row: int, label: str, var: StringVar, browse_cmd) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=var, width=82).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", command=browse_cmd).grid(row=row, column=2, padx=4, pady=4)
        parent.columnconfigure(1, weight=1)

    def _choose_dir(self, var: StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _choose_file(self, var: StringVar, filetypes) -> None:
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _build_train_tab(self) -> None:
        self.train_data = StringVar()
        self.train_out = StringVar(value=str(Path.cwd() / "models" / "my_profile"))
        self.train_name = StringVar(value="my_profile")
        self.train_consent = BooleanVar(value=False)

        self._entry_row(self.train_frame, 0, "Data folder", self.train_data, lambda: self._choose_dir(self.train_data))
        self._entry_row(self.train_frame, 1, "Output model folder", self.train_out, lambda: self._choose_dir(self.train_out))
        ttk.Label(self.train_frame, text="Display name").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(self.train_frame, textvariable=self.train_name).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Checkbutton(
            self.train_frame,
            text="I own or am authorized to use these recordings.",
            variable=self.train_consent,
        ).grid(row=3, column=1, sticky="w", pady=8)
        ttk.Button(self.train_frame, text="Start training", command=self._start_training).grid(row=4, column=1, sticky="e", pady=10)

        data_note = (
            "Dataset format: choose a folder containing authorized .wav, .flac, .mp3, .m4a, .aac, "
            "or .ogg files. Use one target speaker, dry isolated voice, stable microphone placement, "
            "minimal room echo, no background music, and no clipping. 48 kHz WAV is preferred; more "
            "clean pitch/style coverage improves the final voice profile."
        )
        ttk.Label(self.train_frame, text=data_note, wraplength=780, justify="left").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=10
        )

        backend_note = (
            "Backend policy: AudioCover selects a packaged backend runtime automatically. Backend workers "
            "run in isolated processes and communicate with the desktop app through a local JSON protocol. "
            "The GUI does not require a separate Python or backend setup."
        )
        ttk.Label(self.train_frame, text=backend_note, wraplength=780, justify="left").grid(
            row=6, column=0, columnspan=3, sticky="w", pady=10
        )

    def _build_render_tab(self) -> None:
        self.song_path = StringVar()
        self.model_yaml = StringVar()
        self.render_out = StringVar(value=str(Path.cwd() / "runs" / "cover"))
        self.render_consent = BooleanVar(value=False)

        audio_types = [("Audio files", "*.wav *.flac *.mp3 *.m4a *.aac *.ogg"), ("All files", "*.*")]
        self._entry_row(self.render_frame, 0, "Song file", self.song_path, lambda: self._choose_file(self.song_path, audio_types))
        self._entry_row(self.render_frame, 1, "Model package model.yaml", self.model_yaml, lambda: self._choose_file(self.model_yaml, [("YAML", "*.yaml *.yml"), ("All files", "*.*")]))
        self._entry_row(self.render_frame, 2, "Output folder", self.render_out, lambda: self._choose_dir(self.render_out))
        ttk.Checkbutton(
            self.render_frame,
            text="I have rights/permission to use this song and model.",
            variable=self.render_consent,
        ).grid(row=3, column=1, sticky="w", pady=8)
        ttk.Button(self.render_frame, text="Generate cover", command=self._start_render).grid(row=4, column=1, sticky="e", pady=10)

        render_note = (
            "Rendering policy: AudioCover uses the built-in preset, selects packaged backend runtimes "
            "automatically, applies the standard mix/QC chain, and avoids overwriting an existing run by "
            "creating a timestamped folder when needed."
        )
        ttk.Label(self.render_frame, text=render_note, wraplength=780, justify="left").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=10
        )

    def _build_log_tab(self) -> None:
        self.logs = ScrolledText(self.log_frame, wrap="word")
        self.logs.pack(fill="both", expand=True)
        ttk.Button(self.log_frame, text="Clear", command=lambda: self.logs.delete("1.0", "end")).pack(anchor="e", pady=4)

    def _log_with_scope(self, scope: str, message: str) -> None:
        prefix = f"[{scope}] "
        if message.startswith("\r"):
            self.log_queue.put("\r" + prefix + message[1:])
        else:
            self.log_queue.put(prefix + message)

    def _start_training(self) -> None:
        if not self.train_consent.get():
            messagebox.showerror("Consent required", "Confirm that you own or are authorized to use the recordings.")
            return
        cfg = TrainingConfig()
        self.worker.run(
            "training",
            train_model,
            Path(self.train_data.get()),
            Path(self.train_out.get()),
            display_name=self.train_name.get(),
            config=cfg,
            consent=True,
            log=lambda message: self._log_with_scope("training", message),
        )

    def _dedicated_output_dir(self, requested: Path) -> Path:
        if not requested.exists() or not any(requested.iterdir()):
            return requested
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return requested.parent / f"{requested.name}-{stamp}"

    def _start_render(self) -> None:
        if not self.render_consent.get():
            messagebox.showerror("Consent required", "Confirm that you have permission to use the song and model.")
            return
        cfg = RenderConfig.from_yaml(default_config_path())
        out_dir = self._dedicated_output_dir(Path(self.render_out.get()))
        if out_dir != Path(self.render_out.get()):
            self.log_queue.put(f"[render] output folder exists; using {out_dir}")
        self.worker.run(
            "render",
            render_cover,
            Path(self.song_path.get()),
            Path(self.model_yaml.get()),
            out_dir,
            config=cfg,
            consent=True,
            log=lambda message: self._log_with_scope("render", message),
        )

    def _poll_logs(self) -> None:
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if msg.startswith("\r"):
                msg = msg[1:]
            self.logs.insert("end", msg + "\n")
            self.logs.see("end")
        self.root.after(200, self._poll_logs)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if "--smoke-test" in sys.argv:
        return
    AudioCoverGui().run()


if __name__ == "__main__":
    main()
