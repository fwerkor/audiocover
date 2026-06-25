from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import RenderConfig, TrainingConfig, default_config_path
from .pipeline import render_cover
from .training import train_model


class Worker:
    def __init__(self, log_queue: "queue.Queue[str]") -> None:
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
        self.root = Tk()
        self.root.title("AudioCover Lab")
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
        self.train_backend = StringVar(value="simple-timbre")
        self.train_segment = StringVar(value="12")
        self.train_epochs = StringVar(value="200")
        self.train_batch = StringVar(value="8")
        self.train_consent = BooleanVar(value=False)

        self._entry_row(self.train_frame, 0, "Data folder", self.train_data, lambda: self._choose_dir(self.train_data))
        self._entry_row(self.train_frame, 1, "Output model folder", self.train_out, lambda: self._choose_dir(self.train_out))
        ttk.Label(self.train_frame, text="Display name").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(self.train_frame, textvariable=self.train_name).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(self.train_frame, text="Backend").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Combobox(self.train_frame, textvariable=self.train_backend, values=["simple-timbre", "external"], state="readonly").grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(self.train_frame, text="Segment seconds").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(self.train_frame, textvariable=self.train_segment).grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Label(self.train_frame, text="Epochs").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(self.train_frame, textvariable=self.train_epochs).grid(row=5, column=1, sticky="ew", pady=4)
        ttk.Label(self.train_frame, text="Batch size").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(self.train_frame, textvariable=self.train_batch).grid(row=6, column=1, sticky="ew", pady=4)
        ttk.Checkbutton(
            self.train_frame,
            text="I own or am authorized to use these recordings.",
            variable=self.train_consent,
        ).grid(row=7, column=1, sticky="w", pady=8)
        ttk.Button(self.train_frame, text="Start training", command=self._start_training).grid(row=8, column=1, sticky="e", pady=10)

        note = (
            "For best quality, use external backend commands to call RVC/Seed-VC/So-VITS training. "
            "The built-in simple-timbre backend is a local fallback and CI-testable model package."
        )
        ttk.Label(self.train_frame, text=note, wraplength=780).grid(row=9, column=0, columnspan=3, sticky="w", pady=10)

    def _build_render_tab(self) -> None:
        self.song_path = StringVar()
        self.model_yaml = StringVar()
        self.render_out = StringVar(value=str(Path.cwd() / "runs" / "cover"))
        self.config_path = StringVar(value=str(default_config_path()))
        self.render_overwrite = BooleanVar(value=True)
        self.render_consent = BooleanVar(value=False)

        audio_types = [("Audio files", "*.wav *.flac *.mp3 *.m4a *.aac *.ogg"), ("All files", "*.*")]
        self._entry_row(self.render_frame, 0, "Song file", self.song_path, lambda: self._choose_file(self.song_path, audio_types))
        self._entry_row(self.render_frame, 1, "Model package model.yaml", self.model_yaml, lambda: self._choose_file(self.model_yaml, [("YAML", "*.yaml *.yml"), ("All files", "*.*")]))
        self._entry_row(self.render_frame, 2, "Output folder", self.render_out, lambda: self._choose_dir(self.render_out))
        self._entry_row(self.render_frame, 3, "Render config", self.config_path, lambda: self._choose_file(self.config_path, [("YAML", "*.yaml *.yml"), ("All files", "*.*")]))
        ttk.Checkbutton(self.render_frame, text="Overwrite output folder if needed", variable=self.render_overwrite).grid(row=4, column=1, sticky="w", pady=4)
        ttk.Checkbutton(
            self.render_frame,
            text="I have rights/permission to use this song and model.",
            variable=self.render_consent,
        ).grid(row=5, column=1, sticky="w", pady=8)
        ttk.Button(self.render_frame, text="Generate cover", command=self._start_render).grid(row=6, column=1, sticky="e", pady=10)

    def _build_log_tab(self) -> None:
        self.logs = ScrolledText(self.log_frame, wrap="word")
        self.logs.pack(fill="both", expand=True)
        ttk.Button(self.log_frame, text="Clear", command=lambda: self.logs.delete("1.0", "end")).pack(anchor="e", pady=4)

    def _start_training(self) -> None:
        if not self.train_consent.get():
            messagebox.showerror("Consent required", "Confirm that you own or are authorized to use the recordings.")
            return
        cfg = TrainingConfig(
            backend=self.train_backend.get(),
            segment_seconds=float(self.train_segment.get()),
            epochs=int(self.train_epochs.get()),
            batch_size=int(self.train_batch.get()),
        )
        self.worker.run(
            "training",
            train_model,
            Path(self.train_data.get()),
            Path(self.train_out.get()),
            display_name=self.train_name.get(),
            config=cfg,
            consent=True,
        )

    def _start_render(self) -> None:
        if not self.render_consent.get():
            messagebox.showerror("Consent required", "Confirm that you have permission to use the song and model.")
            return
        cfg = RenderConfig.from_yaml(Path(self.config_path.get()))
        cfg.overwrite = self.render_overwrite.get()
        self.worker.run(
            "render",
            render_cover,
            Path(self.song_path.get()),
            Path(self.model_yaml.get()),
            Path(self.render_out.get()),
            config=cfg,
            consent=True,
        )

    def _poll_logs(self) -> None:
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.logs.insert("end", msg + "\n")
            self.logs.see("end")
        self.root.after(200, self._poll_logs)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    AudioCoverGui().run()


if __name__ == "__main__":
    main()
