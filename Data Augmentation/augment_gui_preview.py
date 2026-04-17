#!/usr/bin/env python3
"""Preview of the Data Augmentation GUI layout (no backend logic)."""

import tkinter as tk
from tkinter import ttk, filedialog


class AugmentGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Data Augmentation Tool")
        self.geometry("720x700")
        self.minsize(650, 600)
        self.configure(bg="#1e1e2e")

        style = ttk.Style(self)
        style.theme_use("clam")

        # Colors
        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#89b4fa"
        surface = "#313244"
        green = "#a6e3a1"
        red = "#f38ba8"
        subtext = "#a6adc8"

        style.configure(".", background=bg, foreground=fg, fieldbackground=surface,
                         borderwidth=0, font=("Helvetica", 12))
        style.configure("TLabel", background=bg, foreground=fg, font=("Helvetica", 12))
        style.configure("Header.TLabel", background=bg, foreground=fg, font=("Helvetica", 14, "bold"))
        style.configure("Sub.TLabel", background=bg, foreground=subtext, font=("Helvetica", 10))
        style.configure("TEntry", fieldbackground=surface, foreground=fg, insertcolor=fg)
        style.configure("TButton", background=surface, foreground=fg, padding=(12, 6),
                         font=("Helvetica", 11))
        style.map("TButton", background=[("active", accent)], foreground=[("active", bg)])
        style.configure("Accent.TButton", background=accent, foreground=bg, padding=(16, 8),
                         font=("Helvetica", 12, "bold"))
        style.map("Accent.TButton", background=[("active", green)])
        style.configure("Remove.TButton", background=red, foreground=bg, padding=(6, 2),
                         font=("Helvetica", 10))
        style.map("Remove.TButton", background=[("active", "#eb6f92")])
        style.configure("TLabelframe", background=bg, foreground=accent, font=("Helvetica", 12, "bold"))
        style.configure("TLabelframe.Label", background=bg, foreground=accent)
        style.configure("green.Horizontal.TProgressbar", troughcolor=surface, background=green)

        pad = {"padx": 12, "pady": 4}

        # === Directories ===
        dir_frame = ttk.LabelFrame(self, text="  Directories  ", padding=10)
        dir_frame.pack(fill="x", padx=16, pady=(16, 8))

        ttk.Label(dir_frame, text="Input Directory").grid(row=0, column=0, sticky="w", **pad)
        self.input_var = tk.StringVar(value="/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_original")
        ttk.Entry(dir_frame, textvariable=self.input_var, width=48).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(dir_frame, text="Browse", command=lambda: self._browse_dir(self.input_var)).grid(row=0, column=2, **pad)

        ttk.Label(dir_frame, text="Output Directory").grid(row=1, column=0, sticky="w", **pad)
        self.output_var = tk.StringVar(value="/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_augmented")
        ttk.Entry(dir_frame, textvariable=self.output_var, width=48).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(dir_frame, text="Browse", command=lambda: self._browse_dir(self.output_var)).grid(row=1, column=2, **pad)

        dir_frame.columnconfigure(1, weight=1)

        # === Noise Sources ===
        noise_frame = ttk.LabelFrame(self, text="  Noise Sources  ", padding=10)
        noise_frame.pack(fill="both", expand=True, padx=16, pady=8)

        # Noise list
        self.noise_list = tk.Listbox(noise_frame, bg=surface, fg=fg, selectbackground=accent,
                                      selectforeground=bg, font=("Helvetica", 11), height=5,
                                      borderwidth=0, highlightthickness=1, highlightcolor=accent)
        self.noise_list.pack(fill="both", expand=True, padx=4, pady=4)

        # Pre-populate with demo entries
        self.noise_list.insert(tk.END, "  White Noise  (generated)")
        self.noise_list.insert(tk.END, "  Crowd Noise  — crowd noise.wav")
        self.noise_list.insert(tk.END, "  Street Noise — street noise.wav")

        btn_row = ttk.Frame(noise_frame)
        btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_row, text="+ White Noise", command=self._add_white).pack(side="left", padx=4)
        ttk.Button(btn_row, text="+ Noise File...", command=self._add_file_noise).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Remove Selected", style="Remove.TButton",
                    command=self._remove_noise).pack(side="right", padx=4)

        # === Parameters ===
        param_frame = ttk.LabelFrame(self, text="  Parameters  ", padding=10)
        param_frame.pack(fill="x", padx=16, pady=8)

        ttk.Label(param_frame, text="SNR Levels (dB)").grid(row=0, column=0, sticky="w", **pad)
        self.snr_var = tk.StringVar(value="20, 10, 0")
        ttk.Entry(param_frame, textvariable=self.snr_var, width=24).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(param_frame, text="comma-separated", style="Sub.TLabel").grid(row=0, column=2, sticky="w", **pad)

        ttk.Label(param_frame, text="Snippet Duration (s)").grid(row=1, column=0, sticky="w", **pad)
        self.snippet_var = tk.StringVar(value="30")
        ttk.Entry(param_frame, textvariable=self.snippet_var, width=8).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(param_frame, text="for file-based noise", style="Sub.TLabel").grid(row=1, column=2, sticky="w", **pad)

        ttk.Label(param_frame, text="Random Seed").grid(row=2, column=0, sticky="w", **pad)
        self.seed_var = tk.StringVar(value="42")
        ttk.Entry(param_frame, textvariable=self.seed_var, width=8).grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(param_frame, text="leave blank for random", style="Sub.TLabel").grid(row=2, column=2, sticky="w", **pad)

        ttk.Label(param_frame, text="Workers").grid(row=3, column=0, sticky="w", **pad)
        self.workers_var = tk.StringVar(value="4")
        ttk.Entry(param_frame, textvariable=self.workers_var, width=8).grid(row=3, column=1, sticky="w", **pad)

        # === Progress ===
        progress_frame = ttk.LabelFrame(self, text="  Progress  ", padding=10)
        progress_frame.pack(fill="x", padx=16, pady=8)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var, style="Sub.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(progress_frame, style="green.Horizontal.TProgressbar",
                                         length=400, mode="determinate", value=35)
        self.progress.pack(fill="x", pady=(4, 0))

        # Log
        self.log = tk.Text(progress_frame, bg=surface, fg=subtext, font=("Menlo", 10),
                            height=4, borderwidth=0, highlightthickness=0, state="disabled")
        self.log.pack(fill="x", pady=(8, 0))
        self._log("Loaded 3 noise sources, 3 SNR levels")
        self._log("Estimated output: 9,000 files")

        # === Controls ===
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill="x", padx=16, pady=(8, 16))
        ttk.Button(ctrl_frame, text="Start Augmentation", style="Accent.TButton").pack(side="right", padx=4)
        ttk.Button(ctrl_frame, text="Cancel").pack(side="right", padx=4)

    def _browse_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _add_white(self):
        self.noise_list.insert(tk.END, "  White Noise  (generated)")

    def _add_file_noise(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if path:
            import os
            name = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
            self.noise_list.insert(tk.END, f"  {name} — {os.path.basename(path)}")

    def _remove_noise(self):
        sel = self.noise_list.curselection()
        if sel:
            self.noise_list.delete(sel[0])

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")


if __name__ == "__main__":
    app = AugmentGUI()
    app.mainloop()
