"""Small native desktop launcher with independent experiment tabs and PNG preview."""
from datetime import datetime
from dataclasses import asdict
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from .cli import MODULES, config_for
from .paths import OUTPUTS_DIR


class SimulatorApp:
    def __init__(self, root, config=None):
        self.root = root
        self.config = config or {}
        self.messages = queue.Queue()
        self.report_path = None
        self.preview_source = None
        self.root.title("光学模拟实验室 · 衍射 / 泰伯 / 离轴全息")
        self.root.geometry("1380x900")
        self.root.minsize(1100, 740)
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Microsoft YaHei", 17, "bold"))
        style.configure("TLabel", font=("Microsoft YaHei", 10))
        style.configure("TButton", font=("Microsoft YaHei", 10), padding=6)
        shell = ttk.Frame(root, padding=20)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="光学模拟实验室", style="Title.TLabel").pack(anchor="w")
        ttk.Label(shell, text="统一复振幅框架  ·  LightPipes 传播  ·  参数、图表、原始数据独立导出").pack(anchor="w", pady=(5, 18))
        body = ttk.Panedwindow(shell, orient="horizontal")
        body.pack(fill="both", expand=True)
        left, right = ttk.Frame(body, width=450), ttk.Frame(body)
        body.add(left, weight=0)
        body.add(right, weight=1)
        common = ttk.LabelFrame(left, text="共用参数", padding=12)
        common.pack(fill="x", padx=(0, 16))
        shared = self.config.get("shared", {})
        self.shared_vars = {}
        for row, (key, label, default) in enumerate((("n", "每边像素 n（偶数）", shared.get("n", 512)),
                                                    ("wavelength_nm", "波长 λ（nm）", shared.get("wavelength", 632.8e-9)*1e9),
                                                    ("backend", "菲涅耳传播后端", shared.get("backend", "lightpipes")))):
            self.shared_vars[key] = self._input(common, row, label, default, ["lightpipes", "scipy"] if key=="backend" else None)
        self.tabs = ttk.Notebook(left)
        self.tabs.pack(fill="both", expand=True, padx=(0, 16), pady=(14, 8))
        self.local_vars = {}
        self.module_by_tab = {}
        self.defaults = {}
        specifications = {
            "diffraction": [("dx", "采样 dx（μm）", 1e6, None), ("z", "传播距离 z（m）", 1, None),
                            ("width", "孔径宽度 / 直径（μm）", 1e6, None),
                            ("kind", "几何孔径", 1, ["circle", "square", "slit"]),
                            ("method", "传播方法", 1, ["fresnel", "angular", "fraunhofer", "fresnel_one_fft"]),
                            ("cdi_iterations", "CDI 迭代次数（0关闭）", 1, None)],
            "talbot": [("dx", "采样 dx（μm）", 1e6, None), ("period", "光栅周期 d（μm）", 1e6, None),
                       ("duty", "占空比（0至1之间）", 1, None),
                       ("kind", "光栅类型", 1, ["binary", "sine", "phase", "phase_sine"]),
                       ("phase_depth", "相位深度（rad）", 1, None), ("dimensions", "光栅维度", 1, ["1", "2"]),
                       ("finite_fraction", "有效窗口比例（1为周期模型）", 1, None), ("z_steps", "x–z 扫描平面数", 1, None)],
            "holography": [("dx", "采样 dx（μm）", 1e6, None), ("z", "物体到记录面 z（m）", 1, None),
                           ("theta_deg", "参考角 θ（°）", 1, None), ("bandwidth", "物光带宽 B（m⁻¹）", 1, None),
                           ("reference_amplitude", "参考波振幅", 1, None)]}
        self.specifications = specifications
        for name, (klass, run, label) in MODULES.items():
            tab = ttk.Frame(self.tabs, padding=12)
            self.tabs.add(tab, text=label)
            self.module_by_tab[str(tab)] = name
            defaults = asdict(config_for(name, self.config))
            self.defaults[name] = defaults
            self.local_vars[name] = {}
            for row, (key, text, scale, choices) in enumerate(specifications[name]):
                value = defaults[key]
                shown = value*scale if isinstance(value, (int, float)) else value
                self.local_vars[name][key] = self._input(tab, row, text, shown, choices)
            row = len(specifications[name])
            if name != "talbot":
                ttk.Separator(tab).grid(row=row, column=0, columnspan=2, sticky="ew", pady=14)
                self.local_vars[name]["image_path"] = self._input(tab, row+1, "图片（留空用默认物体）", defaults.get("image_path") or "")
                self.local_vars[name]["image_mode"] = self._input(tab, row+2, "图片透过率", defaults["image_mode"], ["amplitude", "phase"])
                ttk.Button(tab, text="选择图片", command=lambda module=name: self.choose_image(module)).grid(row=row+3, column=1, sticky="ew", pady=5)
                row += 4
            help_text = {"diffraction": "圆孔、方孔、单缝及图片物体；解析解与 CDI 闭环自动生成。",
                         "talbot": "要求窗口含整数周期、每周期至少8像素。整距原像，半距横移半周期。",
                         "holography": "默认按已知 B 带限物体。分离条件：3B < |fc| < fNyquist − B。"}[name]
            ttk.Label(tab, text=help_text, wraplength=365, foreground="#52657a").grid(row=row+1, column=0, columnspan=2, sticky="w", pady=12)
        self.run_button = ttk.Button(left, text="计算当前模块并导出", command=self.start)
        self.run_button.pack(fill="x", padx=(0, 16), pady=(8, 0))
        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill="x", padx=(0, 16), pady=8)
        self.status = tk.StringVar(value="选择模块并调整参数后运行。")
        ttk.Label(left, textvariable=self.status, wraplength=405).pack(anchor="w", padx=(0, 16))
        toolbar = ttk.Frame(right)
        toolbar.pack(fill="x")
        self.figure_select = ttk.Combobox(toolbar, state="readonly", width=35)
        self.figure_select.pack(side="left", fill="x", expand=True)
        self.figure_select.bind("<<ComboboxSelected>>", lambda event: self.show_selected())
        ttk.Button(toolbar, text="打开报告", command=self.open_report).pack(side="left", padx=6)
        ttk.Button(toolbar, text="数据目录", command=self.open_directory).pack(side="left")
        self.preview = ttk.Label(right, text="计算后在此预览实验图表", anchor="center", background="#eaf0f6")
        self.preview.pack(fill="both", expand=True, pady=12)
        self.preview.bind("<Configure>", self.resize_preview)
        self.details = tk.Text(right, height=11, wrap="word", font=("Consolas", 10), relief="flat", background="#f5f7fb")
        self.details.pack(fill="x")
        self.details.insert("1.0", "结果依次展示：\n  01_输入图：屏/光栅/透明片的物理示意\n  02_输入复振幅：输入振幅图、相位图\n  随后为传播、成像和验证结果\n\nreport.html：全部图表与指标\ndata.npz：原始复场、坐标和强度")
        self.details.configure(state="disabled")
        self.root.after(150, self.poll)

    def _input(self, frame, row, label, value, choices=None):
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
        text = f"{value:.8g}" if isinstance(value, float) else str(value)
        variable = tk.StringVar(value=text)
        widget = ttk.Combobox(frame, textvariable=variable, values=choices, state="readonly", width=18) if choices else ttk.Entry(frame, textvariable=variable, width=20)
        widget.grid(row=row, column=1, sticky="ew", pady=5)
        return variable

    def choose_image(self, module):
        path = filedialog.askopenfilename(title="选择灰度/彩色图片", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("全部文件", "*.*")])
        if path:
            self.local_vars[module]["image_path"].set(path)

    def collect(self, name):
        values = dict(self.defaults[name])
        values.update(n=int(self.shared_vars["n"].get()), wavelength=float(self.shared_vars["wavelength_nm"].get())*1e-9,
                      backend=self.shared_vars["backend"].get())
        for key, text, scale, choices in self.specifications[name]:
            raw = self.local_vars[name][key].get().strip()
            old = values[key]
            values[key] = int(raw) if isinstance(old, int) else (float(raw)/scale if isinstance(old, float) else raw)
        if name != "talbot":
            values["image_path"] = self.local_vars[name]["image_path"].get().strip() or None
            values["image_mode"] = self.local_vars[name]["image_mode"].get()
        return MODULES[name][0](**values)

    def start(self):
        name = self.module_by_tab[self.tabs.select()]
        try:
            config = self.collect(name)
        except ValueError as exc:
            messagebox.showerror("参数格式错误", str(exc))
            return
        self.run_button.configure(state="disabled")
        self.progress.start(15)
        self.status.set(f"正在计算{MODULES[name][2]}并生成报告……")
        output = OUTPUTS_DIR/datetime.now().strftime("%Y%m%d_%H%M%S_%f")/name
        def worker():
            try:
                from .reporting import export_result
                result = MODULES[name][1](config)
                report = export_result(result, output)
                self.messages.put(("success", report, result.metrics))
            except Exception as exc:
                self.messages.put(("error", str(exc), None))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            state, data, metrics = self.messages.get_nowait()
            self.run_button.configure(state="normal")
            self.progress.stop()
            if state=="error":
                self.status.set("计算失败，请检查参数。")
                messagebox.showerror("计算失败", data)
            else:
                import json
                self.report_path = data
                figure_index = data.parent/"figures.json"
                self.figure_select["values"] = json.loads(figure_index.read_text(encoding="utf-8")) if figure_index.is_file() else sorted(path.relative_to(data.parent).as_posix() for path in data.parent.rglob("*.png"))
                self.figure_select.current(0)
                self.show_selected()
                self.status.set(f"计算完成。已保存至：\n{data.parent}")
                self.details.configure(state="normal")
                self.details.delete("1.0", "end")
                self.details.insert("1.0", json.dumps(metrics, ensure_ascii=False, indent=2))
                self.details.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(150, self.poll)

    def show_selected(self):
        if self.report_path and self.figure_select.get():
            with Image.open(self.report_path.parent/self.figure_select.get()) as source:
                self.preview_source = source.convert("RGB")
            self.resize_preview()

    def resize_preview(self, event=None):
        if self.preview_source:
            picture = self.preview_source.copy()
            picture.thumbnail((max(1, self.preview.winfo_width()-12), max(1, self.preview.winfo_height()-12)), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(picture)
            self.preview.configure(image=self.preview_photo, text="")

    def open_report(self):
        if self.report_path:
            os.startfile(self.report_path)

    def open_directory(self):
        if self.report_path:
            os.startfile(self.report_path.parent)


def launch(config=None):
    root = tk.Tk()
    SimulatorApp(root, config)
    root.mainloop()
