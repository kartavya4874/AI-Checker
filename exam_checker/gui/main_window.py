"""
Tkinter Desktop GUI for the Hybrid AI Exam Checker.
Full-featured GUI with threading for non-blocking processing.
"""

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from database.db_manager import DatabaseManager
from processing.course_processor import CourseProcessor


class ExamCheckerGUI(tk.Tk):
    """Main Tkinter application window."""

    def __init__(self):
        super().__init__()
        self.title("Hybrid AI University Exam Checker")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # Configure dark theme
        self.configure(bg="#1a1a2e")
        self._setup_styles()

        # Database
        self.db = DatabaseManager()
        self.db.init_db()

        # State
        self.processing = False
        self.current_course_id = None
        self.current_assessment_id = None
        self.answer_key_path = None
        self.student_files = []
        self.sample_files = []
        self.marks_per_question = []

        # Build UI
        self._create_menu()
        self._create_main_layout()
        # Note: courses are now auto-created from folder metadata,
        # no need to populate a dropdown at startup.

    def _setup_styles(self):
        """Configure ttk styles for dark theme."""
        style = ttk.Style(self)
        style.theme_use("clam")

        # Colors
        bg = "#1a1a2e"
        fg = "#e0e0e0"
        accent = "#0f3460"
        highlight = "#16213e"
        btn_bg = "#e94560"
        btn_fg = "#ffffff"
        entry_bg = "#16213e"

        style.configure(".", background=bg, foreground=fg, fieldbackground=entry_bg)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg, font=("Segoe UI", 10))
        style.configure("TButton", background=btn_bg, foreground=btn_fg, font=("Segoe UI", 10, "bold"), padding=8)
        style.map("TButton", background=[("active", "#c0392b")])
        style.configure("Accent.TButton", background="#27ae60", foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#2ecc71")])
        style.configure("TEntry", fieldbackground=entry_bg, foreground=fg)
        style.configure("TCombobox", fieldbackground=entry_bg, foreground=fg)
        style.configure("TNotebook", background=bg)
        style.configure("TNotebook.Tab", background=accent, foreground=fg, padding=[12, 6])
        style.map("TNotebook.Tab", background=[("selected", highlight)])
        style.configure("Treeview", background=entry_bg, foreground=fg, fieldbackground=entry_bg, rowheight=28)
        style.configure("Treeview.Heading", background=accent, foreground=fg, font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", btn_bg)])
        style.configure("TLabelframe", background=bg, foreground=fg)
        style.configure("TLabelframe.Label", background=bg, foreground=fg, font=("Segoe UI", 11, "bold"))
        style.configure("Horizontal.TProgressbar", troughcolor=entry_bg, background=btn_bg)

    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self, bg="#16213e", fg="#e0e0e0", activebackground="#e94560")

        file_menu = tk.Menu(menubar, tearoff=0, bg="#16213e", fg="#e0e0e0")
        file_menu.add_command(label="Export Results (CSV)", command=self._export_csv)
        file_menu.add_command(label="Export Results (Excel)", command=self._export_excel)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=0, bg="#16213e", fg="#e0e0e0")
        settings_menu.add_command(label="Settings", command=self._show_settings)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        self.config(menu=menubar)

    def _create_main_layout(self):
        """Create the main layout with notebook tabs."""
        # Header
        header = ttk.Frame(self)
        header.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Label(header, text="🎓 Hybrid AI Exam Checker", font=("Segoe UI", 18, "bold")).pack(side="left")

        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # Tab 1: Setup & Process
        self.setup_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.setup_tab, text="  📝 Setup & Process  ")
        self._create_setup_tab()

        # Tab 2: Results
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="  📊 Results  ")
        self._create_results_tab()

        # Tab 3: Student Detail
        self.detail_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.detail_tab, text="  👤 Student Detail  ")
        self._create_detail_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, font=("Segoe UI", 9))
        status_bar.pack(fill="x", padx=10, pady=(0, 5))

    # ==================== Setup Tab ====================

    def _create_setup_tab(self):
        """Create the setup and processing tab."""
        main = ttk.Frame(self.setup_tab)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel: Exam folder & auto-detected details
        left = ttk.LabelFrame(main, text="Exam Folder & Course", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # ── Single exam folder ────────────────────────────────────
        ttk.Label(
            left,
            text="Exam Folder Path:",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(4, 0))

        folder_frame = ttk.Frame(left)
        folder_frame.pack(fill="x", pady=(2, 4))
        self.folder_var = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var)
        folder_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(folder_frame, text="Browse", command=self._browse_exam_folder).pack(
            side="right", padx=(5, 0)
        )

        # Info hint
        ttk.Label(
            left,
            text=(
                "📂  Name your files:\n"
                "  • answer_key.pdf  (must include 'key' or 'answer')\n"
                "  • question_paper.pdf  (optional)\n"
                "  • All other PDFs → student sheets\n"
                "  • Add course_info.txt for exact course details"
            ),
            font=("Segoe UI", 8),
            foreground="#888888",
            justify="left",
            wraplength=290,
        ).pack(anchor="w", pady=(0, 6))

        # ── Auto-detected course info (editable override) ─────────
        ttk.Label(left, text="Auto-Detected Course Code:",
                  font=("Segoe UI", 9)).pack(anchor="w")
        self.course_code_var = tk.StringVar(value="")
        ttk.Entry(left, textvariable=self.course_code_var,
                  font=("Segoe UI", 9)).pack(fill="x", pady=(0, 6))

        ttk.Label(left, text="Auto-Detected Assessment Title:",
                  font=("Segoe UI", 9)).pack(anchor="w")
        self.assessment_var = tk.StringVar(value="")
        ttk.Entry(left, textvariable=self.assessment_var,
                  font=("Segoe UI", 9)).pack(fill="x", pady=(0, 4))

        ttk.Label(
            left,
            text="ℹ️  Edit the fields above to override auto-detected values",
            font=("Segoe UI", 8, "italic"),
            foreground="#58a6ff",
            wraplength=290,
        ).pack(anchor="w", pady=(0, 6))

        # Scan preview label
        self.scan_preview_var = tk.StringVar(value="")
        self.scan_preview_label = ttk.Label(
            left,
            textvariable=self.scan_preview_var,
            font=("Consolas", 8),
            foreground="#aaaaaa",
            wraplength=290,
            justify="left",
        )
        self.scan_preview_label.pack(anchor="w", pady=(0, 8))

        # Bind folder changes to live preview + auto-fill
        self.folder_var.trace_add("write", self._on_folder_changed)


        # Process button
        self.process_btn = ttk.Button(left, text="⚡ Process All", style="Accent.TButton", command=self._start_processing)
        self.process_btn.pack(fill="x", pady=(15, 5))

        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(left, variable=self.progress_var, mode="determinate")
        self.progress_bar.pack(fill="x", pady=(0, 5))
        self.progress_label = ttk.Label(left, text="")
        self.progress_label.pack(anchor="w")

        # Right panel: Log
        right = ttk.LabelFrame(main, text="Processing Log", padding=10)
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))

        self.log_text = scrolledtext.ScrolledText(
            right, bg="#0d1117", fg="#c9d1d9", font=("Consolas", 9),
            insertbackground="#c9d1d9", wrap="word", state="disabled"
        )
        self.log_text.pack(fill="both", expand=True)

    # ==================== Results Tab ====================

    def _create_results_tab(self):
        """Create the results viewing tab."""
        top = ttk.Frame(self.results_tab)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="Assessment:").pack(side="left")
        self.results_assessment_combo = ttk.Combobox(top, state="readonly", width=40)
        self.results_assessment_combo.pack(side="left", padx=5)
        ttk.Button(top, text="Load Results", command=self._load_results).pack(side="left", padx=5)
        ttk.Button(top, text="Refresh", command=self._refresh_assessments_combo).pack(side="left")

        # Results table
        columns = ("roll", "name", "marks", "total", "percentage", "grade", "status")
        self.results_tree = ttk.Treeview(self.results_tab, columns=columns, show="headings", selectmode="browse")
        self.results_tree.heading("roll", text="Roll Number")
        self.results_tree.heading("name", text="Student Name")
        self.results_tree.heading("marks", text="Marks")
        self.results_tree.heading("total", text="Total")
        self.results_tree.heading("percentage", text="%")
        self.results_tree.heading("grade", text="Grade")
        self.results_tree.heading("status", text="Status")

        self.results_tree.column("roll", width=120)
        self.results_tree.column("name", width=200)
        self.results_tree.column("marks", width=80)
        self.results_tree.column("total", width=80)
        self.results_tree.column("percentage", width=80)
        self.results_tree.column("grade", width=60)
        self.results_tree.column("status", width=100)

        scrollbar = ttk.Scrollbar(self.results_tab, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        self.results_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10), side="left")
        scrollbar.pack(fill="y", side="right", pady=(0, 10), padx=(0, 10))

        self.results_tree.bind("<Double-1>", self._on_result_double_click)

    # ==================== Detail Tab ====================

    def _create_detail_tab(self):
        """Create the student detail tab."""
        # Student info
        info = ttk.LabelFrame(self.detail_tab, text="Student Information", padding=10)
        info.pack(fill="x", padx=10, pady=(10, 5))

        self.detail_info_var = tk.StringVar(value="Select a student from the Results tab")
        ttk.Label(info, textvariable=self.detail_info_var, font=("Segoe UI", 12)).pack(anchor="w")

        # Reprocess button
        btn_frame = ttk.Frame(info)
        btn_frame.pack(fill="x", pady=(5, 0))
        ttk.Button(btn_frame, text="🔄 Re-process Student", command=self._reprocess_student).pack(side="right")

        # Question results table
        columns = ("q_num", "marks_alloc", "marks_obt", "type", "status", "feedback")
        self.detail_tree = ttk.Treeview(self.detail_tab, columns=columns, show="headings", selectmode="browse")
        self.detail_tree.heading("q_num", text="Q#")
        self.detail_tree.heading("marks_alloc", text="Allocated")
        self.detail_tree.heading("marks_obt", text="Obtained")
        self.detail_tree.heading("type", text="Type")
        self.detail_tree.heading("status", text="Status")
        self.detail_tree.heading("feedback", text="Feedback")

        self.detail_tree.column("q_num", width=50)
        self.detail_tree.column("marks_alloc", width=80)
        self.detail_tree.column("marks_obt", width=80)
        self.detail_tree.column("type", width=100)
        self.detail_tree.column("status", width=90)
        self.detail_tree.column("feedback", width=500)

        detail_scroll = ttk.Scrollbar(self.detail_tab, orient="vertical", command=self.detail_tree.yview)
        self.detail_tree.configure(yscrollcommand=detail_scroll.set)

        self.detail_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10), side="left")
        detail_scroll.pack(fill="y", side="right", pady=(0, 10), padx=(0, 10))

    # ==================== Event Handlers ====================

    def _refresh_courses(self):
        """Refresh cached course list (no combo to update — courses are auto-detected)."""
        self._courses_data = self.db.get_all_courses()

    def _on_course_selected(self, event=None):
        """Legacy stub — course selection is now automatic."""
        pass

    def _add_course(self):
        """Add a new course dialog."""
        dialog = tk.Toplevel(self)
        dialog.title("Add Course")
        dialog.geometry("400x250")
        dialog.configure(bg="#1a1a2e")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="Course Code:").pack(anchor="w", padx=20, pady=(20, 0))
        code_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=code_var).pack(fill="x", padx=20)

        ttk.Label(dialog, text="Course Name:").pack(anchor="w", padx=20, pady=(10, 0))
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var).pack(fill="x", padx=20)

        ttk.Label(dialog, text="Department:").pack(anchor="w", padx=20, pady=(10, 0))
        dept_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=dept_var).pack(fill="x", padx=20)

        def save():
            if code_var.get() and name_var.get():
                self.db.create_course(name=name_var.get(), code=code_var.get(), department=dept_var.get())
                self._refresh_courses()
                dialog.destroy()

        ttk.Button(dialog, text="Save", style="Accent.TButton", command=save).pack(pady=20)

    def _browse_exam_folder(self):
        """Browse for the root exam folder."""
        folder = filedialog.askdirectory(title="Select Exam Folder")
        if folder:
            self.folder_var.set(folder)

    def _on_folder_changed(self, *_):
        """Live-preview folder scan + auto-fill course/assessment fields."""
        folder = self.folder_var.get().strip()
        if not folder:
            self.scan_preview_var.set("")
            return
        from utils.folder_scanner import scan_exam_folder, describe_scan, parse_exam_metadata
        from pathlib import Path
        p = Path(folder)
        if not p.is_dir():
            self.scan_preview_var.set("⚠ Folder not found")
            return

        # Auto-fill course details
        meta = parse_exam_metadata(folder)
        if not self.course_code_var.get():  # only auto-fill if not already set
            self.course_code_var.set(meta.course_code)
        if not self.assessment_var.get():   # only auto-fill if not already set
            self.assessment_var.set(meta.assessment_title)

        # Live scan preview
        scan = scan_exam_folder(folder)
        preview = (
            f"Course Code:  {meta.course_code}  (from {meta.source})\n"
            f"Assessment:   {meta.assessment_title}\n"
            + describe_scan(scan)
        )
        self.scan_preview_var.set(preview)

    # Legacy stubs kept for any existing code references
    def _browse_answer_key(self):
        pass

    def _browse_students(self):
        pass

    def _browse_samples(self):
        pass

    def _start_processing(self):
        """Start processing in a background thread."""
        if self.processing:
            messagebox.showwarning("Processing", "Already processing!")
            return

        # Validate inputs
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showerror("Error", "Please select an exam folder")
            return
        from pathlib import Path as _Path
        if not _Path(folder).is_dir():
            messagebox.showerror("Error", f"Folder not found: {folder}")
            return

        # Quick scan to check folder has required files
        from utils.folder_scanner import scan_exam_folder
        scan = scan_exam_folder(folder)
        if not scan.answer_key_path:
            messagebox.showerror(
                "Folder Error",
                "No answer key found in folder.\n\n"
                "Rename the answer key file to include 'answer_key', 'key', or 'solution'."
            )
            return
        if not scan.student_paths:
            messagebox.showerror(
                "Folder Error",
                "No student answer sheets found in folder.\n\n"
                "Place student PDFs (without 'key'/'question' in their names) in the folder."
            )
            return

        # Marks are auto-detected from answer key; no manual input needed

        # Validate API keys
        try:
            config.validate()
        except ValueError as e:
            messagebox.showerror("Configuration Error", str(e))
            return

        self.processing = True
        self.process_btn.configure(state="disabled")
        self.progress_var.set(0)
        self._log("Starting processing...\n")

        thread = threading.Thread(target=self._process_thread, daemon=True)
        thread.start()

    def _process_thread(self):
        """Background processing thread."""
        try:
            folder = self.folder_var.get().strip()

            # Read the (possibly user-edited) auto-filled fields as overrides.
            # If left blank, process_from_folder() auto-detects from folder name/metadata.
            course_code = self.course_code_var.get().strip() or None
            assessment_title = self.assessment_var.get().strip() or None

            def progress_cb(msg):
                self._log(msg)
                if "Processing student" in msg:
                    try:
                        parts = msg.split("Processing student ")[1].split("/")
                        current = int(parts[0])
                        total = int(parts[1].split(":")[0])
                        self.progress_var.set((current / total) * 100)
                        self.progress_label.config(text=f"Student {current}/{total}")
                    except (IndexError, ValueError):
                        pass

            processor = CourseProcessor(self.db)
            summary = processor.process_from_folder(
                root_folder=folder,
                course_code=course_code,
                assessment_title=assessment_title,
                progress_callback=progress_cb,
            )


            if summary.get("errors"):
                self._log("\n⚠ Warnings/Errors:")
                for e in summary["errors"]:
                    self._log(f"  {e}")

            self.progress_var.set(100)
            self._log("\n✓ Processing complete!")
            self.after(0, lambda: messagebox.showinfo("Complete", "Processing finished!"))

        except Exception as e:
            self._log(f"\n✗ Error: {str(e)}")
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

        finally:
            self.processing = False
            self.after(0, lambda: self.process_btn.configure(state="normal"))
            self.after(0, self._refresh_courses)

    def _log(self, message):
        """Append message to log widget (thread-safe)."""
        def update():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.after(0, update)

    # ==================== Results ====================

    def _refresh_assessments_combo(self):
        """Refresh assessments combo in results tab."""
        all_courses = self.db.get_all_courses()
        assessments = []
        for c in all_courses:
            for a in self.db.get_assessments_by_course(c["id"]):
                assessments.append({
                    **a,
                    "course_code": c["code"],
                    "display": f"{c['code']} — {a['title']}",
                })
        self._assessments_data = assessments
        self.results_assessment_combo["values"] = [a["display"] for a in assessments]

    def _load_results(self):
        """Load and display results for selected assessment."""
        idx = self.results_assessment_combo.current()
        if idx < 0:
            return

        assessment = self._assessments_data[idx]
        results = self.db.get_results_by_assessment(assessment["id"])

        # Clear tree
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Populate
        self._results_data = results
        for r in results:
            self.results_tree.insert("", "end", values=(
                r["roll_number"], r["student_name"],
                r["total_marks_obtained"], r["total_marks_allocated"],
                f"{r['percentage']:.1f}%", r["grade"],
                r["processing_status"],
            ))

        self.status_var.set(f"Loaded {len(results)} results")

    def _on_result_double_click(self, event):
        """Show student detail when double-clicked."""
        sel = self.results_tree.selection()
        if not sel:
            return

        idx = self.results_tree.index(sel[0])
        if idx < len(self._results_data):
            result = self._results_data[idx]
            self._show_student_detail(result)
            self.notebook.select(self.detail_tab)

    def _show_student_detail(self, result):
        """Display detailed results for a student."""
        self._current_detail_result = result
        self.detail_info_var.set(
            f"{result['student_name']} ({result['roll_number']}) — "
            f"{result['total_marks_obtained']}/{result['total_marks_allocated']} "
            f"({result['percentage']:.1f}%) — Grade: {result['grade']}"
        )

        # Load question results
        qrs = self.db.get_question_results(result["id"])

        for item in self.detail_tree.get_children():
            self.detail_tree.delete(item)

        for qr in qrs:
            feedback = qr["feedback"][:100] + "..." if len(qr.get("feedback", "")) > 100 else qr.get("feedback", "")
            self.detail_tree.insert("", "end", values=(
                f"Q{qr['question_number']}",
                qr["marks_allocated"],
                qr["marks_obtained"],
                qr["content_type"],
                qr["status"],
                feedback,
            ))

    def _reprocess_student(self):
        """Re-process the currently selected student."""
        if not hasattr(self, '_current_detail_result'):
            messagebox.showinfo("Info", "No student selected")
            return
        messagebox.showinfo("Re-process", "Re-processing will be implemented with existing pipeline")

    # ==================== Export ====================

    def _export_csv(self):
        """Export results to CSV."""
        idx = self.results_assessment_combo.current()
        if idx < 0:
            messagebox.showinfo("Info", "Please select an assessment first")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if path:
            assessment = self._assessments_data[idx]
            csv_data = self.db.export_results_csv(assessment["id"])
            with open(path, "w", newline="") as f:
                f.write(csv_data)
            messagebox.showinfo("Exported", f"Results exported to {path}")

    def _export_excel(self):
        """Export results to Excel."""
        idx = self.results_assessment_combo.current()
        if idx < 0:
            messagebox.showinfo("Info", "Please select an assessment first")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if path:
            assessment = self._assessments_data[idx]
            self.db.export_results_excel(assessment["id"], path)
            messagebox.showinfo("Exported", f"Results exported to {path}")

    # ==================== Settings ====================

    def _show_settings(self):
        """Show settings dialog."""
        dialog = tk.Toplevel(self)
        dialog.title("Settings")
        dialog.geometry("500x450")
        dialog.configure(bg="#1a1a2e")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="⚙️ Settings", font=("Segoe UI", 16, "bold")).pack(pady=(15, 10))

        # API Keys
        ttk.Label(dialog, text="OpenAI API Key:").pack(anchor="w", padx=20, pady=(10, 0))
        api_var = tk.StringVar(value=config.OPENAI_API_KEY)
        ttk.Entry(dialog, textvariable=api_var, show="*").pack(fill="x", padx=20)

        ttk.Label(dialog, text="Google Vision Credentials Path:").pack(anchor="w", padx=20, pady=(10, 0))
        gv_var = tk.StringVar(value=config.GOOGLE_VISION_CREDENTIALS)
        ttk.Entry(dialog, textvariable=gv_var).pack(fill="x", padx=20)

        # Grade Boundaries
        ttk.Label(dialog, text="Grade Boundaries (JSON):").pack(anchor="w", padx=20, pady=(10, 0))
        gb_var = tk.StringVar(value=json.dumps(config.GRADE_BOUNDARIES))
        ttk.Entry(dialog, textvariable=gb_var).pack(fill="x", padx=20)

        # Negative Marking
        ttk.Label(dialog, text="Negative Marking Factor:").pack(anchor="w", padx=20, pady=(10, 0))
        nm_var = tk.StringVar(value=str(config.NEGATIVE_MARKING_FACTOR))
        ttk.Entry(dialog, textvariable=nm_var).pack(fill="x", padx=20)

        # Rate Limit
        ttk.Label(dialog, text="API Rate Limit (req/min):").pack(anchor="w", padx=20, pady=(10, 0))
        rl_var = tk.StringVar(value=str(config.RATE_LIMIT_RPM))
        ttk.Entry(dialog, textvariable=rl_var).pack(fill="x", padx=20)

        def save_settings():
            config.OPENAI_API_KEY = api_var.get()
            config.GOOGLE_VISION_CREDENTIALS = gv_var.get()
            try:
                config.GRADE_BOUNDARIES = json.loads(gb_var.get())
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Invalid grade boundaries JSON")
                return
            try:
                config.NEGATIVE_MARKING_FACTOR = float(nm_var.get())
                config.RATE_LIMIT_RPM = int(rl_var.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid numeric value")
                return

            if config.GOOGLE_VISION_CREDENTIALS:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GOOGLE_VISION_CREDENTIALS

            messagebox.showinfo("Saved", "Settings updated for this session")
            dialog.destroy()

        ttk.Button(dialog, text="Save", style="Accent.TButton", command=save_settings).pack(pady=20)


if __name__ == "__main__":
    app = ExamCheckerGUI()
    app.mainloop()
