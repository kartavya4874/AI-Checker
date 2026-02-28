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
        self._refresh_courses()

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

        # Left panel: Course/Assessment setup
        left = ttk.LabelFrame(main, text="Course & Assessment", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Course selection
        ttk.Label(left, text="Course:").pack(anchor="w")
        course_frame = ttk.Frame(left)
        course_frame.pack(fill="x", pady=(0, 8))
        self.course_combo = ttk.Combobox(course_frame, state="readonly", width=25)
        self.course_combo.pack(side="left", fill="x", expand=True)
        self.course_combo.bind("<<ComboboxSelected>>", self._on_course_selected)
        ttk.Button(course_frame, text="+", width=3, command=self._add_course).pack(side="right", padx=(5, 0))

        # Assessment title
        ttk.Label(left, text="Assessment Title:").pack(anchor="w")
        self.assessment_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.assessment_var).pack(fill="x", pady=(0, 8))

        # Answer key
        ttk.Label(left, text="Answer Key PDF:").pack(anchor="w")
        key_frame = ttk.Frame(left)
        key_frame.pack(fill="x", pady=(0, 8))
        self.key_label = ttk.Label(key_frame, text="No file selected", wraplength=250)
        self.key_label.pack(side="left", fill="x", expand=True)
        ttk.Button(key_frame, text="Browse", command=self._browse_answer_key).pack(side="right")

        # Marks per question
        ttk.Label(left, text="Marks per Question (comma-separated):").pack(anchor="w")
        self.marks_var = tk.StringVar(value="10,10,10,10,10")
        ttk.Entry(left, textvariable=self.marks_var).pack(fill="x", pady=(0, 8))

        # Student PDFs
        ttk.Label(left, text="Student Answer Sheets:").pack(anchor="w")
        student_frame = ttk.Frame(left)
        student_frame.pack(fill="x", pady=(0, 8))
        self.student_label = ttk.Label(student_frame, text="0 files selected")
        self.student_label.pack(side="left", fill="x", expand=True)
        ttk.Button(student_frame, text="Browse", command=self._browse_students).pack(side="right")

        # Teacher samples (optional)
        ttk.Label(left, text="Teacher Samples (optional):").pack(anchor="w")
        sample_frame = ttk.Frame(left)
        sample_frame.pack(fill="x", pady=(0, 8))
        self.sample_label = ttk.Label(sample_frame, text="No samples selected")
        self.sample_label.pack(side="left", fill="x", expand=True)
        ttk.Button(sample_frame, text="Browse", command=self._browse_samples).pack(side="right")

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
        """Refresh course list."""
        courses = self.db.get_all_courses()
        values = [f"{c['code']} — {c['name']}" for c in courses]
        self.course_combo["values"] = values
        self._courses_data = courses

    def _on_course_selected(self, event=None):
        idx = self.course_combo.current()
        if idx >= 0 and idx < len(self._courses_data):
            self.current_course_id = self._courses_data[idx]["id"]
            self._refresh_assessments_combo()

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

    def _browse_answer_key(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if path:
            self.answer_key_path = path
            self.key_label.config(text=Path(path).name)

    def _browse_students(self):
        paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
        if paths:
            self.student_files = list(paths)
            self.student_label.config(text=f"{len(self.student_files)} files selected")

    def _browse_samples(self):
        paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
        if paths:
            self.sample_files = list(paths)
            self.sample_label.config(text=f"{len(self.sample_files)} samples selected")

    def _start_processing(self):
        """Start processing in a background thread."""
        if self.processing:
            messagebox.showwarning("Processing", "Already processing!")
            return

        # Validate inputs
        if not self.answer_key_path:
            messagebox.showerror("Error", "Please select an answer key PDF")
            return
        if not self.student_files:
            messagebox.showerror("Error", "Please select student answer sheets")
            return
        if not self.assessment_var.get():
            messagebox.showerror("Error", "Please enter an assessment title")
            return

        # Parse marks
        try:
            marks_str = self.marks_var.get().strip()
            self.marks_per_question = [float(m.strip()) for m in marks_str.split(",")]
        except ValueError:
            messagebox.showerror("Error", "Invalid marks format. Use comma-separated numbers.")
            return

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
            # Create course if needed
            course_code = "COURSE"
            if self.course_combo.get():
                course_code = self.course_combo.get().split(" — ")[0]
            else:
                course_code = "COURSE001"

            # Copy student files to a temp directory
            import shutil
            temp_students = config.TEMP_DIR / "students"
            temp_students.mkdir(parents=True, exist_ok=True)

            for f in self.student_files:
                shutil.copy2(f, temp_students / Path(f).name)

            # Copy samples if any
            samples_dir = None
            if self.sample_files:
                samples_dir = str(config.TEMP_DIR / "samples")
                Path(samples_dir).mkdir(parents=True, exist_ok=True)
                for f in self.sample_files:
                    shutil.copy2(f, Path(samples_dir) / Path(f).name)

            total = len(self.student_files)

            def progress_cb(msg):
                self._log(msg)
                # Update progress bar
                if "Processing student" in msg:
                    try:
                        parts = msg.split("Processing student ")[1].split("/")
                        current = int(parts[0])
                        self.progress_var.set((current / total) * 100)
                        self.progress_label.config(text=f"Student {current}/{total}")
                    except (IndexError, ValueError):
                        pass

            processor = CourseProcessor(self.db)
            processor.process(
                course_code=course_code,
                assessment_title=self.assessment_var.get(),
                answer_key_path=self.answer_key_path,
                students_dir=str(temp_students),
                marks_per_question=self.marks_per_question,
                samples_dir=samples_dir,
                progress_callback=progress_cb,
            )

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
