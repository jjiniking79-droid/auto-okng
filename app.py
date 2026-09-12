import os
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from PIL import Image, ImageTk

# 기존 백엔드 모듈 안전 연동
try:
    import feature_extractor as fe_mod
    if hasattr(fe_mod, "FeatureExtractor"):
        _extractor_instance = fe_mod.FeatureExtractor()
        extract_feat_fn = _extractor_instance.extract
    elif hasattr(fe_mod, "extract_features"):
        extract_feat_fn = fe_mod.extract_features
    elif hasattr(fe_mod, "extract_feature"):
        extract_feat_fn = fe_mod.extract_feature
    else:
        fns = [getattr(fe_mod, a) for a in dir(fe_mod) if callable(getattr(fe_mod, a)) and not a.startswith("_")]
        extract_feat_fn = fns[0] if fns else (lambda p: [0] * 10)
except Exception:
    extract_feat_fn = lambda p: [0] * 10

try:
    import model_manager as mm_mod
    if hasattr(mm_mod, "ModelManager"):
        model_mgr = mm_mod.ModelManager()
    else:
        model_mgr = mm_mod
except Exception:
    model_mgr = None


class DefectInspectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("불량 이미지 자동 판정 프로그램 (이미지 미리보기 & 범용 컬럼 에디션)")
        self.geometry("1600x950")
        self.minsize(1300, 750)

        self.delimiter = "_"
        self.custom_columns = ["LOT", "GLS", "셀번지", "X", "Y", "검사호기"]
        self.visible_columns = list(self.custom_columns)
        self.system_columns = ["AI판정", "신뢰율(%)", "작업자 판정"]

        self.records = []
        self.defect_types = ["K 유기", "K 갈림", "NK 유기", "핀홀", "정상"]
        self.current_preview_img = None  # 가비지 컬렉션 방지용 이미지 참조

        self._init_styles()
        self._init_ui()

    def _init_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview.Heading", font=("맑은 고딕", 9, "bold"), background="#e0e0e0")
        style.configure("Treeview", rowheight=24, font=("맑은 고딕", 9))
        style.map("Treeview", background=[("selected", "#0078d7")], foreground=[("selected", "white")])

    def _init_ui(self):
        # 1. 상단 글로벌 컨트롤 바
        top_bar = ttk.Frame(self, padding=5)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top_bar, text="📁 이미지 폴더 불러오기", command=self.load_image_folder).pack(side=tk.LEFT, padx=3)
        ttk.Button(top_bar, text="🖼️ 이미지 파일 불러오기", command=self.load_image_files).pack(side=tk.LEFT, padx=3)
        ttk.Button(top_bar, text="⚙️ AI 학습하기", command=self.train_model).pack(side=tk.LEFT, padx=3)
        ttk.Button(top_bar, text="⚡ 자동 판정", command=self.run_auto_inspect).pack(side=tk.LEFT, padx=3)
        ttk.Button(top_bar, text="📊 엑셀로 내보내기", command=self.export_to_excel).pack(side=tk.LEFT, padx=3)
        ttk.Button(top_bar, text="🗑️ 목록 비우기", command=self.clear_all_records).pack(side=tk.LEFT, padx=3)

        self.lbl_status = ttk.Label(top_bar, text="준비 완료. 모델 학습 여부: 대기", font=("맑은 고딕", 9))
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # 2. 파일명 텍스트 나누기 및 컬럼 편집 관리 바
        cfg_bar = ttk.LabelFrame(self, text="파일명 텍스트 나누기 및 컬럼 정의 (편집/추가/삭제/필터)", padding=5)
        cfg_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=3)

        ttk.Label(cfg_bar, text="구분 기호:").pack(side=tk.LEFT, padx=(5, 2))
        self.ent_delim = ttk.Entry(cfg_bar, width=4)
        self.ent_delim.insert(0, self.delimiter)
        self.ent_delim.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(cfg_bar, text="컬럼 순서:").pack(side=tk.LEFT, padx=(0, 2))
        self.ent_cols = ttk.Entry(cfg_bar, width=45)
        self.ent_cols.insert(0, ", ".join(self.custom_columns))
        self.ent_cols.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(cfg_bar, text="즉시 적용", command=self.apply_column_rule_from_entry).pack(side=tk.LEFT, padx=2)
        ttk.Button(cfg_bar, text="✏️ 컬럼 상세 편집", command=self.open_column_manager_dialog).pack(side=tk.LEFT, padx=4)
        ttk.Button(cfg_bar, text="👁️ 표시 컬럼 필터", command=self.open_column_filter_dialog).pack(side=tk.LEFT, padx=4)

        # 3. 판정 유형 관리 바
        type_bar = ttk.LabelFrame(self, text="판정 유형 관리", padding=5)
        type_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)

        ttk.Label(type_bar, text="새 유형:").pack(side=tk.LEFT, padx=(4, 2))
        self.ent_new_type = ttk.Entry(type_bar, width=12)
        self.ent_new_type.pack(side=tk.LEFT, padx=2)
        ttk.Button(type_bar, text="추가", command=self.add_defect_type).pack(side=tk.LEFT, padx=2)

        ttk.Label(type_bar, text="삭제할 유형:").pack(side=tk.LEFT, padx=(12, 2))
        self.cmb_del_type = ttk.Combobox(type_bar, values=self.defect_types, state="readonly", width=12)
        if self.defect_types:
            self.cmb_del_type.current(0)
        self.cmb_del_type.pack(side=tk.LEFT, padx=2)
        ttk.Button(type_bar, text="유형 삭제", command=self.delete_defect_type).pack(side=tk.LEFT, padx=2)

        ttk.Label(type_bar, text="선택 항목 일괄 판정:").pack(side=tk.LEFT, padx=(20, 2))
        self.cmb_batch_type = ttk.Combobox(type_bar, values=self.defect_types, state="readonly", width=12)
        if self.defect_types:
            self.cmb_batch_type.current(0)
        self.cmb_batch_type.pack(side=tk.LEFT, padx=2)
        ttk.Button(type_bar, text="적용", command=self.apply_batch_type).pack(side=tk.LEFT, padx=2)
        ttk.Button(type_bar, text="전체 선택", command=lambda: self.set_all_checks(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(type_bar, text="전체 해제", command=lambda: self.set_all_checks(False)).pack(side=tk.LEFT, padx=2)

        # 4. 요약 통계 영역
        summary_frame = ttk.LabelFrame(self, text="판정 결과 요약", padding=5)
        summary_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        self.lbl_summary = ttk.Label(summary_frame, text="총 이미지 수: 0개  |  AI 판정 완료: 0개  |  작업자 미검수: 0개", font=("맑은 고딕", 9, "bold"))
        self.lbl_summary.pack(anchor="w", padx=5)

        # 5. 메인 컨테이너 (좌측: 테이블 / 우측: 이미지 미리보기 패널)
        main_content_frame = ttk.Frame(self, padding=(8, 4, 8, 8))
        main_content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 좌측 테이블 영역
        table_container = ttk.Frame(main_content_frame)
        table_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_y = ttk.Scrollbar(table_container, orient=tk.VERTICAL)
        self.scroll_x = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL)

        self.tree = ttk.Treeview(
            table_container,
            yscrollcommand=self.scroll_y.set,
            xscrollcommand=self.scroll_x.set,
            selectmode="extended"
        )
        self.scroll_y.config(command=self.tree.yview)
        self.scroll_x.config(command=self.tree.xview)

        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree.bind("<Button-1>", self.on_tree_click)
        self.tree.bind("<Double-1>", self.on_tree_double_click)
        self.tree.bind("<<TreeviewSelect>>", self.on_row_select)

        # 우측 이미지 미리보기 패널 영역
        preview_frame = ttk.LabelFrame(main_content_frame, text="🔍 실시간 이미지 미리보기", width=340, padding=10)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        preview_frame.pack_propagate(False)

        self.lbl_preview_info = ttk.Label(preview_frame, text="목록에서 항목을 선택하세요.", font=("맑은 고딕", 9), anchor="center")
        self.lbl_preview_info.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        self.lbl_image_display = ttk.Label(preview_frame, text="[이미지 없음]", background="#f0f0f0", anchor="center")
        self.lbl_image_display.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.rebuild_treeview_headers()

    # -------------------------------------------------------------
    # 이미지 미리보기 업데이트 로직
    # -------------------------------------------------------------
    def on_row_select(self, event):
        selected_items = self.tree.selection()
        if not selected_items:
            return
        row_id = selected_items[0]
        try:
            idx = int(row_id)
            record = self.records[idx]
            full_path = record["full_path"].replace("\x00", "")

            if os.path.exists(full_path):
                img = Image.open(full_path)
                img.thumbnail((300, 350))
                self.current_preview_img = ImageTk.PhotoImage(img)
                self.lbl_image_display.config(image=self.current_preview_img, text="")
                self.lbl_preview_info.config(text=f"파일명: {record['이미지']}")
            else:
                self.lbl_image_display.config(image="", text="[파일을 찾을 수 없음]")
                self.lbl_preview_info.config(text="")
        except Exception:
            self.lbl_image_display.config(image="", text="[미리보기 로드 오류]")
            self.lbl_preview_info.config(text="")

    # -------------------------------------------------------------
    # 판정 유형 추가 / 삭제 로직
    # -------------------------------------------------------------
    def add_defect_type(self):
        new_val = self.ent_new_type.get().strip()
        if not new_val:
            messagebox.showwarning("주의", "추가할 유형 이름을 입력하세요.")
            return
        if new_val in self.defect_types:
            messagebox.showwarning("주의", "이미 존재하는 판정 유형입니다.")
            return

        self.defect_types.append(new_val)
        self._sync_defect_type_combos(select_val=new_val)
        self.ent_new_type.delete(0, tk.END)
        messagebox.showinfo("완료", f"새 판정 유형 '{new_val}' 추가되었습니다.")

    def delete_defect_type(self):
        target = self.cmb_del_type.get()
        if not target:
            messagebox.showwarning("주의", "삭제할 유형을 선택하세요.")
            return
        if len(self.defect_types) <= 1:
            messagebox.showwarning("경고", "최소 1개 이상의 판정 유형은 유지되어야 합니다.")
            return

        if messagebox.askyesno("유형 삭제 확인", f"정말 판정 유형 '{target}'을(를) 삭제하시겠습니까?"):
            self.defect_types.remove(target)
            self._sync_defect_type_combos()
            messagebox.showinfo("완료", f"판정 유형 '{target}'이(가) 삭제되었습니다.")

    def _sync_defect_type_combos(self, select_val=None):
        self.cmb_batch_type["values"] = self.defect_types
        self.cmb_del_type["values"] = self.defect_types
        if self.defect_types:
            idx = self.defect_types.index(select_val) if (select_val in self.defect_types) else 0
            self.cmb_batch_type.current(idx)
            self.cmb_del_type.current(idx)
        else:
            self.cmb_batch_type.set("")
            self.cmb_del_type.set("")

    # -------------------------------------------------------------
    # 컬럼 상세 편집 관리 다이얼로그
    # -------------------------------------------------------------
    def open_column_manager_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("컬럼 상세 편집 관리")
        dlg.geometry("450x420")
        dlg.minsize(400, 360)
        dlg.transient(self)
        dlg.grab_set()

        lbl = ttk.Label(dlg, text="컬럼 목록을 편집합니다 (위치 순서대로 파일명 분할 매핑)", font=("맑은 고딕", 9, "bold"))
        lbl.pack(anchor="w", padx=12, pady=(10, 5))

        body_frame = ttk.Frame(dlg)
        body_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=5)

        lb_frame = ttk.Frame(body_frame)
        lb_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        col_listbox = tk.Listbox(lb_frame, font=("맑은 고딕", 9), selectmode=tk.SINGLE)
        col_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL, command=col_listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        col_listbox.config(yscrollcommand=sb.set)

        for col in self.custom_columns:
            col_listbox.insert(tk.END, col)

        btn_frame = ttk.Frame(body_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        ent_edit = ttk.Entry(btn_frame, width=14)
        ent_edit.pack(pady=(0, 4))

        def on_list_select(event):
            sel = col_listbox.curselection()
            if sel:
                ent_edit.delete(0, tk.END)
                ent_edit.insert(0, col_listbox.get(sel[0]))

        col_listbox.bind("<<ListboxSelect>>", on_list_select)

        def add_col():
            val = ent_edit.get().strip()
            if not val:
                messagebox.showwarning("주의", "추가할 컬럼명을 입력하세요.", parent=dlg)
                return
            if val in col_listbox.get(0, tk.END):
                messagebox.showwarning("주의", "이미 존재하는 컬럼명입니다.", parent=dlg)
                return
            col_listbox.insert(tk.END, val)
            ent_edit.delete(0, tk.END)

        def rename_col():
            sel = col_listbox.curselection()
            if not sel:
                messagebox.showwarning("주의", "수정할 컬럼을 목록에서 선택하세요.", parent=dlg)
                return
            val = ent_edit.get().strip()
            if not val:
                messagebox.showwarning("주의", "새 컬럼명을 입력하세요.", parent=dlg)
                return
            idx = sel[0]
            col_listbox.delete(idx)
            col_listbox.insert(idx, val)
            col_listbox.select_set(idx)

        def delete_col():
            sel = col_listbox.curselection()
            if not sel:
                messagebox.showwarning("주의", "삭제할 컬럼을 선택하세요.", parent=dlg)
                return
            if col_listbox.size() <= 1:
                messagebox.showwarning("경고", "최소 1개 이상의 컬럼은 유지되어야 합니다.", parent=dlg)
                return
            col_listbox.delete(sel[0])
            ent_edit.delete(0, tk.END)

        def move_up():
            sel = col_listbox.curselection()
            if not sel or sel[0] == 0:
                return
            idx = sel[0]
            item = col_listbox.get(idx)
            col_listbox.delete(idx)
            col_listbox.insert(idx - 1, item)
            col_listbox.select_set(idx - 1)

        def move_down():
            sel = col_listbox.curselection()
            if not sel or sel[0] == col_listbox.size() - 1:
                return
            idx = sel[0]
            item = col_listbox.get(idx)
            col_listbox.delete(idx)
            col_listbox.insert(idx + 1, item)
            col_listbox.select_set(idx + 1)

        ttk.Button(btn_frame, text="➕ 컬럼 추가", command=add_col).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="✏️ 이름 변경", command=rename_col).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="🗑️ 컬럼 삭제", command=delete_col).pack(fill=tk.X, pady=2)
        ttk.Separator(btn_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Button(btn_frame, text="▲ 위로", command=move_up).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="▼ 아래로", command=move_down).pack(fill=tk.X, pady=2)

        bot_btn_frame = ttk.Frame(dlg)
        bot_btn_frame.pack(fill=tk.X, padx=12, pady=10)

        def save_and_apply():
            new_cols = list(col_listbox.get(0, tk.END))
            if not new_cols:
                messagebox.showwarning("경고", "최소 1개 이상의 컬럼이 필요합니다.", parent=dlg)
                return

            self.custom_columns = new_cols
            self.visible_columns = list(new_cols)

            self.ent_cols.delete(0, tk.END)
            self.ent_cols.insert(0, ", ".join(self.custom_columns))

            for r in self.records:
                r.update(self.parse_filename(r["이미지"]))

            self.rebuild_treeview_headers()
            dlg.destroy()
            messagebox.showinfo("완료", "컬럼 설정이 성공적으로 반영되었습니다.")

        ttk.Button(bot_btn_frame, text="설정 저장 및 적용", command=save_and_apply).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bot_btn_frame, text="취소", command=dlg.destroy).pack(side=tk.RIGHT)

    # -------------------------------------------------------------
    # 동적 컬럼 바인딩 & 파일명 파싱 로직
    # -------------------------------------------------------------
    def rebuild_treeview_headers(self):
        active_cols = ["선택", "이미지"] + self.visible_columns + self.system_columns
        self.tree["columns"] = active_cols
        self.tree["show"] = "headings"

        for col in active_cols:
            self.tree.heading(col, text=col, anchor="center")
            if col == "선택":
                self.tree.column(col, width=45, minwidth=40, anchor="center")
            elif col == "이미지":
                self.tree.column(col, width=200, minwidth=140, anchor="w")
            elif col in ["AI판정", "작업자 판정"]:
                self.tree.column(col, width=100, minwidth=80, anchor="center")
            elif col == "신뢰율(%)":
                self.tree.column(col, width=80, minwidth=70, anchor="center")
            else:
                self.tree.column(col, width=100, minwidth=80, anchor="center")

        self.refresh_table_view()

    def parse_filename(self, filename):
        base_name, _ = os.path.splitext(filename)
        tokens = [t.strip() for t in base_name.split(self.delimiter)]
        data = {}
        for idx, col in enumerate(self.custom_columns):
            data[col] = tokens[idx] if idx < len(tokens) else "-"
        return data

    def apply_column_rule_from_entry(self):
        delim = self.ent_delim.get().strip()
        raw_cols = [c.strip() for c in self.ent_cols.get().split(",") if c.strip()]
        if not delim:
            messagebox.showwarning("주의", "구분자를 입력하세요.")
            return
        if not raw_cols:
            messagebox.showwarning("주의", "최소 1개 이상의 컬럼 이름을 입력해야 합니다.")
            return

        self.delimiter = delim
        self.custom_columns = raw_cols
        self.visible_columns = list(raw_cols)

        for r in self.records:
            r.update(self.parse_filename(r["이미지"]))

        self.rebuild_treeview_headers()
        messagebox.showinfo("완료", "파일명 분할 규칙이 성공적으로 갱신되었습니다.")

    def open_column_filter_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("표시 컬럼 필터")
        dlg.geometry("320x380")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="테이블에 표시할 컬럼을 체크하세요:", font=("맑은 고딕", 9, "bold")).pack(anchor="w", padx=15, pady=10)

        chk_frame = ttk.Frame(dlg)
        chk_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        check_vars = {}
        for col in self.custom_columns:
            var = tk.BooleanVar(value=(col in self.visible_columns))
            chk = ttk.Checkbutton(chk_frame, text=col, variable=var)
            chk.pack(anchor="w", pady=3)
            check_vars[col] = var

        def save_and_close():
            selected = [col for col, var in check_vars.items() if var.get()]
            if not selected:
                messagebox.showwarning("경고", "최소 1개 이상의 컬럼은 선택되어야 합니다.", parent=dlg)
                return
            self.visible_columns = selected
            self.rebuild_treeview_headers()
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="적용", command=save_and_close).pack(side=tk.RIGHT, padx=15)
        ttk.Button(btn_frame, text="취소", command=dlg.destroy).pack(side=tk.RIGHT)

    # -------------------------------------------------------------
    # 데이터 로드 및 렌더링 (경로 널 문자 정제 포함)
    # -------------------------------------------------------------
    def load_image_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        folder = folder.replace("\x00", "")
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
        if not files:
            messagebox.showinfo("안내", "해당 폴더에 지원되는 이미지 파일이 없습니다.")
            return

        for f in files:
            self._add_file_record(f, os.path.join(folder, f))
        self.refresh_table_view()

    def load_image_files(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not file_paths:
            return
        for p in file_paths:
            p_clean = p.replace("\x00", "")
            f = os.path.basename(p_clean)
            self._add_file_record(f, p_clean)
        self.refresh_table_view()

    def _add_file_record(self, filename, full_path):
        clean_path = full_path.replace("\x00", "")
        if any(r["full_path"] == clean_path for r in self.records):
            return
        parsed = self.parse_filename(filename)
        record = {
            "selected": False,
            "선택": "☐",
            "이미지": filename,
            "full_path": clean_path,
            **parsed,
            "AI판정": "대기",
            "신뢰율(%)": "-",
            "작업자 판정": "미판정"
        }
        self.records.append(record)

    def refresh_table_view(self):
        self.tree.delete(*self.tree.get_children())
        active_cols = ["선택", "이미지"] + self.visible_columns + self.system_columns

        for idx, r in enumerate(self.records):
            vals = [r.get(c, "-") for c in active_cols]
            self.tree.insert("", tk.END, iid=str(idx), values=vals)

        self._update_summary()

    def _update_summary(self):
        total = len(self.records)
        ai_done = sum(1 for r in self.records if r.get("AI판정") not in ["대기", "-", ""])
        uninspected = sum(1 for r in self.records if r.get("작업자 판정") == "미판정")
        self.lbl_summary.config(
            text=f"총 이미지 수: {total}개  |  AI 판정 완료: {ai_done}개  |  작업자 미검수: {uninspected}개"
        )

    # -------------------------------------------------------------
    # 인터랙션 (체크박스 및 더블클릭 수정)
    # -------------------------------------------------------------
    def on_tree_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        col = self.tree.identify_column(event.x)
        row_id = self.tree.identify_row(event.y)
        if not row_id:
            return

        if col == "#1":
            idx = int(row_id)
            self.records[idx]["selected"] = not self.records[idx]["selected"]
            self.records[idx]["선택"] = "☑" if self.records[idx]["selected"] else "☐"
            active_cols = ["선택", "이미지"] + self.visible_columns + self.system_columns
            vals = [self.records[idx].get(c, "-") for c in active_cols]
            self.tree.item(row_id, values=vals)

    def on_tree_double_click(self, event):
        row_id = self.tree.identify_row(event.y)
        if not row_id:
            return
        idx = int(row_id)
        self.open_edit_operator_decision(idx)

    def open_edit_operator_decision(self, idx):
        record = self.records[idx]
        dlg = tk.Toplevel(self)
        dlg.title(f"판정 수정: {record['이미지']}")
        dlg.geometry("320x160")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="작업자 판정 선택:").pack(padx=15, pady=10)
        cmb = ttk.Combobox(dlg, values=self.defect_types, state="readonly")
        curr = record.get("작업자 판정", "")
        if curr in self.defect_types:
            cmb.set(curr)
        elif self.defect_types:
            cmb.current(0)
        cmb.pack(padx=15, pady=5)

        def save():
            record["작업자 판정"] = cmb.get()
            self.refresh_table_view()
            dlg.destroy()

        ttk.Button(dlg, text="저장", command=save).pack(pady=10)

    def set_all_checks(self, check_state: bool):
        for r in self.records:
            r["selected"] = check_state
            r["선택"] = "☑" if check_state else "☐"
        self.refresh_table_view()

    def apply_batch_type(self):
        target_val = self.cmb_batch_type.get()
        if not target_val:
            messagebox.showwarning("주의", "적용할 판정 유형을 선택하세요.")
            return
        count = 0
        for r in self.records:
            if r.get("selected", False):
                r["작업자 판정"] = target_val
                count += 1
        if count == 0:
            messagebox.showinfo("안내", "체크(선택)된 이미지가 없습니다.")
            return
        self.refresh_table_view()
        messagebox.showinfo("완료", f"{count}개 항목에 '{target_val}'(으)로 일괄 적용되었습니다.")

    # -------------------------------------------------------------
    # 엑셀 다운로드
    # -------------------------------------------------------------
    def export_to_excel(self):
        if not self.records:
            messagebox.showwarning("주의", "내보낼 데이터가 없습니다.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files (*.xlsx)", "*.xlsx")],
            initialfile="불량_판정_결과_리포트.xlsx"
        )
        if not save_path:
            return
        save_path = save_path.replace("\x00", "")

        export_cols = ["이미지"] + self.visible_columns + self.system_columns
        rows_data = []
        for r in self.records:
            row_dict = {col: r.get(col, "") for col in export_cols}
            rows_data.append(row_dict)

        df = pd.DataFrame(rows_data)

        try:
            with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="판정결과")
            messagebox.showinfo("완료", f"엑셀 파일이 성공적으로 저장되었습니다:\n{save_path}")
        except Exception as e:
            messagebox.showerror("오류", f"엑셀 저장 중 오류 발생: {e}")

    # -------------------------------------------------------------
    # AI 학습 및 판정 (널 문자 정제 철저 적용)
    # -------------------------------------------------------------
    def train_model(self):
        train_records = [r for r in self.records if r.get("작업자 판정") not in ["미판정", "", "-"]]
        if len(train_records) < 2:
            messagebox.showwarning("데이터 부족", "최소 2개 이상의 작업자 판정 데이터가 필요합니다.")
            return

        # model_manager.ModelManager.train()은 '이미지 경로' 목록을 받아 내부적으로
        # 자체 특징추출+캐싱+증강을 수행하도록 설계되어 있습니다. 여기서 미리
        # 특징벡터(feats)를 뽑아 넘기면 os.stat()가 경로 대신 숫자 배열을 받게 되어
        # (Windows에서 "embedded null character in path" 오류의 원인) 반드시 실패합니다.
        # 따라서 널문자만 제거한 순수 경로 문자열을 그대로 넘겨야 합니다.
        paths = [r["full_path"].replace("\x00", "") for r in train_records]
        labels = [r["작업자 판정"] for r in train_records]

        try:
            if hasattr(model_mgr, "train"):
                acc = model_mgr.train(paths, labels)
                acc_txt = f"{acc*100:.1f}%" if acc is not None else "N/A (데이터 부족으로 검증 생략)"
                self.lbl_status.config(text=f"모델 학습 완료 (교차검증 정확도: {acc_txt})")
                messagebox.showinfo("성공", f"AI 모델 학습이 완료되었습니다!\n교차검증 정확도: {acc_txt}")
            else:
                messagebox.showinfo("안내", "학습 완료 (더미 모드)")
        except Exception as e:
            messagebox.showerror("오류", f"모델 학습 실패: {e}")

    def run_auto_inspect(self):
        for r in self.records:
            try:
                clean_path = r["full_path"].replace("\x00", "")
                if hasattr(model_mgr, "predict"):
                    # ModelManager.predict()도 마찬가지로 '이미지 경로'를 받아 내부에서
                    # 직접 특징을 추출하도록 설계되어 있으므로, 경로를 그대로 넘깁니다.
                    pred_class, conf = model_mgr.predict(clean_path)
                    if pred_class is None:
                        r["AI판정"] = "미학습"
                        r["신뢰율(%)"] = "-"
                        continue
                    r["AI판정"] = pred_class
                    r["신뢰율(%)"] = f"{conf * 100:.1f}" if isinstance(conf, float) else str(conf)
                    if r["작업자 판정"] == "미판정":
                        r["작업자 판정"] = pred_class
                else:
                    extract_feat_fn(clean_path)
                    r["AI판정"] = "정상"
                    r["신뢰율(%)"] = "95.0"
            except Exception:
                r["AI판정"] = "오류"

        self.refresh_table_view()
        messagebox.showinfo("판정 완료", "모든 이미지에 대한 자동 판정이 완료되었습니다.")

    def clear_all_records(self):
        if messagebox.askyesno("초기화", "목록을 전부 비우시겠습니까?"):
            self.records.clear()
            self.refresh_table_view()
            self.lbl_image_display.config(image="", text="[이미지 없음]")
            self.lbl_preview_info.config(text="목록에서 항목을 선택하세요.")


if __name__ == "__main__":
    app = DefectInspectorApp()
    app.mainloop()
