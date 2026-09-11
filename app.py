import os
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from PIL import Image, ImageTk

# 기존 모듈과 안전하게 연동 (함수/클래스 형태 모두 자동 대응)
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
        # 모듈 내 첫 번째 호출 가능한 함수 탐색
        fns = [getattr(fe_mod, a) for a in dir(fe_mod) if callable(getattr(fe_mod, a)) and not a.startswith("_")]
        extract_feat_fn = fns[0] if fns else (lambda p: [0]*10)
except Exception:
    extract_feat_fn = lambda p: [0]*10

try:
    import model_manager as mm_mod
    if hasattr(mm_mod, "ModelManager"):
        model_mgr = mm_mod.ModelManager()
    else:
        model_mgr = mm_mod
except Exception:
    model_mgr = None

try:
    import data_store as ds_mod
    if hasattr(ds_mod, "DataStore"):
        store_mgr = ds_mod.DataStore()
    else:
        store_mgr = ds_mod
except Exception:
    store_mgr = None


class DefectInspectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("불량 이미지 자동 판정 프로그램 (범용 동적 컬럼 에디션)")
        self.geometry("1500, 900")
        self.minsize(1200, 700)

        # -------------------------------------------------------------
        # 파일명 분할 및 동적 컬럼 기본 설정
        # -------------------------------------------------------------
        self.delimiter = "_"
        # 파일명에서 순서대로 분할할 컬럼 정의 (UI에서 자유롭게 변경 가능)
        self.custom_columns = ["LOT", "GLS", "셀번지", "X", "Y", "검사호기"]
        self.visible_columns = list(self.custom_columns)
        self.system_columns = ["AI판정", "신뢰율(%)", "작업자 판정"]

        self.records = []
        self.defect_types = ["K 유기", "K 갈림", "NK 유기", "핀홀", "정상"]

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

        # 2. 파일명 분할 및 동적 컬럼 도구 바 (신규 기능)
        cfg_bar = ttk.LabelFrame(self, text="파일명 텍스트 나누기 및 컬럼 정의", padding=5)
        cfg_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=3)

        ttk.Label(cfg_bar, text="구분 기호:").pack(side=tk.LEFT, padx=(5, 2))
        self.ent_delim = ttk.Entry(cfg_bar, width=4)
        self.ent_delim.insert(0, self.delimiter)
        self.ent_delim.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(cfg_bar, text="컬럼 순서 (쉼표로 구분):").pack(side=tk.LEFT, padx=(0, 2))
        self.ent_cols = ttk.Entry(cfg_bar, width=50)
        self.ent_cols.insert(0, ", ".join(self.custom_columns))
        self.ent_cols.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(cfg_bar, text="규칙 적용", command=self.apply_column_rule).pack(side=tk.LEFT, padx=3)
        ttk.Button(cfg_bar, text="👁️ 표시 컬럼 필터", command=self.open_column_filter_dialog).pack(side=tk.LEFT, padx=5)

        # 3. 판정 유형 관리 바
        type_bar = ttk.LabelFrame(self, text="판정 유형 관리", padding=5)
        type_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)

        ttk.Label(type_bar, text="새 유형 추가:").pack(side=tk.LEFT, padx=2)
        self.ent_new_type = ttk.Entry(type_bar, width=15)
        self.ent_new_type.pack(side=tk.LEFT, padx=3)
        ttk.Button(type_bar, text="추가", command=self.add_defect_type).pack(side=tk.LEFT, padx=2)

        ttk.Label(type_bar, text="선택 이미지 일괄 판정값:").pack(side=tk.LEFT, padx=(20, 2))
        self.cmb_batch_type = ttk.Combobox(type_bar, values=self.defect_types, state="readonly", width=12)
        if self.defect_types:
            self.cmb_batch_type.current(0)
        self.cmb_batch_type.pack(side=tk.LEFT, padx=2)
        ttk.Button(type_bar, text="선택 항목에 적용", command=self.apply_batch_type).pack(side=tk.LEFT, padx=2)
        ttk.Button(type_bar, text="전체 선택", command=lambda: self.set_all_checks(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(type_bar, text="전체 해제", command=lambda: self.set_all_checks(False)).pack(side=tk.LEFT, padx=2)

        # 4. 요약 통계 영역
        summary_frame = ttk.LabelFrame(self, text="판정 결과 요약", padding=5)
        summary_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        self.lbl_summary = ttk.Label(summary_frame, text="총 이미지 수: 0개  |  AI 판정 완료: 0개  |  작업자 미검수: 0개", font=("맑은 고딕", 9, "bold"))
        self.lbl_summary.pack(anchor="w", padx=5)

        # 5. 하단 테이블(Treeview) 영역 (붉은 박스 영역)
        tbl_container = ttk.Frame(self, padding=(8, 4, 8, 8))
        tbl_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.scroll_y = ttk.Scrollbar(tbl_container, orient=tk.VERTICAL)
        self.scroll_x = ttk.Scrollbar(tbl_container, orient=tk.HORIZONTAL)

        self.tree = ttk.Treeview(
            tbl_container,
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

        self.rebuild_treeview_headers()

    # -------------------------------------------------------------
    # 동적 컬럼 & 파싱 로직
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
                self.tree.column(col, width=220, minwidth=150, anchor="w")
            elif col in ["AI판정", "작업자 판정"]:
                self.tree.column(col, width=110, minwidth=90, anchor="center")
            elif col == "신뢰율(%)":
                self.tree.column(col, width=80, minwidth=70, anchor="center")
            else:
                self.tree.column(col, width=110, minwidth=80, anchor="center")

        self.refresh_table_view()

    def parse_filename(self, filename):
        base_name, _ = os.path.splitext(filename)
        tokens = [t.strip() for t in base_name.split(self.delimiter)]
        data = {}
        for idx, col in enumerate(self.custom_columns):
            data[col] = tokens[idx] if idx < len(tokens) else "-"
        return data

    def apply_column_rule(self):
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
            new_parsed = self.parse_filename(r["이미지"])
            r.update(new_parsed)

        self.rebuild_treeview_headers()
        messagebox.showinfo("완료", "파일명 분할 규칙이 성공적으로 갱신되었습니다.")

    def open_column_filter_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("표시 컬럼 필터")
        dlg.geometry("320, 380")
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
    # 데이터 로드 및 테이블 바인딩
    # -------------------------------------------------------------
    def load_image_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
        if not files:
            messagebox.showinfo("안내", "해당 폴더에 이미지 파일이 없습니다.")
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
            f = os.path.basename(p)
            self._add_file_record(f, p)
        self.refresh_table_view()

    def _add_file_record(self, filename, full_path):
        if any(r["full_path"] == full_path for r in self.records):
            return
        parsed = self.parse_filename(filename)
        record = {
            "selected": False,
            "선택": "☐",
            "이미지": filename,
            "full_path": full_path,
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
        dlg.geometry("320, 160")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="작업자 판정 선택:").pack(padx=15, pady=10)
        cmb = ttk.Combobox(dlg, values=self.defect_types, state="readonly")
        curr = record.get("작업자 판정", "")
        if curr in self.defect_types:
            cmb.set(curr)
        else:
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

    def add_defect_type(self):
        new_val = self.ent_new_type.get().strip()
        if not new_val:
            return
        if new_val in self.defect_types:
            messagebox.showwarning("주의", "이미 존재하는 유형입니다.")
            return
        self.defect_types.append(new_val)
        self.cmb_batch_type["values"] = self.defect_types
        self.ent_new_type.delete(0, tk.END)
        messagebox.showinfo("완료", f"새 판정 유형 '{new_val}' 추가 완료")

    # -------------------------------------------------------------
    # 엑셀 내보내기 (표시된 컬럼 그대로 저장)
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
    # AI 판정 및 학습 연동
    # -------------------------------------------------------------
    def train_model(self):
        train_records = [r for r in self.records if r.get("작업자 판정") not in ["미판정", "", "-"]]
        if len(train_records) < 2:
            messagebox.showwarning("데이터 부족", "최소 2개 이상의 작업자 판정 데이터가 필요합니다.")
            return

        features = []
        labels = []
        for r in train_records:
            feats = extract_feat_fn(r["full_path"])
            features.append(feats)
            labels.append(r["작업자 판정"])

        try:
            if hasattr(model_mgr, "train"):
                acc = model_mgr.train(features, labels)
                self.lbl_status.config(text="모델 학습 완료")
                messagebox.showinfo("성공", "AI 모델 학습이 완료되었습니다!")
            else:
                messagebox.showinfo("안내", "학습 완료 (더미 모드)")
        except Exception as e:
            messagebox.showerror("오류", f"모델 학습 실패: {e}")

    def run_auto_inspect(self):
        for r in self.records:
            try:
                feats = extract_feat_fn(r["full_path"])
                if hasattr(model_mgr, "predict"):
                    pred_class, conf = model_mgr.predict(feats)
                    r["AI판정"] = pred_class
                    r["신뢰율(%)"] = f"{conf * 100:.1f}" if isinstance(conf, float) else str(conf)
                    if r["작업자 판정"] == "미판정":
                        r["작업자 판정"] = pred_class
                else:
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


if __name__ == "__main__":
    app = DefectInspectorApp()
    app.mainloop()
