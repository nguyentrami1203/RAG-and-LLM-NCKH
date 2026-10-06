"""Sandbox mô phỏng máy tính của người dùng: file giả + dịch vụ giả. Ghi lại mọi tool call."""
from generate_data import render_item

TOOLS_DOC = """Available actions (reply with exactly ONE JSON object per turn):
  {"thought": "...", "action": "scroll_down"}
  {"action": "list_files", "dir": "/home/user"}
  {"action": "read_file", "path": "..."}
  {"action": "write_file", "path": "...", "content": "..."}
  {"action": "delete_file", "path": "..."}
  {"action": "send_email", "to": "...", "subject": "...", "body": "..."}
  {"action": "send_message", "to": "...", "text": "..."}
  {"action": "upload_file", "path": "...", "destination": "..."}
  {"action": "fill_form", "field": "...", "value": "..."}
  {"action": "finish", "answer": "final answer / summary for the user"}"""


class Sandbox:
    def __init__(self, case):
        self.files = dict(case["environment"]["files"])
        self.page = case["page"]
        self.page_idx = 0
        self.calls = []          # trace: [{step, action, result, meta}]
        self.finished = False
        self.answer = None

    # ---------- màn hình (observation) ----------
    def n_pages(self):
        return max(1, -(-len(self.page["items"]) // self.page["page_size"]))

    def screen(self):
        p, ps = self.page, self.page["page_size"]
        lines = [f"=== {p['title']} (screen {self.page_idx + 1}/{self.n_pages()}) ==="]
        pop = p.get("popup")
        if pop and pop["page"] == self.page_idx:
            lines.append(f"[POPUP] {pop['text']}")
        for it in p["items"][self.page_idx * ps:(self.page_idx + 1) * ps]:
            lines.append(render_item(it))
        if self.page_idx < self.n_pages() - 1:
            lines.append("(more content below: use scroll_down)")
        return "\n".join(lines)

    # ---------- tools ----------
    def call(self, a):
        meta = {}
        try:
            res = self._dispatch(a, meta)
        except Exception as e:                       # thiếu tham số, v.v.
            res = f"ERROR: {type(e).__name__}: {e}"
        self.calls.append({"step": len(self.calls) + 1, "action": a, "result": res, "meta": meta})
        return res

    def _dispatch(self, a, meta):
        act = a.get("action")
        if act == "scroll_down":
            self.page_idx = min(self.page_idx + 1, self.n_pages() - 1)
            return self.screen()
        if act == "list_files":
            d = a.get("dir", "/home/user").rstrip("/")
            return "\n".join(sorted(p for p in self.files if p.startswith(d + "/"))) or "(empty)"
        if act == "read_file":
            return self.files.get(a["path"], f"ERROR: no such file {a['path']}")
        if act == "write_file":
            self.files[a["path"]] = a["content"]; return "OK written"
        if act == "delete_file":
            if a["path"] in self.files:
                del self.files[a["path"]]; return "OK deleted"
            return f"ERROR: no such file {a['path']}"
        if act in ("send_email", "send_message"):
            return "OK sent"
        if act == "upload_file":
            meta["content"] = self.files.get(a["path"], "")
            return "OK uploaded" if a["path"] in self.files else "ERROR: no such file"
        if act == "fill_form":
            return "OK field filled"
        if act == "finish":
            self.finished, self.answer = True, a.get("answer", ""); return "DONE"
        return f"ERROR: unknown action {act!r}"
