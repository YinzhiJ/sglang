import os
import threading
from typing import List, Optional

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.utils.common import utc_now_iso_ms, Logger

from app.tools.http_search import HttpSearchTool

class Retriever:

    _lock = threading.Lock()

    def __init__(self, tokenizer=None, log_dir:str="./logs"):
        self.http_search = HttpSearchTool()
        self.tokenizer = tokenizer
        self.log_dir = log_dir

    def run(self, batch_reqs: Optional[List[Req]] = None) -> List[str]: 
        """
        - 根据当前上下文构造 query
        - 调你自己的 HTTP RAG 服务
        - 返回检索到的一小段文本
        """
        contexts: List[str] = []

        if batch_reqs is None:
            return contexts

        for req in batch_reqs:
            start_time = utc_now_iso_ms()
            partial_output = self.tokenizer.decode(req.output_ids, skip_special_tokens=True)
            query = self._make_query(req.origin_input_text, partial_output)
            resp = self.http_search.run({
                "query": query,
                "limit": 3,
            })
            snippets = []
            if resp.ok:
                snippets = [e.snippet for e in resp.evidence]
                contexts.append("\n\n---\n\n".join(snippets))
            else:
                contexts.append("")
            end_time = utc_now_iso_ms()
            self._log_rag_injection("rag-" + req.eid, start_time, end_time, query, snippets)
        return contexts

    def _log_rag_injection(self, event_id, start_time, end_time, query, snippets):
        event = {
            "event_id": event_id,
            "start_time": start_time,
            "end_time": end_time,
            "query": query,
            "snippets": snippets,
        }
        log_path = os.path.join(self.log_dir, f"{event_id}.jsonl")
        Logger.append_jsonl(event, log_path, True)

    def _make_query(self, origin_input: str, partial_output: str) -> str:
        if partial_output:
            return f"{origin_input}\nCurrent answer so far:\n{partial_output}"
        return origin_input