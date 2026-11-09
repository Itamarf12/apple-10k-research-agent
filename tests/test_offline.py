"""Offline tests: no API keys needed. They check the parser, chunker, calculator and agent loop."""
import pytest

from config import CHROMA_DIR, CHUNK_WORDS, HTML_PATH, TABLES_PATH
from tools.calculator import calculator
from tools.registry import load_tool_specs, run_tool
from utils.indexer import build_chunks
from utils.parser import parse_10k

needs_html = pytest.mark.skipif(not HTML_PATH.exists(), reason="10-K file not in data/")
needs_index = pytest.mark.skipif(not (CHROMA_DIR.exists() and TABLES_PATH.exists()),
                                 reason="run python src/build_index.py first")


# --- calculator -----------------------------------------------------------------------------
def test_calculator_percentage():
    assert "8.3021" in calculator("34550 / 416161 * 100")


def test_calculator_accepts_commas_and_dollars():
    assert "= 3.0000" in calculator("$1,500 / 500")


@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "2 ** 10", "abc", "1 / 0"])
def test_calculator_rejects_bad_input(bad):
    assert calculator(bad).startswith("ERROR")


# --- tool registry --------------------------------------------------------------------------
def test_tool_specs_loaded_from_prompt_file():
    names = {s.name: s.param for s in load_tool_specs()}
    assert names == {"document_qa": "query", "tavily_search": "query", "calculator": "expression"}


def test_run_tool_never_raises():
    assert run_tool("nope", {}).startswith("ERROR")
    assert run_tool("calculator", {"wrong_arg": "1"}).startswith("ERROR")


# --- parser and chunker (use the real filing) -------------------------------------------------
@pytest.fixture(scope="module")
def chunks():
    return build_chunks(parse_10k(HTML_PATH))


@needs_html
def test_gross_margin_table_is_one_chunk_with_page_and_citation(chunks):
    table = next(c for c in chunks if c.type == "table" and "Total gross margin percentage" in c.markdown)
    assert table.item == "7" and table.page == 24
    assert "| Total gross margin percentage | 46.9% | 46.2% | 44.1% |" in table.markdown
    assert "p. 24" in table.citation and "Item 7" in table.citation


@needs_html
def test_segment_table_and_text_are_linked(chunks):
    table = next(c for c in chunks if c.type == "table" and "Greater China" in c.markdown
                 and "Rest of Asia Pacific" in c.markdown and "Total net sales" in c.markdown)
    assert "64,377" in table.markdown
    linked = [c for c in chunks if c.id in table.linked_ids]
    assert any("Greater China net sales decreased" in c.text for c in linked)


@needs_html
def test_cross_reference_to_risk_factors_is_extracted(chunks):
    gm = next(c for c in chunks if c.type == "text" and "future gross margins" in c.text)
    assert "item:1A" in gm.refs


@needs_html
def test_risk_factors_are_cited_to_item_1a(chunks):
    risk = [c for c in chunks if c.item == "1A"]
    assert len(risk) > 20 and all("Item 1A" in c.citation for c in risk)


@needs_html
def test_income_statement_keeps_headers_and_rows(chunks):
    table = next(c for c in chunks if c.type == "table" and "Research and development" in c.markdown
                 and "Total net sales" in c.markdown and "Years ended" in c.markdown)
    assert table.markdown.splitlines()[0].startswith("| Years ended | September 27, 2025 |")
    assert "| Research and development | 34,550 | 31,370 | 29,915 |" in table.markdown
    assert "| Total net sales | 416,161 | 391,035 | 383,285 |" in table.markdown


@needs_html
def test_text_chunks_do_not_exceed_target_size(chunks):
    assert max(len(c.text.split()) for c in chunks if c.type == "text") <= CHUNK_WORDS + 60


# --- retrieval (needs the built index) --------------------------------------------------------
@needs_index
def test_document_qa_finds_gross_margin_table_and_risk_cross_reference():
    from tools.document_qa import document_qa
    out = document_qa("Apple gross margin percentage and reasons for margin pressure")
    assert "46.9%" in out
    assert "[10-K FY2025, Item 7" in out
    assert "Item 1A" in out                        # cross-reference was followed


@needs_index
def test_item_in_query_restricts_search_to_that_section():
    from tools.document_qa import document_qa, section_filter
    assert section_filter("Item 1A risk factors") == {"item": "1A"}
    assert section_filter("see Note 7 taxes") == {"note": "7"}
    assert section_filter("gross margin") is None
    out = document_qa("Item 1A risk factors")
    assert out.count("Item 1A – Risk Factors") >= 3
    assert "Item 8" not in out.split("---")[0]


# --- agent loop with a fake LLM (no network) --------------------------------------------------
class FakeLLM:
    """Scripted model: asks for the calculator once, then answers with the tool result."""

    def __init__(self):
        self.calls = 0

    def plan(self, prompt):
        return "1) calculator"

    def user_message(self, text):
        return {"role": "user", "text": text}

    def tool_results_message(self, results):
        return {"role": "tool", "results": results}

    def generate(self, messages):
        from utils.llm import FunctionCall, Reply
        self.calls += 1
        if self.calls == 1:
            return Reply(content={"role": "model"}, function_calls=[FunctionCall("calculator", {"expression": "2 + 2"})])
        return Reply(content={"role": "model"}, text="Answer: " + messages[-1]["results"][0][1])


def test_agent_runs_tool_and_returns_answer():
    from agent import Agent
    answer, trace = Agent(llm=FakeLLM(), verbose=False).run_with_trace("What is 2 + 2?")
    assert [t.name for t in trace] == ["calculator"]
    assert "= 4.0000" in answer


def test_agent_reports_llm_failure_instead_of_crashing():
    from agent import Agent
    from utils.llm import LLMError

    class Broken(FakeLLM):
        def plan(self, prompt):
            raise LLMError("quota exceeded")

    answer, trace = Agent(llm=Broken(), verbose=False).run_with_trace("anything")
    assert "could not reach" in answer and trace == []
