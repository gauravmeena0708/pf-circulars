import { I18nProvider, useI18n } from "./UiLanguage.jsx";
import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import { createRoot } from "react-dom/client";
import { AlertCircle, LoaderCircle, X } from "lucide-react";
import "./style.css";
import "./friendly.css";
import { Assistant, PdfTools, SessionSettings } from "./Workspace";
import Divisions from "./Divisions";
import HistoryPage from "./HistoryPage.jsx";
import { PageHeading, Sidebar, Topbar } from "./Layout.jsx";
import LibraryPage from "./LibraryPage.jsx";
import ResearchComposer from "./ResearchComposer.jsx";
import ResearchResults from "./ResearchResults.jsx";
import { headers, request as fetchWorkspace, saveResponse } from "./client";
import { api, download, post } from "./lib";

// App holds the state shared between pages; each page's markup lives in its own file.
function App() {
  const { t: tr } = useI18n();
  const [catalog, setCatalog] = useState(null),
    [page, setPage] = useState("research"),
    [menu, setMenu] = useState(false),
    [query, setQuery] = useState(""),
    [policy, setPolicy] = useState("auto"),
    [queryTopic, setQueryTopic] = useState(""),
    [day, setDay] = useState(new Date().toLocaleDateString("en-CA")),
    [language, setLanguage] = useState("English"),
    [detail, setDetail] = useState("brief");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [result, setResult] = useState(null),
    [answer, setAnswer] = useState(""),
    [generating, setGenerating] = useState(false),
    [source, setSource] = useState(null),
    [sourceOpen, setSourceOpen] = useState(false),
    [resultTab, setResultTab] = useState("answer"),
    [recent, setRecent] = useState([]),
    [history, setHistory] = useState(null),
    [selectedHistory, setSelectedHistory] = useState("jd"),
    [historyTopic, setHistoryTopic] = useState(""),
    [hasToken, setHasToken] = useState(false),
    [librarySort, setLibrarySort] = useState("latest"),
    [divisionFocus, setDivisionFocus] = useState("");
  const requestSeq = useRef(0),
    answerAbort = useRef(null),
    researchOptions = useRef(null),
    input = useRef(null);
  useEffect(() => {
    api("/api/catalog")
      .then((c) => {
        setCatalog(c);
        setDay(c.today);
      })
      .catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setPage("research");
        setTimeout(() => input.current?.focus(), 0);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  useEffect(() => {
    if (page !== "history") return;
    let live = true;
    setHistory(null);
    api(
      `/api/policies/${selectedHistory}?as_of=${day}${historyTopic ? "&topic=" + encodeURIComponent(historyTopic) : ""}`,
    )
      .then((h) => {
        if (live) setHistory(h);
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, [page, selectedHistory, historyTopic, day]);
  async function search(
    e,
    question = query,
    mode = policy,
    documentsOnly = false,
  ) {
    e?.preventDefault();
    documentsOnly ||= e?.nativeEvent?.submitter?.value === "documents";
    if (question.trim().length < 3) return;
    const seq = ++requestSeq.current;
    answerAbort.current?.abort();
    setGenerating(false);
    setBusy(true);
    setError("");
    setAnswer("");
    setResult(null);
    setSource(null);
    setSourceOpen(false);
    setResultTab("answer");
    setPage("research");
    try {
      const r = await api(
        "/api/search",
        post({
          question: question.trim(),
          policy: mode,
          topic: (
            catalog?.policies.find((p) => p.id === mode)?.topics || []
          ).some((t) => t.id === queryTopic)
            ? queryTopic
            : null,
          as_of: day,
        }),
      );
      if (seq !== requestSeq.current) return;
      const packet = { ...r, selection_mode: mode };
      setResult(packet);
      setSource(r.sources[0] || null);
      if (documentsOnly || !(catalog?.answer_enabled || hasToken))
        setResultTab("sources");
      setRecent((v) =>
        [question, ...v.filter((x) => x !== question)].slice(0, 5),
      );
      setBusy(false);
      if (
        !documentsOnly &&
        r.sources.length &&
        (catalog?.answer_enabled || hasToken)
      ) {
        await generate(packet, seq);
      }
    } catch (e) {
      if (seq === requestSeq.current) setError(e.message);
    } finally {
      if (seq === requestSeq.current) setBusy(false);
    }
  }
  async function generate(packet = result, seq = requestSeq.current) {
    if (!packet || seq !== requestSeq.current) return;
    answerAbort.current?.abort();
    const controller = new AbortController();
    answerAbort.current = controller;
    setGenerating(true);
    setAnswer("");
    setError("");
    try {
      const r = await fetch("/api/answers/" + packet.packet_id, {
        ...post({ language, detail }),
        headers: headers({ "Content-Type": "application/json" }),
        signal: controller.signal,
      });
      if (!r.ok) {
        const d = await r.json();
        throw Error(d.detail);
      }
      const reader = r.body.getReader(),
        decoder = new TextDecoder();
      let pending = "";
      while (true) {
        const { done, value } = await reader.read();
        pending += decoder.decode(value || new Uint8Array(), { stream: !done });
        let lines = pending.split("\n");
        pending = lines.pop();
        for (const line of lines) {
          if (!line) continue;
          const item = JSON.parse(line);
          if (item.error) throw Error(item.error);
          if (
            item.text &&
            seq === requestSeq.current &&
            !controller.signal.aborted
          )
            setAnswer((v) => v + item.text);
        }
        if (done) break;
      }
    } catch (e) {
      if (e.name !== "AbortError" && seq === requestSeq.current)
        setError(e.message);
    } finally {
      if (seq === requestSeq.current && answerAbort.current === controller)
        setGenerating(false);
    }
  }
  function newQuestion() {
    ++requestSeq.current;
    answerAbort.current?.abort();
    if (researchOptions.current) researchOptions.current.open = false;
    setBusy(false);
    setGenerating(false);
    setQuery("");
    setAnswer("");
    setResult(null);
    setSource(null);
    setSourceOpen(false);
    setError("");
    setPolicy("auto");
    setQueryTopic("");
    if (catalog?.today) setDay(catalog.today);
    setPage("research");
    setMenu(false);
    setTimeout(() => input.current?.focus(), 0);
  }
  const selectSource = useCallback(
    (id) => {
      const selectedSource = result?.sources.find((s) => s.id === Number(id));
      if (selectedSource) {
        setSource(selectedSource);
        setSourceOpen(true);
      }
    },
    [result],
  );
  // Keep citation elements mounted while the drawer opens so native dialog
  // focus restoration returns to the link that was selected.
  const answerComponents = useMemo(
    () => ({
      a: ({ href, children }) =>
        href?.startsWith("#source-") ? (
          <button
            className="citation"
            onClick={() => selectSource(href.slice(8))}
          >
            {children}
          </button>
        ) : (
          <a href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        ),
    }),
    [selectSource],
  );
  const context = result?.policy_context,
    policies = catalog?.policies || [],
    selected = policies.find((p) => p.id === policy);
  async function exportResearch(format) {
    try {
      await saveResponse(
        await fetchWorkspace(
          "/api/exports/research",
          post({ packet_id: result.packet_id, answer, format }),
        ),
        `epfo-research-report.${format}`,
      );
    } catch (error) {
      setError(error.message);
    }
  }
  const exportReport = () =>
    download(
      "epfo-research-report.md",
      `# ${result.question}\n\nEvidence as of ${result.as_of}\n\n${answer || "Sources retrieved; no AI answer generated."}\n\n## Sources\n` +
        result.sources
          .map(
            (s) =>
              `[${s.id}] ${s.metadata.title} — ${s.metadata.date}\n${s.metadata.english_pdf_link || ""}\n`,
          )
          .join("\n") +
        "\n\n" +
        (context
          ? "Evidence is incomplete. Complete applicability is not established."
          : "Search of all documents; the policy-history date limit is not applied."),
    );
  const canGenerate = Boolean(catalog?.answer_enabled || hasToken);
  return (
    <div className="app friendly-workspace">
      <Sidebar
        page={page}
        menu={menu}
        setPage={setPage}
        setMenu={setMenu}
        policies={policies}
        recent={recent}
        catalog={catalog}
        newQuestion={newQuestion}
        setQuery={setQuery}
        search={search}
      />
      {menu && <div className="scrim" onClick={() => setMenu(false)} />}
      <div className="shell">
        <Topbar page={page} menu={menu} setMenu={setMenu} />
        <main
          className={
            page === "research"
              ? `research-main ${!result && !busy ? "is-home" : "has-results"}`
              : "workspace-main"
          }
        >
          <PageHeading page={page} />
          {error && (
            <div className="error-box" role="alert">
              <AlertCircle size={18} />
              <span>{error}</span>
              <button
                className="icon-btn"
                aria-label={tr("Dismiss error")}
                onClick={() => setError("")}
              >
                <X size={16} />
              </button>
            </div>
          )}
          {page === "research" && (
            <>
              <ResearchComposer
                input={input}
                researchOptions={researchOptions}
                catalog={catalog}
                policies={policies}
                selected={selected}
                hasToken={hasToken}
                busy={busy}
                generating={generating}
                result={result}
                query={query}
                setQuery={setQuery}
                language={language}
                setLanguage={setLanguage}
                detail={detail}
                setDetail={setDetail}
                policy={policy}
                setPolicy={setPolicy}
                queryTopic={queryTopic}
                setQueryTopic={setQueryTopic}
                day={day}
                setDay={setDay}
                search={search}
                setPage={setPage}
              />
              {busy && (
                <section className="loading-state" role="status">
                  <LoaderCircle className="spin" size={28} />
                  <h2>{tr("Finding relevant sources")}</h2>
                  <p>
                    {tr(
                      "Searching circulars and manuals for your question. The first search may take a little longer.",
                    )}
                  </p>
                </section>
              )}
              {result && (
                <ResearchResults
                  result={result}
                  answer={answer}
                  generating={generating}
                  canGenerate={canGenerate}
                  generate={generate}
                  source={source}
                  sourceOpen={sourceOpen}
                  setSourceOpen={setSourceOpen}
                  selectSource={selectSource}
                  answerComponents={answerComponents}
                  resultTab={resultTab}
                  setResultTab={setResultTab}
                  exportResearch={exportResearch}
                  exportReport={exportReport}
                  setError={setError}
                />
              )}
            </>
          )}
          {page === "history" && (
            <HistoryPage
              catalog={catalog}
              policies={policies}
              history={history}
              selectedHistory={selectedHistory}
              setSelectedHistory={setSelectedHistory}
              historyTopic={historyTopic}
              setHistoryTopic={setHistoryTopic}
              day={day}
              setDay={setDay}
              setPolicy={setPolicy}
              setQueryTopic={setQueryTopic}
              setPage={setPage}
              input={input}
            />
          )}
          {page === "library" && (
            <LibraryPage
              policies={policies}
              librarySort={librarySort}
              setLibrarySort={setLibrarySort}
              setSelectedHistory={setSelectedHistory}
              setHistoryTopic={setHistoryTopic}
              setPage={setPage}
              setDivisionFocus={setDivisionFocus}
            />
          )}
          {page === "divisions" && (
            <Divisions
              focus={divisionFocus}
              onOpenPolicy={(id) => {
                setSelectedHistory(id);
                setHistoryTopic("");
                setPage("history");
              }}
            />
          )}
          <div hidden={page !== "assistant"}>
            <Assistant
              language={language}
              onLanguage={setLanguage}
              enabled={Boolean(catalog?.answer_enabled || hasToken)}
            />
          </div>
          <div hidden={page !== "tools"}>
            <PdfTools />
          </div>
          <div hidden={page !== "settings"}>
            <SessionSettings hasToken={hasToken} onToken={setHasToken} />
          </div>
          <footer>
            <span>
              {tr(
                "Independent research assistance. Confirm the original source before acting.",
              )}
            </span>
            <span>{tr("Issue-date evidence · Traceable citations")}</span>
          </footer>
        </main>
      </div>
    </div>
  );
}
createRoot(document.getElementById("root")).render(
  <I18nProvider>
    <App />
  </I18nProvider>,
);
