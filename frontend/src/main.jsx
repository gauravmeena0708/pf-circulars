import { I18nProvider, useI18n } from "./UiLanguage.jsx";
import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import Markdown from "./Markdown";
import {
  Search,
  LibraryBig,
  Building2,
  GitBranch,
  Files,
  ArrowUpRight,
  ArrowRight,
  ChevronRight,
  ChevronDown,
  CalendarDays,
  FileText,
  Download,
  Check,
  X,
  PanelLeftClose,
  Menu,
  CircleHelp,
  Settings2,
  BookOpen,
  History,
  LoaderCircle,
  ScanLine,
  AlertCircle,
  ExternalLink,
  Copy,
  Sparkles,
} from "lucide-react";
import "./style.css";
import { Assistant, PdfTools, SessionSettings } from "./Workspace";
import Divisions, { LibraryRules, SpanBar, sortItems } from "./Divisions";
import { headers, request as fetchWorkspace, saveResponse } from "./client";

const dateText = (v) => {
  if (!v) return "Not established";
  const value = /^\d{2}\/\d{2}\/\d{4}$/.test(v)
    ? v.split("/").reverse().join("-")
    : v;
  const d = new Date(value + "T00:00:00");
  return Number.isNaN(d.getTime())
    ? v
    : d.toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric",
      });
};
async function api(path, options = {}) {
  const r = await fetch(path, {
    ...options,
    headers: headers(options.headers),
  });
  if (!r.ok) {
    let detail;
    try {
      detail = (await r.json()).detail;
    } catch {}
    throw Error(
      typeof detail === "string" ? detail : `Request failed (${r.status}).`,
    );
  }
  return r.json();
}
const post = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
function download(name, text, type = "text/markdown") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function Status({ children, kind = "neutral" }) {
  return (
    <span className={"status " + kind}>
      <i />
      {children}
    </span>
  );
}
function App() {
  const { t: tr, locale, setLocale } = useI18n();
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
    [sourceTab, setSourceTab] = useState("excerpt"),
    [resultTab, setResultTab] = useState("answer"),
    [recent, setRecent] = useState([]),
    [history, setHistory] = useState(null),
    [selectedHistory, setSelectedHistory] = useState("jd"),
    [historyTopic, setHistoryTopic] = useState(""),
    [copied, setCopied] = useState(false),
    [hasToken, setHasToken] = useState(false),
    [librarySort, setLibrarySort] = useState("latest"),
    [divisionFocus, setDivisionFocus] = useState("");
  const requestSeq = useRef(0),
    answerAbort = useRef(null),
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
  function selectSource(id) {
    const s = result?.sources.find((x) => x.id === Number(id));
    if (s) {
      setSource(s);
      setSourceTab("excerpt");
      document
        .getElementById("source-panel")
        ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }
  const context = result?.policy_context,
    policies = catalog?.policies || [],
    selected = policies.find((p) => p.id === policy);
  const examples = [
    [
      "Correct member details",
      "What is the procedure for a joint declaration profile update?",
    ],
    [
      "Transfer PF",
      "Can a transfer claim be rejected for overlapping service?",
    ],
    ["Pension", "How does CPPS handle a pensioner changing bank branch?"],
    [
      "Employer compliance",
      "What are the employer requirements for filing ECR returns?",
    ],
    [
      "Staff matters",
      "What is the procedure for an EPFO staff transfer request?",
    ],
  ];
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
  return (
    <div className="app">
      <aside id="workspace-nav" className={"sidebar " + (menu ? "open" : "")}>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("research");
          }}
        >
          <span className="brand-symbol">
            <BookOpen size={23} />
          </span>
          <span>
            {tr("EPFO")}
            <span className="brand-sub">{tr("KNOWLEDGE ASSISTANT")}</span>
          </span>
        </a>
        <div className="sidebar-label">{tr("WORKSPACE")}</div>
        <nav>
          <button
            className={page === "research" ? "active" : ""}
            onClick={() => {
              setPage("research");
              setMenu(false);
            }}
          >
            <Search size={18} />
            {tr("Ask about EPFO rules")}
            <span className="nav-shortcut">{tr("⌘ K")}</span>
          </button>
          <button
            className={page === "history" ? "active" : ""}
            onClick={() => {
              setPage("history");
              setMenu(false);
            }}
          >
            <GitBranch size={18} />
            {tr("Policy changes")}
            <span className="nav-count">{policies.length || "—"}</span>
          </button>
          <button
            className={page === "library" ? "active" : ""}
            onClick={() => {
              setPage("library");
              setMenu(false);
            }}
          >
            <LibraryBig size={18} />
            {tr("Browse policies")}
          </button>
          <button
            className={page === "divisions" ? "active" : ""}
            onClick={() => {
              setPage("divisions");
              setMenu(false);
            }}
          >
            <Building2 size={18} />
            {tr("Browse by division")}
          </button>
          <button
            className={page === "assistant" ? "active" : ""}
            onClick={() => {
              setPage("assistant");
              setMenu(false);
            }}
          >
            <FileText size={18} />
            {tr("File assistant")}
          </button>
          <button
            className={page === "tools" ? "active" : ""}
            onClick={() => {
              setPage("tools");
              setMenu(false);
            }}
          >
            <Files size={18} />
            {tr("PDF tools")}
          </button>
          <button
            className={page === "settings" ? "active" : ""}
            onClick={() => {
              setPage("settings");
              setMenu(false);
            }}
          >
            <Settings2 size={18} />
            {tr("Settings")}
          </button>
        </nav>
        <div className="sidebar-divider" />
        <div className="sidebar-label">{tr("RECENT RESEARCH")}</div>
        <div className="recent">
          {recent.length ? (
            recent.map((q) => (
              <button
                key={q}
                onClick={(e) => {
                  setQuery(q);
                  search(e, q);
                }}
              >
                <History size={14} />
                <span>{q}</span>
              </button>
            ))
          ) : (
            <p>
              {tr("Your recent questions will")}
              <br />
              {tr("appear here during this visit.")}
            </p>
          )}
        </div>
        <div className="sidebar-bottom">
          <div className="collection">
            <span className="collection-icon">
              <LibraryBig size={18} />
            </span>
            <div>
              {tr("Research collection")}
              <small>
                {catalog?.passage_count?.toLocaleString() || tr("Loading…")}{" "}
                {tr("searchable passages")}
                {catalog?.corpus_as_of && (
                  <>
                    {tr(" · circulars to ")}
                    {dateText(catalog.corpus_as_of)}
                  </>
                )}
              </small>
            </div>
          </div>
          <div className="independent">
            <span />
            {tr("Independent research tool")}
            <small>{tr("Not an official EPFO service")}</small>
          </div>
        </div>
      </aside>
      {menu && <div className="scrim" onClick={() => setMenu(false)} />}
      <div className="shell">
        <header className="topbar">
          <button
            className="mobile-menu icon-btn"
            aria-label={tr("Open navigation")}
            aria-expanded={menu}
            aria-controls="workspace-nav"
            onClick={() => setMenu(!menu)}
          >
            <Menu size={20} />
          </button>
          <div className="breadcrumb">
            {tr("Workspace")}
            <ChevronRight size={13} />
            <strong>
              {tr(
                {
                  research: "Ask about EPFO rules",
                  history: "Policy changes",
                  library: "Browse policies",
                  divisions: "Browse by division",
                  assistant: "File assistant",
                  tools: "PDF tools",
                  settings: "Settings",
                }[page],
              )}
            </strong>
          </div>
          <div className="top-actions">
            <div
              className="interface-language"
              role="group"
              aria-label={tr("Interface language")}
            >
              <button
                type="button"
                lang="en"
                aria-pressed={locale === "en"}
                onClick={() => setLocale("en")}
              >
                English
              </button>
              <button
                type="button"
                lang="hi"
                aria-pressed={locale === "hi"}
                onClick={() => setLocale("hi")}
              >
                हिन्दी
              </button>
            </div>
            <span className="preview-tag">
              {tr("RESEARCH & OFFICE WORKSPACE")}
            </span>
            <a
              href="https://www.epfo.gov.in/circulars"
              target="_blank"
              rel="noreferrer"
            >
              {tr("EPFO circulars")}
              <ArrowUpRight size={14} />
            </a>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                {tr("CIRCULARS · SCHEMES · STATUTORY MANUALS")}
              </div>
              <h1>
                {tr(
                  {
                    research: "Find the EPFO guidance you need.",
                    history: "Follow the policy through time.",
                    library: "Browse policies by topic.",
                    divisions: "Find policies by EPFO division.",
                    assistant: "Work from the document.",
                    tools: "Prepare documents with care.",
                    settings: "Your workspace preferences.",
                  }[page],
                )}
              </h1>
              <p>
                {tr(
                  {
                    research:
                      "Ask a question, get an answer with references, and check the original circular.",
                    history:
                      "Review issued instructions, amendments and the links that still need verification.",
                    library:
                      "Explore policy changes, circulars and statutory manuals by topic.",
                    divisions:
                      "Choose a division to explore its policy topics and source circulars.",
                    assistant:
                      "Review a PDF, explore a CSV, and draft file-grounded notes in English or Hindi.",
                    tools:
                      "Merge, extract, rotate, compress and OCR your office documents.",
                    settings:
                      "Manage your optional personal token for this visit.",
                  }[page],
                )}
              </p>
            </div>
            <div className="heading-stamp">
              <span className="stamp-line" />
              <span>
                {tr("EVIDENCE FIRST")}
                <small>{tr("Sources. Dates. Context.")}</small>
              </span>
            </div>
          </div>
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
              <section className="search-card">
                <div className="section-kicker">
                  <span className="number-label">01</span>
                  {tr(" ASK YOUR QUESTION")}
                  <span className="search-label">
                    <span className="live-dot" />
                    {catalog
                      ? tr("Connected to the research collection")
                      : tr("Loading the research collection…")}
                  </span>
                </div>
                <form onSubmit={search}>
                  <div className="query-field">
                    <Search size={21} />
                    <input
                      ref={input}
                      aria-label={tr("Policy question")}
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder={tr(
                        "For example, what is the procedure for a joint declaration profile update?",
                      )}
                      required
                      minLength={3}
                      maxLength={6000}
                    />
                    <button className="primary" disabled={busy || !catalog}>
                      {busy ? (
                        <LoaderCircle className="spin" size={17} />
                      ) : (
                        <Search size={17} />
                      )}{" "}
                      {busy ? tr("Finding sources…") : tr("Get answer")}
                    </button>
                  </div>
                  <div className="search-actions">
                    <button
                      type="submit"
                      value="documents"
                      className="text-button"
                      disabled={busy || !catalog}
                    >
                      <Search size={16} />
                      {tr(" Search documents only")}
                    </button>
                    {catalog && !(catalog.answer_enabled || hasToken) && (
                      <p>
                        {tr(
                          "Answers are unavailable right now. You can still search and read source documents.",
                        )}
                      </p>
                    )}
                  </div>
                  <div className="filters">
                    <div className="automatic-scope">
                      <GitBranch size={15} />
                      <span>
                        {policy === "auto"
                          ? tr("Topic selected from your question")
                          : tr("Scope: {value0}{value1}", {
                              value0: selected?.label || "Entire collection",
                              value1: selected?.topics?.find(
                                (t) => t.id === queryTopic,
                              )
                                ? " · " +
                                  selected.topics.find(
                                    (t) => t.id === queryTopic,
                                  ).label
                                : "",
                            })}
                      </span>
                      {policy !== "auto" && (
                        <button
                          type="button"
                          className="text-button"
                          onClick={() => {
                            setPolicy("auto");
                            setQueryTopic("");
                          }}
                        >
                          {tr("Use automatic")}
                        </button>
                      )}
                    </div>
                    <div className="filter-divider" />
                    <label>
                      {tr("CIRCULARS ISSUED BY")}
                      <div className="date-field">
                        <CalendarDays size={15} />
                        <input
                          aria-label={tr("Circulars issued by")}
                          type="date"
                          min="1952-01-01"
                          max={catalog?.today}
                          required
                          value={day}
                          onChange={(e) => setDay(e.target.value)}
                        />
                      </div>
                    </label>
                    <div className="filter-divider" />
                    <label>
                      {tr("ANSWER LANGUAGE")}
                      <select
                        aria-label={tr("Answer language")}
                        value={language}
                        disabled={busy || generating}
                        onChange={(e) => setLanguage(e.target.value)}
                      >
                        <option value="English">{tr("English")}</option>
                        <option value="Hindi">{tr("Hindi")}</option>
                      </select>
                    </label>
                    <label>
                      {tr("ANSWER LENGTH")}
                      <select
                        aria-label={tr("Answer length")}
                        value={detail}
                        disabled={busy || generating}
                        onChange={(e) => setDetail(e.target.value)}
                      >
                        <option value="brief">{tr("Brief answer")}</option>
                        <option value="detailed">
                          {tr("Detailed explanation")}
                        </option>
                      </select>
                    </label>
                    <p className="filter-note">
                      <CircleHelp size={16} />
                      {tr(
                        "The date limits policy histories. It does not confirm which rules still apply.",
                      )}
                    </p>
                  </div>
                  <details className="search-refinements">
                    <summary>
                      {tr("Choose a policy or search all documents ")}
                      <span>{tr("Optional")}</span>
                    </summary>
                    <div className="refinement-fields">
                      <label>
                        {tr("POLICY TOPIC (OPTIONAL)")}
                        <select
                          aria-label={tr("Policy collection")}
                          value={policy}
                          onChange={(e) => {
                            setPolicy(e.target.value);
                            setQueryTopic("");
                          }}
                        >
                          <option value="auto">
                            {tr("Detect from question")}
                          </option>
                          {policies.map((p) => (
                            <option key={p.id} value={p.id}>
                              {p.label}
                            </option>
                          ))}
                          <option value="hybrid">
                            {tr("Search all documents")}
                          </option>
                        </select>
                      </label>
                      {(
                        catalog?.policies.find((p) => p.id === policy)
                          ?.topics || []
                      ).length > 0 && (
                        <label>
                          {tr("SUBTOPIC")}
                          <select
                            aria-label={tr("Policy subtopic")}
                            value={queryTopic}
                            onChange={(e) => setQueryTopic(e.target.value)}
                          >
                            <option value="">
                              {tr("Detect from question")}
                            </option>
                            {catalog.policies
                              .find((p) => p.id === policy)
                              .topics.map((t) => (
                                <option key={t.id} value={t.id}>
                                  {t.label}
                                </option>
                              ))}
                          </select>
                        </label>
                      )}
                      <p>
                        {tr(
                          "We match your question to a policy topic automatically. Choose a topic here to narrow your search, or search all documents. The date limit applies to policy histories only.",
                        )}
                      </p>
                    </div>
                  </details>
                </form>
              </section>
              {!result && !busy && (
                <>
                  <div className="suggestions">
                    <span>{tr("START WITH A TASK")}</span>
                    {examples.map(([label, q]) => (
                      <button
                        key={tr(label)}
                        onClick={(e) => {
                          setQuery(q);
                          setPolicy("auto");
                          setQueryTopic("");
                          search(e, q, "auto");
                        }}
                      >
                        {tr(label)}
                        <ArrowUpRight size={13} />
                      </button>
                    ))}
                  </div>
                  <div className="start-grid">
                    <section className="start-main">
                      <span className="eyebrow">
                        {tr("ANSWERS WITH REFERENCES")}
                      </span>
                      <h2>
                        {tr("From a question")}
                        <br />
                        {tr("to the original circular.")}
                      </h2>
                      <p>
                        {tr(
                          "See the answer, source pages and policy changes together. Ask a specific question or browse a policy topic.",
                        )}
                      </p>
                      <div className="steps">
                        <div>
                          <span>01</span>
                          <strong>{tr("Locate provisions")}</strong>
                          <p>{tr("Search circulars and statutory manuals.")}</p>
                        </div>
                        <div>
                          <span>02</span>
                          <strong>{tr("Understand changes")}</strong>
                          <p>{tr("See what later instructions changed.")}</p>
                        </div>
                        <div>
                          <span>03</span>
                          <strong>{tr("Check the original")}</strong>
                          <p>
                            {tr("Read the cited page before relying on it.")}
                          </p>
                        </div>
                      </div>
                    </section>
                    <section className="collection-card">
                      <div className="collection-card-title">
                        <LibraryBig size={19} />
                        <span>{tr("THE RESEARCH COLLECTION")}</span>
                      </div>
                      <div className="big-stat">
                        {catalog?.passage_count?.toLocaleString() || "—"}
                        <span>{tr("searchable passages")}</span>
                      </div>
                      <div className="stat-rule" />
                      <div className="small-stats">
                        <div>
                          <strong>{policies.length || "—"}</strong>
                          <span>{tr("policy histories")}</span>
                        </div>
                        <div>
                          <strong>{tr("PDF")}</strong>
                          <span>{tr("original source pages")}</span>
                        </div>
                      </div>
                      <button
                        className="text-button"
                        onClick={() => setPage("library")}
                      >
                        {tr("Explore policy collections")}
                        <ArrowRight size={16} />
                      </button>
                    </section>
                  </div>
                  <section className="topics">
                    <div className="section-title">
                      <div>
                        <span className="eyebrow">{tr("POLICY TOPICS")}</span>
                        <h2>{tr("Start with a policy.")}</h2>
                      </div>
                      <button
                        className="text-button"
                        onClick={() => setPage("library")}
                      >
                        {tr("View all {count} collections", {
                          count: policies.length,
                        })}
                        <ArrowRight size={15} />
                      </button>
                    </div>
                    <div className="topic-grid">
                      {policies
                        .filter((p) =>
                          [
                            "jd",
                            "transfers",
                            "higher_pension",
                            "mid_delinking",
                          ].includes(p.id),
                        )
                        .map((p, i) => (
                          <button
                            key={p.id}
                            className="topic-card"
                            onClick={() => {
                              setSelectedHistory(p.id);
                              setHistoryTopic("");
                              setPage("history");
                            }}
                          >
                            <span className="topic-index">0{i + 1}</span>
                            <GitBranch size={20} />
                            <h3>{p.label}</h3>
                            <p>
                              {p.document_count}
                              {tr(" documents · ")}
                              {p.checked_count} {tr("checked sources")}
                            </p>
                            <ArrowUpRight className="topic-arrow" size={18} />
                          </button>
                        ))}
                    </div>
                  </section>
                </>
              )}
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
                <>
                  <div className="results-heading">
                    <div>
                      <span className="eyebrow">{tr("RESEARCH RESULT")}</span>
                      <h2>{result.question}</h2>
                    </div>
                    <button
                      className="secondary"
                      onClick={exportReport}
                      disabled={generating}
                    >
                      <Download size={15} />
                      {tr("Export report")}
                    </button>
                    <button
                      className="secondary"
                      disabled={generating}
                      onClick={async () => {
                        try {
                          await saveResponse(
                            await fetchWorkspace(
                              "/api/exports/research",
                              post({ packet_id: result.packet_id, answer }),
                            ),
                            "epfo-research-report.docx",
                          );
                        } catch (e) {
                          setError(e.message);
                        }
                      }}
                    >
                      <Download size={15} />
                      {tr("Word report")}
                    </button>
                  </div>
                  <div className="evidence-notice">
                    <AlertCircle size={17} />
                    <span>
                      {context ? (
                        <>
                          {result.selection_mode === "auto"
                            ? tr("Detected policy: ")
                            : tr("Selected policy: ")}
                          <strong>{context.policy_label}</strong>.
                          {context.selected_topics?.length > 0 && (
                            <>
                              {" "}
                              {tr("Subtopics:")}{" "}
                              {context.topics
                                .filter((t) =>
                                  context.selected_topics.includes(t.id),
                                )
                                .map((t) => t.label)
                                .join(", ")}
                              .{" "}
                            </>
                          )}{" "}
                          {tr("Instructions issued by")}{" "}
                          <strong>{dateText(result.as_of)}</strong>
                          {tr(
                            ". The evidence inventory is incomplete; current applicability is not certified.",
                          )}
                        </>
                      ) : (
                        <>
                          {result.selection_mode === "auto"
                            ? tr(
                                "No policy topic matched; all documents were searched. ",
                              )
                            : tr("Searching the full collection. ")}
                          {tr("The date limit was not applied.")}
                          {!result.currency &&
                            tr(
                              " Later changes and continued applicability have not been verified.",
                            )}
                        </>
                      )}
                    </span>
                  </div>
                  {result.currency && (
                    <div
                      className={`currency-notice ${result.currency.level}`}
                      role="note"
                    >
                      <strong>
                        {tr("Check whether this is still current.")}
                      </strong>{" "}
                      {result.currency.notice}
                      {result.currency.newer_related.length > 0 && (
                        <ul>
                          {result.currency.newer_related.map((c) => (
                            <li key={c.date + (c.circular_no || c.title)}>
                              {c.url ? (
                                <a
                                  href={c.url}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  {c.title}
                                </a>
                              ) : (
                                c.title
                              )}{" "}
                              — {dateText(c.date)}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                  <div className="results-grid">
                    <section className="answer-card">
                      <div className="tabbar">
                        {["answer", "sources", "history"].map((t) => (
                          <button
                            className={resultTab === t ? "selected" : ""}
                            onClick={() => setResultTab(t)}
                            key={t}
                          >
                            {t === "answer"
                              ? tr("Answer")
                              : t === "sources"
                                ? tr("Sources ({value0})", {
                                    value0: result.sources.length,
                                  })
                                : tr("Policy history")}
                          </button>
                        ))}
                      </div>
                      {resultTab === "answer" && (
                        <div className="answer-content">
                          <div className="answer-meta">
                            <Status kind="teal">
                              {context?.policy_label || tr("All documents")}
                            </Status>
                            <span>
                              {result.sources.length}
                              {tr(" source pages")}
                            </span>
                          </div>
                          {answer ? (
                            <div className="markdown">
                              <Markdown
                                components={{
                                  a: ({ href, children }) =>
                                    href?.startsWith("#source-") ? (
                                      <button
                                        className="citation"
                                        onClick={() =>
                                          selectSource(href.slice(8))
                                        }
                                      >
                                        {children}
                                      </button>
                                    ) : (
                                      <a
                                        href={href}
                                        target="_blank"
                                        rel="noreferrer"
                                      >
                                        {children}
                                      </a>
                                    ),
                                }}
                              >
                                {answer.replace(
                                  /\[(\d+)\]/g,
                                  "[$1](#source-$1)",
                                )}
                              </Markdown>
                            </div>
                          ) : (
                            <div className="generate-state">
                              <span className="generate-icon">
                                <FileText size={26} />
                              </span>
                              <h3>
                                {generating
                                  ? tr("Writing your answer…")
                                  : result.sources.length
                                    ? tr("Your source documents are ready.")
                                    : tr("No matching sources were found.")}
                              </h3>
                              <p>
                                {generating
                                  ? tr(
                                      "You can read the source pages while your answer is prepared.",
                                    )
                                  : result.sources.length
                                    ? tr(
                                        "Read the original pages, or get an answer with references.",
                                      )
                                    : tr(
                                        "Try a more specific question or search all documents.",
                                      )}
                              </p>
                              <button
                                className="primary"
                                disabled={
                                  !result.sources.length ||
                                  generating ||
                                  !(catalog?.answer_enabled || hasToken)
                                }
                                onClick={() => generate()}
                              >
                                {generating ? (
                                  <LoaderCircle size={16} className="spin" />
                                ) : (
                                  <Sparkles size={16} />
                                )}{" "}
                                {generating
                                  ? tr("Generating…")
                                  : tr("Get answer with references")}
                              </button>
                              {!(catalog?.answer_enabled || hasToken) && (
                                <small>
                                  {tr(
                                    "AI generation is not configured. Sources remain available.",
                                  )}
                                </small>
                              )}
                            </div>
                          )}
                          {generating && (
                            <div className="streaming-indicator" role="status">
                              <LoaderCircle size={14} className="spin" />
                              {tr("Preparing your answer with references…")}
                            </div>
                          )}
                          {answer && (
                            <div className="answer-actions">
                              <button
                                className="text-button"
                                onClick={async () => {
                                  try {
                                    await navigator.clipboard.writeText(answer);
                                    setCopied(true);
                                    setTimeout(() => setCopied(false), 2000);
                                  } catch {
                                    setError(
                                      "Clipboard is unavailable. Use Export report.",
                                    );
                                  }
                                }}
                              >
                                {copied ? (
                                  <Check size={14} />
                                ) : (
                                  <Copy size={14} />
                                )}{" "}
                                {copied ? tr("Copied") : tr("Copy answer")}
                              </button>
                              <button
                                className="text-button"
                                onClick={() => generate()}
                                disabled={generating}
                              >
                                {tr("Regenerate")}
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                      {resultTab === "sources" && (
                        <div className="source-list">
                          {result.sources.map((s) => (
                            <button
                              key={s.id}
                              onClick={() => selectSource(s.id)}
                              className={source?.id === s.id ? "chosen" : ""}
                            >
                              <span className="source-num">{s.id}</span>
                              <div>
                                <strong>
                                  {s.metadata.title || tr("EPFO document")}
                                </strong>
                                <small>
                                  {s.metadata.date || tr("Date unavailable")} ·{" "}
                                  {s.metadata.page_provenance ===
                                  "physical_pdf_page"
                                    ? tr("PDF page {value0}", {
                                        value0: s.metadata.page_number,
                                      })
                                    : tr("Page number unverified")}
                                </small>
                              </div>
                              <ChevronRight size={16} />
                            </button>
                          ))}
                        </div>
                      )}
                      {resultTab === "history" && (
                        <div className="inline-history">
                          {context ? (
                            <Timeline
                              documents={context.documents}
                              relationships={context.relationships}
                              available={(d) =>
                                result.sources.some(
                                  (s) => s.document_id === d.id,
                                )
                              }
                              onDocument={(d) => {
                                const s = result.sources.find(
                                  (s) => s.document_id === d.id,
                                );
                                if (s) selectSource(s.id);
                              }}
                            />
                          ) : (
                            <p>
                              {tr(
                                "This search covered all documents. Choose a policy topic to explore its history.",
                              )}
                            </p>
                          )}
                        </div>
                      )}
                    </section>
                    <aside id="source-panel" className="source-panel">
                      <div className="source-panel-heading">
                        <span>
                          <BookOpen size={17} />
                          {tr("SOURCE DOCUMENT")}
                        </span>
                        {source && (
                          <span className="source-number">[{source.id}]</span>
                        )}
                      </div>
                      {source ? (
                        <>
                          <div className="source-details">
                            <Status
                              kind={
                                source.metadata.source_host_kind ===
                                "mirror_copy"
                                  ? "gold"
                                  : "teal"
                              }
                            >
                              {source.metadata.source_host_kind ===
                              "mirror_copy"
                                ? tr("Mirror copy")
                                : tr("Source document")}
                            </Status>
                            <h3>
                              {source.metadata.title || tr("EPFO document")}
                            </h3>
                            <code>
                              {source.metadata.circular_no ||
                                tr("Identifier unavailable")}
                            </code>
                            <div className="source-facts">
                              <span>
                                {tr("ISSUED")}
                                <strong>
                                  {dateText(source.metadata.date)}
                                </strong>
                              </span>
                              <span>
                                {tr("SOURCE PAGE")}
                                <strong>
                                  {source.metadata.page_provenance ===
                                  "physical_pdf_page"
                                    ? tr("PDF page {value0}", {
                                        value0: source.metadata.page_number,
                                      })
                                    : tr("Page number unverified")}
                                </strong>
                              </span>
                            </div>
                          </div>
                          <div className="source-tabs">
                            <button
                              className={
                                sourceTab === "excerpt" ? "selected" : ""
                              }
                              onClick={() => setSourceTab("excerpt")}
                            >
                              {tr("Source text")}
                            </button>
                            <button
                              className={sourceTab === "pdf" ? "selected" : ""}
                              disabled={!source.pdf_url}
                              onClick={() => setSourceTab("pdf")}
                            >
                              {tr("Original PDF")}
                            </button>
                          </div>
                          {sourceTab === "pdf" ? (
                            <iframe
                              title={tr("Original PDF source")}
                              src={`${source.pdf_url}#page=${source.metadata.page_number || 1}`}
                              className="pdf-viewer"
                            />
                          ) : (
                            <pre className="excerpt">{source.text}</pre>
                          )}
                          <div className="source-foot">
                            <span>
                              {source.metadata.evidence_status ||
                                tr("Applicability unverified")}
                            </span>
                            <a
                              href={
                                source.pdf_url ||
                                source.metadata.english_pdf_link ||
                                source.metadata.source_pdf
                              }
                              target="_blank"
                              rel="noreferrer"
                            >
                              {tr("Open document")}
                              <ArrowUpRight size={14} />
                            </a>
                          </div>
                        </>
                      ) : (
                        <p className="empty">
                          {tr(
                            "Choose a numbered source to inspect the evidence.",
                          )}
                        </p>
                      )}
                    </aside>
                  </div>
                  {context?.fallback_reason && (
                    <details className="gaps">
                      <summary>
                        {tr("Missing sources and limitations")}
                        <ChevronDown size={15} />
                      </summary>
                      <p>{context.fallback_reason}</p>
                      <ul>
                        {context.limitations.map((l, i) => (
                          <li key={i}>{l}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                </>
              )}
            </>
          )}
          {page === "history" && (
            <>
              <div className="history-controls">
                {(
                  catalog?.policies.find((p) => p.id === selectedHistory)
                    ?.topics || []
                ).length > 0 && (
                  <label>
                    {tr("SUBTOPIC")}
                    <select
                      aria-label={tr("History subtopic")}
                      value={historyTopic}
                      onChange={(e) => setHistoryTopic(e.target.value)}
                    >
                      <option value="">{tr("All subtopics")}</option>
                      {catalog.policies
                        .find((p) => p.id === selectedHistory)
                        .topics.map((t) => (
                          <option key={t.id} value={t.id}>
                            {t.label}
                          </option>
                        ))}
                    </select>
                  </label>
                )}

                <label>
                  {tr("POLICY COLLECTION")}
                  <select
                    aria-label={tr("History policy collection")}
                    value={selectedHistory}
                    onChange={(e) => {
                      setSelectedHistory(e.target.value);
                      setHistoryTopic("");
                    }}
                  >
                    {policies.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  {tr("CIRCULARS ISSUED BY")}
                  <input
                    type="date"
                    min="1952-01-01"
                    max={catalog?.today}
                    value={day}
                    onChange={(e) => setDay(e.target.value)}
                    required
                  />
                </label>
                <button
                  className="primary"
                  onClick={() => {
                    setPolicy(selectedHistory);
                    setQueryTopic(historyTopic);
                    setPage("research");
                    input.current?.focus();
                  }}
                >
                  {tr("Ask about this policy")}
                  <ArrowRight size={16} />
                </button>
              </div>
              {history ? (
                <div className="history-grid">
                  <section className="timeline-card">
                    <div className="section-kicker">
                      <span className="number-label">02</span>
                      {history.label.toUpperCase()}
                    </div>
                    <p className="scope">{history.scope}</p>
                    <Timeline
                      documents={history.documents}
                      relationships={history.relationships}
                      onDocument={(d) => {
                        if (d.pdf_url)
                          window.open(
                            d.pdf_url,
                            "_blank",
                            "noopener,noreferrer",
                          );
                      }}
                    />
                  </section>
                  <aside className="history-note">
                    <GitBranch size={28} />
                    <h2>{tr("Dates need context.")}</h2>
                    <p>
                      {tr(
                        "A circular’s issue date, its commencement and its continued applicability are different facts.",
                      )}
                    </p>
                    <div className="rule-note">
                      <Status kind="teal">{tr("Source checked")}</Status>
                      <p>
                        {tr(
                          "The source text has been examined. It does not certify current applicability.",
                        )}
                      </p>
                    </div>
                    <div className="rule-note">
                      <Status kind="gold">{tr("Evidence gap")}</Status>
                      <p>
                        {tr(
                          "A missing original or unresolved relationship remains visible.",
                        )}
                      </p>
                    </div>
                    <h3>{tr("Recorded limitations")}</h3>
                    <ul>
                      {history.limitations.map((l, i) => (
                        <li key={i}>{l}</li>
                      ))}
                    </ul>
                    <button
                      className="secondary"
                      onClick={() =>
                        download(
                          "policy-evidence-review.json",
                          JSON.stringify(history, null, 2),
                          "application/json",
                        )
                      }
                    >
                      <Download size={15} />
                      {tr("Download review")}
                    </button>
                  </aside>
                </div>
              ) : (
                <div className="loading-state">
                  <LoaderCircle size={24} className="spin" />
                  {tr("Loading policy history…")}
                </div>
              )}
            </>
          )}
          {page === "library" && (
            <>
              <div className="library-heading">
                <span>
                  {policies.length}
                  {tr(" policy histories")}
                </span>
                <span>
                  {tr(
                    "Checked originals and unresolved evidence, shown together",
                  )}
                </span>
              </div>
              <div
                className="library-sort"
                role="group"
                aria-label={tr("Order by timeline")}
              >
                <span>{tr("Timeline")}</span>
                {[
                  ["latest", "Latest first"],
                  ["earliest", "Earliest first"],
                  ["az", "A to Z"],
                ].map(([v, label]) => (
                  <button
                    key={v}
                    className={librarySort === v ? "active" : ""}
                    onClick={() => setLibrarySort(v)}
                  >
                    {tr(label)}
                  </button>
                ))}
              </div>
              <div className="library-grid">
                {sortItems(
                  policies.map((p) => ({
                    p,
                    name: p.label,
                    start: Number((p.first_issued || "").slice(0, 4)) || 0,
                    end: Number((p.last_issued || "").slice(0, 4)) || 0,
                  })),
                  librarySort,
                ).map(({ p, start, end }, i) => (
                  <button
                    className="library-card"
                    key={p.id}
                    onClick={() => {
                      setSelectedHistory(p.id);
                      setHistoryTopic("");
                      setPage("history");
                    }}
                  >
                    <div>
                      <span className="topic-index">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <ArrowUpRight size={18} />
                    </div>
                    <h2>{p.label}</h2>
                    <p>{p.scope}</p>
                    <SpanBar start={start} end={end} />
                    <div className="library-facts">
                      <span>
                        {p.checked_count}/{p.document_count}
                        {tr(" sources checked")}
                      </span>
                      <span>
                        {start === end
                          ? start
                          : tr("{value0}-{value1}", {
                              value0: start,
                              value1: end,
                            })}
                        {tr(" · latest")} {dateText(p.last_issued)}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
              <LibraryRules
                sort={librarySort}
                onOpenDivision={(id) => {
                  setDivisionFocus(id);
                  setPage("divisions");
                }}
              />
            </>
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
function Timeline({ documents, relationships, onDocument, available }) {
  const { t: tr } = useI18n();
  return (
    <div className="timeline">
      {[...documents]
        .sort((a, b) => b.issued_on.localeCompare(a.issued_on))
        .map((d, i) => {
          const checked = d.status !== "source_unverified";
          const relations = relationships.filter((r) => r.source === d.id);
          return (
            <article
              key={d.id}
              className={"timeline-item " + (!checked ? "unverified" : "")}
            >
              <div className="timeline-dot">
                {checked ? <Check size={11} /> : <CircleHelp size={11} />}
              </div>
              <div className="timeline-date">
                {dateText(d.issued_on)}
                {i === 0 && <span>{tr("Most recent in inventory")}</span>}
              </div>
              <h3>{d.title || d.version_label || d.id}</h3>
              <code>{d.circular_no}</code>
              <div className="timeline-status">
                <Status kind={checked ? "teal" : "gold"}>
                  {checked ? tr("Source checked") : tr("Original unverified")}
                </Status>
                <span>
                  {d.effective_from
                    ? tr("Commencement recorded: {value0}", {
                        value0: dateText(d.effective_from),
                      })
                    : tr("Commencement unverified")}
                </span>
              </div>
              {relations.map((r) => (
                <div className="relationship" key={r.id}>
                  <GitBranch size={14} />
                  <div>
                    <strong>{r.type.replaceAll("_", " ")}</strong> ·{" "}
                    {r.verification === "source_checked"
                      ? tr("Source passage checked")
                      : tr("Target unresolved")}
                    <p>{r.scope}</p>
                  </div>
                </div>
              ))}
              {(d.pdf_url || available?.(d)) && (
                <button className="text-button" onClick={() => onDocument(d)}>
                  {tr("Review source")}
                  <ArrowUpRight size={14} />
                </button>
              )}
            </article>
          );
        })}
    </div>
  );
}
createRoot(document.getElementById("root")).render(
  <I18nProvider>
    <App />
  </I18nProvider>,
);
