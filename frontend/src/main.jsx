import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import Markdown from "./Markdown";
import {
  Search,
  LibraryBig,
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
  const [catalog, setCatalog] = useState(null),
    [page, setPage] = useState("research"),
    [menu, setMenu] = useState(false),
    [query, setQuery] = useState(""),
    [policy, setPolicy] = useState("auto"),
    [queryTopic, setQueryTopic] = useState(""),
    [day, setDay] = useState(new Date().toLocaleDateString("en-CA")),
    [language, setLanguage] = useState("English");
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
    [hasToken, setHasToken] = useState(false);
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
  async function search(e, question = query, mode = policy) {
    e?.preventDefault();
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
      setResult(r);
      setSource(r.sources[0] || null);
      setRecent((v) =>
        [question, ...v.filter((x) => x !== question)].slice(0, 5),
      );
    } catch (e) {
      if (seq === requestSeq.current) setError(e.message);
    } finally {
      if (seq === requestSeq.current) setBusy(false);
    }
  }
  async function generate() {
    if (!result) return;
    const seq = requestSeq.current,
      controller = new AbortController();
    answerAbort.current = controller;
    setGenerating(true);
    setAnswer("");
    setError("");
    try {
      const r = await fetch("/api/answers/" + result.packet_id, {
        ...post({ language }),
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
          if (item.text && seq === requestSeq.current)
            setAnswer((v) => v + item.text);
        }
        if (done) break;
      }
    } catch (e) {
      if (e.name !== "AbortError" && seq === requestSeq.current)
        setError(e.message);
    } finally {
      if (seq === requestSeq.current) setGenerating(false);
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
      "Joint declaration",
      "What is the procedure for joint declaration profile update?",
      "jd",
    ],
    [
      "EPF account transfer",
      "Can a transfer claim be rejected for overlapping service?",
      "transfers",
    ],
    [
      "Bank KYC",
      "Is employer approval required for bank account seeding?",
      "bank_kyc",
    ],
    [
      "Pension payment",
      "How does CPPS handle a pensioner changing bank branch?",
      "cpps",
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
          : "Standard hybrid search; the history date cutoff is not applied."),
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
            EPFO<span className="brand-sub">KNOWLEDGE ASSISTANT</span>
          </span>
        </a>
        <div className="sidebar-label">WORKSPACE</div>
        <nav>
          <button
            className={page === "research" ? "active" : ""}
            onClick={() => {
              setPage("research");
              setMenu(false);
            }}
          >
            <Search size={18} />
            Policy research<span className="nav-shortcut">⌘ K</span>
          </button>
          <button
            className={page === "history" ? "active" : ""}
            onClick={() => {
              setPage("history");
              setMenu(false);
            }}
          >
            <GitBranch size={18} />
            Circular history
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
            Policy library
          </button>
          <button
            className={page === "assistant" ? "active" : ""}
            onClick={() => {
              setPage("assistant");
              setMenu(false);
            }}
          >
            <FileText size={18} />
            Document & data assistant
          </button>
          <button
            className={page === "tools" ? "active" : ""}
            onClick={() => {
              setPage("tools");
              setMenu(false);
            }}
          >
            <Files size={18} />
            PDF utilities
          </button>
          <button
            className={page === "settings" ? "active" : ""}
            onClick={() => {
              setPage("settings");
              setMenu(false);
            }}
          >
            <Settings2 size={18} />
            Session settings
          </button>
        </nav>
        <div className="sidebar-divider" />
        <div className="sidebar-label">RECENT RESEARCH</div>
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
              Your recent questions will
              <br />
              appear here during this visit.
            </p>
          )}
        </div>
        <div className="sidebar-bottom">
          <div className="collection">
            <span className="collection-icon">
              <LibraryBig size={18} />
            </span>
            <div>
              Research collection
              <small>
                {catalog?.passage_count?.toLocaleString() || "Loading…"}{" "}
                searchable passages
              </small>
            </div>
          </div>
          <div className="independent">
            <span />
            Independent research tool<small>Not an official EPFO service</small>
          </div>
        </div>
      </aside>
      {menu && <div className="scrim" onClick={() => setMenu(false)} />}
      <div className="shell">
        <header className="topbar">
          <button
            className="mobile-menu icon-btn"
            aria-label="Open navigation"
            aria-expanded={menu}
            aria-controls="workspace-nav"
            onClick={() => setMenu(!menu)}
          >
            <Menu size={20} />
          </button>
          <div className="breadcrumb">
            Workspace
            <ChevronRight size={13} />
            <strong>
              {
                {
                  research: "Policy research",
                  history: "Circular history",
                  library: "Policy library",
                  assistant: "Document & data assistant",
                  tools: "PDF utilities",
                  settings: "Session settings",
                }[page]
              }
            </strong>
          </div>
          <div className="top-actions">
            <span className="preview-tag">RESEARCH & OFFICE WORKSPACE</span>
            <a
              href="https://www.epfo.gov.in/circulars"
              target="_blank"
              rel="noreferrer"
            >
              EPFO circulars
              <ArrowUpRight size={14} />
            </a>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                CIRCULARS · SCHEMES · STATUTORY MANUALS
              </div>
              <h1>
                {
                  {
                    research: "Research with the source in view.",
                    history: "Follow the policy through time.",
                    library: "A structured policy collection.",
                    assistant: "Work from the document.",
                    tools: "Prepare documents with care.",
                    settings: "Your workspace preferences.",
                  }[page]
                }
              </h1>
              <p>
                {
                  {
                    research:
                      "Find provisions, understand changes, and review the evidence behind each answer.",
                    history:
                      "Review issued instructions, amendments and the links that still need verification.",
                    library:
                      "Explore curated histories alongside the complete circular and statutory search index.",
                    assistant:
                      "Review a PDF, explore a CSV, and draft file-grounded notes in English or Hindi.",
                    tools:
                      "Merge, extract, rotate, compress and OCR your office documents.",
                    settings:
                      "Configure an optional inference token for this browser visit.",
                  }[page]
                }
              </p>
            </div>
            <div className="heading-stamp">
              <span className="stamp-line" />
              <span>
                EVIDENCE FIRST<small>Sources. Dates. Context.</small>
              </span>
            </div>
          </div>
          {error && (
            <div className="error-box" role="alert">
              <AlertCircle size={18} />
              <span>{error}</span>
              <button
                className="icon-btn"
                aria-label="Dismiss error"
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
                  <span className="number-label">01</span> ASK A POLICY QUESTION
                  <span className="search-label">
                    <span className="live-dot" />
                    {catalog
                      ? "Connected to the research collection"
                      : "Loading the research collection…"}
                  </span>
                </div>
                <form onSubmit={search}>
                  <div className="query-field">
                    <Search size={21} />
                    <input
                      ref={input}
                      aria-label="Policy question"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="For example, what is the procedure for a joint declaration profile update?"
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
                      {busy ? "Finding evidence…" : "Find evidence"}
                    </button>
                  </div>
                  <div className="filters">
                    <label>
                      POLICY COLLECTION
                      <select
                        aria-label="Policy collection"
                        value={policy}
                        onChange={(e) => {
                          setPolicy(e.target.value);
                          setQueryTopic("");
                        }}
                      >
                        <option value="auto">
                          Automatic · all policy scopes
                        </option>
                        {policies.map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.label}
                          </option>
                        ))}
                        <option value="hybrid">Standard hybrid search</option>
                      </select>
                    </label>
                    {(
                      catalog?.policies.find((p) => p.id === policy)?.topics ||
                      []
                    ).length > 0 && (
                      <label>
                        MATTER
                        <select
                          aria-label="Research matter"
                          value={queryTopic}
                          onChange={(e) => setQueryTopic(e.target.value)}
                        >
                          <option value="">Detect from question</option>
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
                    <div className="filter-divider" />
                    <label>
                      EVIDENCE AS OF
                      <div className="date-field">
                        <CalendarDays size={15} />
                        <input
                          aria-label="Evidence as of"
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
                      ANSWER LANGUAGE
                      <select
                        aria-label="Answer language"
                        value={language}
                        onChange={(e) => setLanguage(e.target.value)}
                      >
                        <option>English</option>
                        <option>Hindi</option>
                      </select>
                    </label>
                    <span className="filter-note">
                      <CircleHelp size={14} />
                      The date filters curated histories.
                      <br />
                      It does not certify applicability.
                    </span>
                  </div>
                </form>
              </section>
              {!result && !busy && (
                <>
                  <div className="suggestions">
                    <span>EXPLORE A QUESTION</span>
                    {examples.map(([label, q, mode]) => (
                      <button
                        key={label}
                        onClick={(e) => {
                          setQuery(q);
                          setPolicy(mode);
                          search(e, q, mode);
                        }}
                      >
                        {label}
                        <ArrowUpRight size={13} />
                      </button>
                    ))}
                  </div>
                  <div className="start-grid">
                    <section className="start-main">
                      <span className="eyebrow">
                        A MORE TRACEABLE WAY TO RESEARCH
                      </span>
                      <h2>
                        From a question
                        <br />
                        to the original circular.
                      </h2>
                      <p>
                        Your workspace brings the answer, its source pages and
                        the policy’s history together. Start with a specific
                        question or explore a curated collection.
                      </p>
                      <div className="steps">
                        <div>
                          <span>01</span>
                          <strong>Locate provisions</strong>
                          <p>Search circulars and statutory manuals.</p>
                        </div>
                        <div>
                          <span>02</span>
                          <strong>Understand changes</strong>
                          <p>Review dates and scoped relationships.</p>
                        </div>
                        <div>
                          <span>03</span>
                          <strong>Check the original</strong>
                          <p>Read the cited page before relying on it.</p>
                        </div>
                      </div>
                    </section>
                    <section className="collection-card">
                      <div className="collection-card-title">
                        <LibraryBig size={19} />
                        <span>THE RESEARCH COLLECTION</span>
                      </div>
                      <div className="big-stat">
                        {catalog?.passage_count?.toLocaleString() || "—"}
                        <span>searchable passages</span>
                      </div>
                      <div className="stat-rule" />
                      <div className="small-stats">
                        <div>
                          <strong>{policies.length || "—"}</strong>
                          <span>curated policy histories</span>
                        </div>
                        <div>
                          <strong>PDF</strong>
                          <span>physical page provenance</span>
                        </div>
                      </div>
                      <button
                        className="text-button"
                        onClick={() => setPage("library")}
                      >
                        Explore policy collections
                        <ArrowRight size={16} />
                      </button>
                    </section>
                  </div>
                  <section className="topics">
                    <div className="section-title">
                      <div>
                        <span className="eyebrow">CURATED COLLECTIONS</span>
                        <h2>Start with a policy.</h2>
                      </div>
                      <button
                        className="text-button"
                        onClick={() => setPage("library")}
                      >
                        View all {policies.length} collections
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
                              {p.document_count} documents · {p.checked_count}{" "}
                              checked sources
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
                  <h2>Assembling the evidence</h2>
                  <p>
                    Matching provisions and checking the dated policy inventory.
                    The first search may take longer while the search models
                    load.
                  </p>
                </section>
              )}
              {result && (
                <>
                  <div className="results-heading">
                    <div>
                      <span className="eyebrow">RESEARCH RESULT</span>
                      <h2>{result.question}</h2>
                    </div>
                    <button className="secondary" onClick={exportReport}>
                      <Download size={15} />
                      Export report
                    </button>
                    <button
                      className="secondary"
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
                      Word report
                    </button>
                  </div>
                  <div className="evidence-notice">
                    <AlertCircle size={17} />
                    <span>
                      {context ? (
                        <>
                          Instructions issued by{" "}
                          <strong>{dateText(result.as_of)}</strong>. The
                          evidence inventory is incomplete; current
                          applicability is not certified.
                        </>
                      ) : (
                        <>
                          Standard hybrid search. The history date cutoff is not
                          applied; amendment and applicability remain
                          unverified.
                        </>
                      )}
                    </span>
                  </div>
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
                              ? "Research answer"
                              : t === "sources"
                                ? `Sources (${result.sources.length})`
                                : "Policy history"}
                          </button>
                        ))}
                      </div>
                      {resultTab === "answer" && (
                        <div className="answer-content">
                          <div className="answer-meta">
                            <Status kind="teal">
                              {context?.policy_label || "Hybrid search"}
                            </Status>
                            <span>{result.sources.length} source pages</span>
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
                                {result.sources.length
                                  ? "The source evidence is ready."
                                  : "No matching evidence was found."}
                              </h3>
                              <p>
                                {result.sources.length
                                  ? "Review the original pages, or generate a cited synthesis from this evidence."
                                  : "Try a narrower question or choose Standard hybrid search."}
                              </p>
                              <button
                                className="primary"
                                disabled={
                                  !result.sources.length ||
                                  generating ||
                                  !(catalog?.answer_enabled || hasToken)
                                }
                                onClick={generate}
                              >
                                {generating ? (
                                  <LoaderCircle size={16} className="spin" />
                                ) : (
                                  <Sparkles size={16} />
                                )}{" "}
                                {generating
                                  ? "Generating…"
                                  : "Generate cited answer"}
                              </button>
                              {!(catalog?.answer_enabled || hasToken) && (
                                <small>
                                  AI generation is not configured. Sources
                                  remain available.
                                </small>
                              )}
                            </div>
                          )}
                          {generating && (
                            <div className="streaming-indicator" role="status">
                              <LoaderCircle size={14} className="spin" />
                              Generating from the numbered evidence…
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
                                {copied ? "Copied" : "Copy answer"}
                              </button>
                              <button
                                className="text-button"
                                onClick={generate}
                                disabled={generating}
                              >
                                Regenerate
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
                                  {s.metadata.title || "EPFO document"}
                                </strong>
                                <small>
                                  {s.metadata.date || "Date unavailable"} ·{" "}
                                  {s.metadata.page_provenance ===
                                  "physical_pdf_page"
                                    ? `PDF page ${s.metadata.page_number}`
                                    : "Page unverified"}
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
                              This question used standard hybrid retrieval.
                              Select a curated collection to explore its dated
                              history.
                            </p>
                          )}
                        </div>
                      )}
                    </section>
                    <aside id="source-panel" className="source-panel">
                      <div className="source-panel-heading">
                        <span>
                          <BookOpen size={17} />
                          SOURCE EVIDENCE
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
                                ? "Mirror copy"
                                : "Source document"}
                            </Status>
                            <h3>{source.metadata.title || "EPFO document"}</h3>
                            <code>
                              {source.metadata.circular_no ||
                                "Identifier unavailable"}
                            </code>
                            <div className="source-facts">
                              <span>
                                ISSUED
                                <strong>
                                  {dateText(source.metadata.date)}
                                </strong>
                              </span>
                              <span>
                                PROVENANCE
                                <strong>
                                  {source.metadata.page_provenance ===
                                  "physical_pdf_page"
                                    ? `Physical page ${source.metadata.page_number}`
                                    : "Document level"}
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
                              Source text
                            </button>
                            <button
                              className={sourceTab === "pdf" ? "selected" : ""}
                              disabled={!source.pdf_url}
                              onClick={() => setSourceTab("pdf")}
                            >
                              Original PDF
                            </button>
                          </div>
                          {sourceTab === "pdf" ? (
                            <iframe
                              title="Original PDF source"
                              src={`${source.pdf_url}#page=${source.metadata.page_number || 1}`}
                              className="pdf-viewer"
                            />
                          ) : (
                            <pre className="excerpt">{source.text}</pre>
                          )}
                          <div className="source-foot">
                            <span>
                              {source.metadata.evidence_status ||
                                "Applicability unverified"}
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
                              Open document
                              <ArrowUpRight size={14} />
                            </a>
                          </div>
                        </>
                      ) : (
                        <p className="empty">
                          Choose a numbered source to inspect the evidence.
                        </p>
                      )}
                    </aside>
                  </div>
                  {context?.fallback_reason && (
                    <details className="gaps">
                      <summary>
                        Coverage and unresolved evidence
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
                    MATTER
                    <select
                      aria-label="History matter"
                      value={historyTopic}
                      onChange={(e) => setHistoryTopic(e.target.value)}
                    >
                      <option value="">All matters</option>
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
                  POLICY COLLECTION
                  <select
                    aria-label="History policy collection"
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
                  EVIDENCE AS OF
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
                  Research this policy
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
                    <h2>Dates need context.</h2>
                    <p>
                      A circular’s issue date, its commencement and its
                      continued applicability are different facts.
                    </p>
                    <div className="rule-note">
                      <Status kind="teal">Source checked</Status>
                      <p>
                        The source text has been examined. It does not certify
                        current applicability.
                      </p>
                    </div>
                    <div className="rule-note">
                      <Status kind="gold">Evidence gap</Status>
                      <p>
                        A missing original or unresolved relationship remains
                        visible.
                      </p>
                    </div>
                    <h3>Recorded limitations</h3>
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
                      Download review
                    </button>
                  </aside>
                </div>
              ) : (
                <div className="loading-state">
                  <LoaderCircle size={24} className="spin" />
                  Loading policy history…
                </div>
              )}
            </>
          )}
          {page === "library" && (
            <>
              <div className="library-heading">
                <span>{policies.length} curated histories</span>
                <span>
                  Checked originals and unresolved evidence, shown together
                </span>
              </div>
              <div className="library-grid">
                {policies.map((p, i) => (
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
                    <div className="library-facts">
                      <span>
                        {p.checked_count}/{p.document_count} sources checked
                      </span>
                      <span>Latest checked: {dateText(p.newest_checked)}</span>
                    </div>
                  </button>
                ))}
              </div>
            </>
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
              Independent research assistance. Confirm the original source
              before acting.
            </span>
            <span>Issue-date evidence · Traceable citations</span>
          </footer>
        </main>
      </div>
    </div>
  );
}
function Timeline({ documents, relationships, onDocument, available }) {
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
                {i === 0 && <span>Most recent in inventory</span>}
              </div>
              <h3>{d.title || d.version_label || d.id}</h3>
              <code>{d.circular_no}</code>
              <div className="timeline-status">
                <Status kind={checked ? "teal" : "gold"}>
                  {checked ? "Source checked" : "Original unverified"}
                </Status>
                <span>
                  {d.effective_from
                    ? `Commencement recorded: ${dateText(d.effective_from)}`
                    : "Commencement unverified"}
                </span>
              </div>
              {relations.map((r) => (
                <div className="relationship" key={r.id}>
                  <GitBranch size={14} />
                  <div>
                    <strong>{r.type.replaceAll("_", " ")}</strong> ·{" "}
                    {r.verification === "source_checked"
                      ? "Source passage checked"
                      : "Target unresolved"}
                    <p>{r.scope}</p>
                  </div>
                </div>
              ))}
              {(d.pdf_url || available?.(d)) && (
                <button className="text-button" onClick={() => onDocument(d)}>
                  Review source
                  <ArrowUpRight size={14} />
                </button>
              )}
            </article>
          );
        })}
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
