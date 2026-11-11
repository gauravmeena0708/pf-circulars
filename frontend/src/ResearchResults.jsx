import { useState } from "react";
import {
  AlertCircle,
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  FileText,
  LoaderCircle,
  Sparkles,
} from "lucide-react";
import AnswerFeedback from "./AnswerFeedback.jsx";
import DownloadMenu from "./DownloadMenu.jsx";
import Markdown from "./Markdown";
import SourceDrawer from "./SourceDrawer.jsx";
import Timeline, { Status } from "./Timeline.jsx";
import { useI18n } from "./UiLanguage.jsx";
import { dateText, linkCitations } from "./lib";

// The result of one search: evidence notices, the answer, its sources and the policy history.
export default function ResearchResults({
  result,
  answer,
  generating,
  canGenerate,
  generate,
  source,
  sourceOpen,
  setSourceOpen,
  selectSource,
  answerComponents,
  resultTab,
  setResultTab,
  exportResearch,
  exportReport,
  setError,
}) {
  const { t: tr } = useI18n();
  const [copied, setCopied] = useState(false);
  const context = result.policy_context;
  return (
    <>
      <div className="results-heading">
        <div>
          <h2>{result.question}</h2>
        </div>
        <DownloadMenu
          ariaLabel="Download research report"
          disabled={generating}
          items={[
            {
              id: "pdf",
              label: "PDF document (.pdf)",
              description: "Ready to print or share",
              onSelect: () => exportResearch("pdf"),
            },
            {
              id: "docx",
              label: "Word document (.docx)",
              description: "Editable document",
              onSelect: () => exportResearch("docx"),
            },
            {
              id: "markdown",
              label: "Markdown text (.md)",
              onSelect: exportReport,
            },
          ]}
        />
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
                <button
                  className="text-button source-count"
                  onClick={() => setResultTab("sources")}
                >
                  <BookOpen size={15} />
                  {tr("Sources ({value0})", {
                    value0: result.sources.length,
                  })}
                </button>
              </div>
              {answer ? (
                <div className="markdown">
                  <Markdown components={answerComponents}>
                    {linkCitations(answer)}
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
                      !canGenerate
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
                  {!canGenerate && (
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
                          "Clipboard is unavailable. Use Download.",
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
                              <AnswerFeedback
                                packetId={result.packet_id}
                                disabled={generating}
                              />
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
        <SourceDrawer
          source={source}
          open={sourceOpen}
          onClose={() => setSourceOpen(false)}
        />
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
  );
}
