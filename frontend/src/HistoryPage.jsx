import { ArrowRight, Download, GitBranch, LoaderCircle } from "lucide-react";
import Timeline, { Status } from "./Timeline.jsx";
import { useI18n } from "./UiLanguage.jsx";
import { download } from "./lib";

// One policy's documents and relationships through time, up to the chosen date.
export default function HistoryPage({
  catalog,
  policies,
  history,
  selectedHistory,
  setSelectedHistory,
  historyTopic,
  setHistoryTopic,
  day,
  setDay,
  setPolicy,
  setQueryTopic,
  setPage,
  input,
}) {
  const { t: tr } = useI18n();
  return (
    <>
      <div className="history-controls">
        {(
          catalog?.policies.find((p) => p.id === selectedHistory)
            ?.topics || []
        ).length > 0 && (
          <label>
            {tr("Subtopic")}
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
          {tr("Policy topic")}
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
          {tr("Circulars issued by")}
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
            <div className="section-kicker">{history.label}</div>
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
  );
}
