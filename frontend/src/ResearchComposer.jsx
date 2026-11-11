import {
  ArrowLeftRight,
  ArrowRight,
  ArrowUp,
  BookOpen,
  BriefcaseBusiness,
  Building2,
  CalendarDays,
  ChevronDown,
  CircleHelp,
  Languages,
  LoaderCircle,
  Search,
  Settings2,
  UsersRound,
  Wallet,
} from "lucide-react";
import { useI18n } from "./UiLanguage.jsx";

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

// The question box, answer preferences, research options and starter tasks.
export default function ResearchComposer({
  input,
  researchOptions,
  catalog,
  policies,
  selected,
  hasToken,
  busy,
  generating,
  result,
  query,
  setQuery,
  language,
  setLanguage,
  detail,
  setDetail,
  policy,
  setPolicy,
  queryTopic,
  setQueryTopic,
  day,
  setDay,
  search,
  setPage,
}) {
  const { t: tr } = useI18n();
  return (
    <>
      <section
        className="search-card"
        aria-label={tr("Ask about EPFO rules")}
      >
        <form
          id="research-question-form"
          onSubmit={search}
          onInvalidCapture={(e) =>
            e.target.closest("details")?.setAttribute("open", "")
          }
        >
          <div className="query-field">
            <textarea
              ref={input}
              aria-label={tr("Policy question")}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  e.currentTarget.form.requestSubmit();
                }
              }}
              placeholder={tr("What would you like to know?")}
              rows={3}
              required
              minLength={3}
              maxLength={6000}
            />
          </div>
          <div className="composer-toolbar">
            <label className="composer-select">
              <Languages size={17} aria-hidden="true" />
              <span className="sr-only">{tr("Answer language")}</span>
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
            <label className="composer-select">
              <span className="sr-only">{tr("Answer length")}</span>
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
            <button
              className="primary composer-submit"
              disabled={busy || !catalog}
            >
              {busy ? tr("Finding sources…") : tr("Get answer")}
              {busy ? (
                <LoaderCircle className="spin" size={18} />
              ) : (
                <ArrowUp size={19} />
              )}
            </button>
          </div>
          <details ref={researchOptions} className="search-options">
            <summary>
              <Settings2 size={16} />
              {tr("Research options")}
              {catalog &&
                (policy !== "auto" || day !== catalog.today) && (
                  <span className="options-changed">
                    {tr("Custom filters")}
                  </span>
                )}
              <ChevronDown size={15} className="options-chevron" />
            </summary>
            <div className="filters">
              <label>
                {tr("Circulars issued by")}
                <div className="date-field">
                  <CalendarDays size={16} />
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
              <label>
                {tr("Policy topic")}
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
              {(selected?.topics || []).length > 0 && (
                <label>
                  {tr("Subtopic")}
                  <select
                    aria-label={tr("Policy subtopic")}
                    value={queryTopic}
                    onChange={(e) => setQueryTopic(e.target.value)}
                  >
                    <option value="">
                      {tr("Detect from question")}
                    </option>
                    {selected.topics.map((topic) => (
                      <option key={topic.id} value={topic.id}>
                        {topic.label}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <p className="filter-note">
                <CircleHelp size={16} />
                {tr(
                  "The date limits policy histories. It does not confirm which rules still apply.",
                )}
              </p>
            </div>
          </details>
        </form>
      </section>
      <div className="search-actions">
        <span className="search-assurance">
          <BookOpen size={15} />
          {tr("Answers with source references")}
        </span>
        <button
          type="submit"
          form="research-question-form"
          value="documents"
          className="text-button"
          disabled={busy || !catalog}
        >
          <Search size={15} />
          {tr("Search documents only")}
        </button>
      </div>
      {catalog && !(catalog.answer_enabled || hasToken) && (
        <p className="inference-unavailable" role="status">
          {tr(
            "Answers are unavailable right now. You can still search and read source documents.",
          )}
        </p>
      )}
      {!result && !busy && (
        <section
          className="task-section"
          aria-label={tr("Start with a task")}
        >
          <p>{tr("A few things I can help with")}</p>
          <div className="suggestions">
            {examples.map(([label, question], i) => {
              const Icon = [
                UsersRound,
                ArrowLeftRight,
                Wallet,
                Building2,
                BriefcaseBusiness,
              ][i];
              return (
                <button
                  key={label}
                  onClick={(e) => {
                    setQuery(question);
                    setPolicy("auto");
                    setQueryTopic("");
                    search(e, question, "auto");
                  }}
                >
                  <Icon size={17} strokeWidth={1.6} />
                  {tr(label)}
                </button>
              );
            })}
          </div>
          <button
            className="browse-topics text-button"
            onClick={() => setPage("library")}
          >
            {tr("Or browse policy topics")}
            <ArrowRight size={15} />
          </button>
        </section>
      )}
    </>
  );
}
