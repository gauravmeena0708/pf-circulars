import { ArrowUpRight } from "lucide-react";
import { LibraryRules, SpanBar, sortItems } from "./Divisions";
import { useI18n } from "./UiLanguage.jsx";
import { dateText } from "./lib";

// All curated policy histories on a shared timeline, plus the rule topics by division.
export default function LibraryPage({
  policies,
  librarySort,
  setLibrarySort,
  setSelectedHistory,
  setHistoryTopic,
  setPage,
  setDivisionFocus,
}) {
  const { t: tr } = useI18n();
  return (
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
  );
}
