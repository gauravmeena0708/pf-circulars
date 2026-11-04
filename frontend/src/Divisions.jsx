import React, { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, Search } from "lucide-react";
import Markdown from "./Markdown";
import { json } from "./client";

const KINDS = [
  ["all", "All statuses"],
  ["active", "Active"],
  ["ended", "Ended, replaced or withdrawn"],
  ["dormant", "No recent circular"],
  ["routine", "Routine or operational"],
];
const KIND_LABEL = {
  active: "Active",
  ended: "Ended / replaced",
  dormant: "No recent circular",
  routine: "Routine",
  other: "Not assessed",
};

// Table cells are short Markdown (bold, code); keep them inline.
const Inline = ({ children }) => (
  <Markdown components={{ p: ({ children: c }) => <span>{c}</span> }}>
    {children}
  </Markdown>
);

export default function Divisions({ onOpenPolicy }) {
  const [data, setData] = useState(null),
    [error, setError] = useState(""),
    [active, setActive] = useState(""),
    [term, setTerm] = useState(""),
    [kind, setKind] = useState("all");
  useEffect(() => {
    let live = true;
    json("/api/divisions")
      .then((d) => {
        if (!live) return;
        setData(d);
        setActive(d.divisions[0]?.id || "");
      })
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, []);
  const division = data?.divisions.find((d) => d.id === active);
  const needle = term.trim().toLowerCase();
  const sections = useMemo(() => {
    if (!division) return [];
    return division.sections.map((s) => ({
      ...s,
      rows: s.rows.filter(
        (r) =>
          (kind === "all" || r.status_kind === kind) &&
          (!needle || r.cells.join(" ").toLowerCase().includes(needle)),
      ),
    }));
  }, [division, needle, kind]);
  const shown = sections.reduce((n, s) => n + s.rows.length, 0);

  if (error) return <p className="division-note error">{error}</p>;
  if (!data) return <p className="division-note">Loading divisions…</p>;
  const total = data.divisions.reduce((n, d) => n + d.row_count, 0);
  return (
    <>
      <div className="library-heading">
        <span>
          {data.divisions.length} division groups · {total} policy topics
        </span>
        <span>
          Status is shown only where a circular says it. “No recent circular” is
          not a finding that a policy has ended.
        </span>
      </div>
      <div className="division-chips" role="tablist">
        {data.divisions.map((d) => (
          <button
            key={d.id}
            role="tab"
            aria-selected={d.id === active}
            className={d.id === active ? "active" : ""}
            onClick={() => setActive(d.id)}
          >
            {d.name}
            <span>{d.row_count}</span>
          </button>
        ))}
      </div>
      {division && (
        <>
          <p className="division-intro">
            <Inline>{division.intro}</Inline>
          </p>
          <div className="division-controls">
            <label className="division-search">
              <Search size={15} />
              <input
                value={term}
                onChange={(e) => setTerm(e.target.value)}
                placeholder={`Search ${division.name}`}
                aria-label="Search policy topics"
              />
            </label>
            <select
              value={kind}
              onChange={(e) => setKind(e.target.value)}
              aria-label="Filter by status"
            >
              {KINDS.map(([v, label]) => (
                <option key={v} value={v}>
                  {label}
                </option>
              ))}
            </select>
            <span className="division-count">
              {shown} of {division.row_count} topics
            </span>
          </div>
          {sections.map((s, i) => (
            <section className="division-section" key={s.heading + i}>
              {s.heading && <h3>{s.heading}</h3>}
              {s.text && (
                <p className="division-intro">
                  <Inline>{s.text}</Inline>
                </p>
              )}
              {s.rows.length > 0 && (
                <div className="table-scroll">
                  <table className="division-table">
                    <thead>
                      <tr>
                        {s.columns
                          .filter((c) => c !== "#")
                          .map((c) => (
                            <th key={c}>{c}</th>
                          ))}
                        <th>Collection</th>
                      </tr>
                    </thead>
                    <tbody>
                      {s.rows.map((r, j) => (
                        <tr key={j}>
                          {s.columns.map((c, n) =>
                            c === "#" ? null : (
                              <td key={c + n}>
                                {/status/i.test(c) && (
                                  <em className={`kind ${r.status_kind}`}>
                                    {KIND_LABEL[r.status_kind]}
                                  </em>
                                )}
                                <Inline>{r.cells[n]}</Inline>
                              </td>
                            ),
                          )}
                          <td className="division-links">
                            {r.collections
                              .filter((id) => data.collections[id])
                              .map((id) => (
                                <button
                                  key={id}
                                  onClick={() => onOpenPolicy(id)}
                                  title={`Open ${data.collections[id]}`}
                                >
                                  {data.collections[id]}
                                  <ArrowUpRight size={12} />
                                </button>
                              ))}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          ))}
          {shown === 0 && (
            <p className="division-note">
              No topics match. Clear the search or the status filter.
            </p>
          )}
          {division.notes && (
            <p className="division-foot">
              <Inline>{division.notes}</Inline>
            </p>
          )}
          <p className="division-foot">
            Source: <code>{division.source_file}</code>. Counts are approximate
            and topics overlap; the division tag is the issuing division.
          </p>
        </>
      )}
    </>
  );
}
