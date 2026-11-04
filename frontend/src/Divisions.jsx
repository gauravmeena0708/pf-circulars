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
// Short names for the six kinds of work; the long labels and descriptions come from the API.
const NATURE_SHORT = {
  rule: "Rule / policy",
  forwarded: "Forwarded order",
  routine: "Routine work",
  data_request: "Data request",
  dissemination: "Dissemination",
  communication: "Communication",
};

// Table cells are short Markdown (bold, code); keep them inline.
const Inline = ({ children }) => (
  <Markdown components={{ p: ({ children: c }) => <span>{c}</span> }}>
    {children}
  </Markdown>
);

function MixBar({ mix, natures }) {
  const total = natures.reduce((n, x) => n + (mix[x.id] || 0), 0);
  if (!total) return null;
  return (
    <div className="mix">
      <div className="mix-bar" role="img" aria-label="Estimated mix of circulars by kind of work">
        {natures
          .filter((x) => mix[x.id])
          .map((x) => (
            <i
              key={x.id}
              className={`nat-${x.id}`}
              style={{ width: `${(100 * mix[x.id]) / total}%` }}
              title={`${NATURE_SHORT[x.id]}: ${mix[x.id]} circulars`}
            />
          ))}
      </div>
      <div className="mix-key">
        {natures
          .filter((x) => mix[x.id])
          .map((x) => (
            <span key={x.id}>
              <i className={`nat-${x.id}`} />
              {NATURE_SHORT[x.id]} {Math.round((100 * mix[x.id]) / total)}%
            </span>
          ))}
        <span className="mix-total">
          estimated from the titles of {total.toLocaleString("en-IN")} circulars
        </span>
      </div>
    </div>
  );
}

export default function Divisions({ onOpenPolicy }) {
  const [data, setData] = useState(null),
    [error, setError] = useState(""),
    [active, setActive] = useState(""),
    [term, setTerm] = useState(""),
    [kind, setKind] = useState("all"),
    [nature, setNature] = useState("all");
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
          (nature === "all" || r.nature === nature) &&
          (!needle || r.cells.join(" ").toLowerCase().includes(needle)),
      ),
    }));
  }, [division, needle, kind, nature]);
  const shown = sections.reduce((n, s) => n + s.rows.length, 0);

  if (error) return <p className="division-note error">{error}</p>;
  if (!data) return <p className="division-note">Loading divisions…</p>;
  const total = data.divisions.reduce((n, d) => n + d.row_count, 0);
  const counts = {};
  division?.sections.forEach((s) =>
    s.rows.forEach((r) => (counts[r.nature] = (counts[r.nature] || 0) + 1)),
  );
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
      <div className="nature-legend" aria-label="Kinds of work">
        <span className="nature-legend-title">Kind of work</span>
        <button
          className={nature === "all" ? "active" : ""}
          onClick={() => setNature("all")}
        >
          All
        </button>
        {data.natures.map((n) => (
          <button
            key={n.id}
            className={nature === n.id ? "active" : ""}
            onClick={() => setNature(nature === n.id ? "all" : n.id)}
            title={n.description}
          >
            <i className={`nat-${n.id}`} />
            {NATURE_SHORT[n.id]}
            <em>{counts[n.id] || 0}</em>
          </button>
        ))}
      </div>
      {division && (
        <>
          <p className="division-intro">
            <Inline>{division.intro}</Inline>
          </p>
          <MixBar mix={division.mix} natures={data.natures} />
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
          {sections.map((s, i) => {
            const cols = s.columns
              .map((c, n) => ({ c, n }))
              .filter((x) => x.c !== "#");
            return (
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
                          {cols.map((x, k) => (
                            <React.Fragment key={x.c + x.n}>
                              <th>{x.c}</th>
                              {k === 0 && <th>Kind of work</th>}
                            </React.Fragment>
                          ))}
                          <th>Collection</th>
                        </tr>
                      </thead>
                      <tbody>
                        {s.rows.map((r, j) => (
                          <tr key={j}>
                            {cols.map((x, k) => (
                              <React.Fragment key={x.c + x.n}>
                                <td>
                                  {/status/i.test(x.c) && (
                                    <em className={`kind ${r.status_kind}`}>
                                      {KIND_LABEL[r.status_kind]}
                                    </em>
                                  )}
                                  <Inline>{r.cells[x.n]}</Inline>
                                </td>
                                {k === 0 && (
                                  <td className="nature-cell">
                                    <span className={`nature nat-${r.nature}`}>
                                      {NATURE_SHORT[r.nature]}
                                    </span>
                                    {r.also.map((a) => (
                                      <span
                                        key={a}
                                        className={`nature also nat-${a}`}
                                        title="Also includes"
                                      >
                                        + {NATURE_SHORT[a]}
                                      </span>
                                    ))}
                                  </td>
                                )}
                              </React.Fragment>
                            ))}
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
            );
          })}
          {shown === 0 && (
            <p className="division-note">
              No topics match. Clear the search or a filter.
            </p>
          )}
          {division.notes && (
            <p className="division-foot">
              <Inline>{division.notes}</Inline>
            </p>
          )}
          <p className="division-foot">
            Source: <code>{division.source_file}</code>. Counts are approximate
            and topics overlap; the division tag is the issuing division. Kind
            of work: each topic is labelled by hand; the bar is an estimate from
            circular titles.
          </p>
        </>
      )}
    </>
  );
}
