import { ArrowUpRight, Check, CircleHelp, GitBranch } from "lucide-react";
import { useI18n } from "./UiLanguage.jsx";
import { dateText } from "./lib";

export function Status({ children, kind = "neutral" }) {
  return (
    <span className={"status " + kind}>
      <i />
      {children}
    </span>
  );
}

export default function Timeline({ documents, relationships, onDocument, available }) {
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
