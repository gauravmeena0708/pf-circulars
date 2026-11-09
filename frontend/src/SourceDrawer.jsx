import React, { useEffect, useRef, useState } from "react";
import { ArrowUpRight, BookOpen, X } from "lucide-react";
import { useI18n } from "./UiLanguage.jsx";

export default function SourceDrawer({ source, open, onClose }) {
  const { t: tr } = useI18n();
  const dialog = useRef(null);
  const [tab, setTab] = useState("excerpt");
  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (open) {
      setTab("excerpt");
      element.scrollTop = 0;
      if (!element.open) element.showModal();
    } else if (element.open) {
      element.close();
    }
  }, [open, source?.id]);

  const metadata = source?.metadata || {};
  const url =
    source?.pdf_url || metadata.english_pdf_link || metadata.source_pdf;
  return (
    <dialog
      ref={dialog}
      className="source-dialog"
      aria-labelledby="source-drawer-title"
      onClose={onClose}
      onClick={(event) => {
        if (event.target !== event.currentTarget) return;
        const rect = event.currentTarget.getBoundingClientRect();
        if (
          event.clientX < rect.left ||
          event.clientX > rect.right ||
          event.clientY < rect.top ||
          event.clientY > rect.bottom
        )
          onClose();
      }}
    >
      <header className="source-dialog-header">
        <span>
          <BookOpen size={18} />
          {tr("Source document")}
          {source && <span className="source-number">[{source.id}]</span>}
        </span>
        <button
          type="button"
          className="icon-btn"
          aria-label={tr("Close source")}
          onClick={onClose}
        >
          <X size={20} />
        </button>
      </header>
      {source && (
        <>
          <div className="source-details">
            {metadata.source_host_kind === "mirror_copy" && (
              <span className="status gold">{tr("Mirror copy")}</span>
            )}
            <h2 id="source-drawer-title">
              {metadata.title || tr("EPFO document")}
            </h2>
            <code>{metadata.circular_no || tr("Identifier unavailable")}</code>
            <div className="source-facts">
              <span>
                {tr("Issued")}
                <strong>{metadata.date || tr("Date unavailable")}</strong>
              </span>
              <span>
                {tr("Source page")}
                <strong>
                  {metadata.page_provenance === "physical_pdf_page"
                    ? tr("PDF page {value0}", { value0: metadata.page_number })
                    : tr("Page number unverified")}
                </strong>
              </span>
            </div>
          </div>
          <div
            className="source-tabs"
            role="group"
            aria-label={tr("Source view")}
          >
            <button
              type="button"
              className={tab === "excerpt" ? "selected" : ""}
              aria-pressed={tab === "excerpt"}
              onClick={() => setTab("excerpt")}
            >
              {tr("Source text")}
            </button>
            <button
              type="button"
              className={tab === "pdf" ? "selected" : ""}
              aria-pressed={tab === "pdf"}
              disabled={!source.pdf_url}
              onClick={() => setTab("pdf")}
            >
              {tr("Original PDF")}
            </button>
          </div>
          {tab === "pdf" ? (
            <iframe
              title={tr("Original PDF source")}
              src={`${source.pdf_url}#page=${metadata.page_number || 1}`}
              className="pdf-viewer"
            />
          ) : (
            <pre className="excerpt">{source.text}</pre>
          )}
          <div className="source-foot">
            <span>
              {metadata.evidence_status || tr("Applicability unverified")}
            </span>
            {url && (
              <a href={url} target="_blank" rel="noreferrer">
                {tr("Open document")}
                <ArrowUpRight size={14} />
              </a>
            )}
          </div>
        </>
      )}
    </dialog>
  );
}
