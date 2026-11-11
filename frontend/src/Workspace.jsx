import { useI18n } from "./UiLanguage.jsx";
import React, { useState, useRef } from "react";
import Markdown from "./Markdown";
import DownloadMenu from "./DownloadMenu.jsx";
import {
  Upload,
  FileText,
  Download,
  LoaderCircle,
  ArrowRight,
  Trash2,
  Files,
  MessageSquare,
  Settings2,
  Check,
} from "lucide-react";
import { json, request, stream, saveResponse, setToken } from "./client";

export function Assistant({ language, onLanguage, enabled }) {
  const { t: tr } = useI18n();
  const [file, setFile] = useState(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [question, setQuestion] = useState(""),
    [mode, setMode] = useState("answer"),
    [messages, setMessages] = useState([]),
    [running, setRunning] = useState(false),
    [term, setTerm] = useState(""),
    [column, setColumn] = useState(""),
    [rows, setRows] = useState(null);
  const abort = useRef(null),
    sequence = useRef(0);
  async function upload(selected) {
    if (!selected) return;
    if (
      selected.size >
      (selected.name.toLowerCase().endsWith(".csv") ? 25 : 50) * 1024 * 1024
    ) {
      setError("PDFs may be up to 50 MB; CSVs up to 25 MB.");
      return;
    }
    const seq = ++sequence.current;
    abort.current?.abort();
    setRunning(false);
    setBusy(true);
    setError("");
    try {
      if (file)
        await request("/api/uploads/" + file.id, { method: "DELETE" }).catch(
          () => {},
        );
      const form = new FormData();
      form.append("file", selected);
      const result = await json("/api/uploads", { method: "POST", body: form });
      if (seq !== sequence.current) return;
      setFile(result);
      setMessages([]);
      setRows(null);
      setTerm("");
      setColumn("");
    } catch (e) {
      setFile(null);
      setError(e.message);
    } finally {
      if (seq === sequence.current) setBusy(false);
    }
  }
  async function remove() {
    sequence.current++;
    abort.current?.abort();
    setRunning(false);
    try {
      if (file) await request("/api/uploads/" + file.id, { method: "DELETE" });
      setFile(null);
      setMessages([]);
      setRows(null);
    } catch (e) {
      setError(e.message);
    }
  }
  async function ask(e) {
    e.preventDefault();
    if (!file || running) return;
    const controller = new AbortController();
    abort.current = controller;
    const seq = sequence.current;
    setRunning(true);
    setError("");
    setMessages((v) => [
      ...v,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);
    const text = question;
    setQuestion("");
    try {
      await stream(
        "/api/uploads/" + file.id + "/chat",
        { question: text, language, mode },
        (part) => {
          if (seq === sequence.current)
            setMessages((v) =>
              v.map((m, i) =>
                i === v.length - 1 ? { ...m, content: m.content + part } : m,
              ),
            );
        },
        controller.signal,
      );
    } catch (e) {
      if (e.name !== "AbortError" && seq === sequence.current)
        setError(e.message);
    } finally {
      if (seq === sequence.current) setRunning(false);
    }
  }
  async function exportFile(format) {
    try {
      await saveResponse(
        await request(`/api/uploads/${file.id}/download?format=${format}`),
        "document",
      );
    } catch (e) {
      setError(e.message);
    }
  }
  async function filter(e) {
    e?.preventDefault();
    try {
      setRows(
        await json(
          `/api/uploads/${file.id}/rows?term=${encodeURIComponent(term)}&column=${encodeURIComponent(column)}`,
        ),
      );
    } catch (e) {
      setError(e.message);
    }
  }
  return (
    <section className="workspace-panel">
      <div className="workspace-intro">
        <FileText size={21} />
        <div>
          <h2>{tr("File assistant")}</h2>
          <select
            aria-label={tr("File answer language")}
            value={language}
            onChange={(e) => onLanguage(e.target.value)}
          >
            <option value="English">{tr("English")}</option>
            <option value="Hindi">{tr("Hindi")}</option>
          </select>
          <p>
            {tr(
              "Read a PDF or examine a CSV, then ask a question or draft an office note.",
            )}
          </p>
        </div>
      </div>
      {error && (
        <div className="error-box" role="alert">
          {error}
        </div>
      )}
      <label className={"upload-zone " + (busy ? "processing" : "")}>
        <Upload size={25} />
        <strong>
          {busy
            ? tr("Extracting and checking the file…")
            : file
              ? tr("Replace the uploaded file")
              : tr("Choose a PDF or CSV")}
        </strong>
        <span>{tr("PDF: up to 50 MB / 100 pages · CSV: up to 25 MB")}</span>
        <input
          aria-label={tr("Upload document or CSV")}
          type="file"
          accept=".pdf,.csv"
          disabled={busy || running}
          onChange={(e) => {
            upload(e.target.files[0]);
            e.target.value = "";
          }}
        />
      </label>
      {file && (
        <>
          <div className="file-toolbar">
            <div>
              <strong>{file.name}</strong>
              <small>
                {file.kind === "pdf"
                  ? tr("{value0} pages · {value1} pages checked for OCR", {
                      value0: file.page_count,
                      value1: file.scanned_page_count,
                    })
                  : tr("{value0} rows · {value1} columns", {
                      value0: file.profile.row_count.toLocaleString(),
                      value1: file.profile.column_count,
                    })}
              </small>
            </div>
            <div>
              <DownloadMenu
                label="Download file"
                disabled={running || busy}
                items={[
                  {
                    id: "docx",
                    label: "Word document (.docx)",
                    description: "Editable document",
                    onSelect: () => exportFile("docx"),
                  },
                  {
                    id: "pdf",
                    label: "PDF document (.pdf)",
                    description: "Ready to print or share",
                    onSelect: () => exportFile("pdf"),
                  },
                  {
                    id: "original",
                    label: "Original file",
                    onSelect: () => exportFile("original"),
                  },
                  {
                    id: "text",
                    label: "Plain text (.txt)",
                    onSelect: () => exportFile("text"),
                  },
                ]}
              />
              <button
                className="icon-btn"
                aria-label={tr("Remove upload")}
                onClick={remove}
              >
                <Trash2 size={16} />
              </button>
            </div>
          </div>
          {(file.warnings || []).map((warning, i) => (
            <div className="evidence-notice" key={i}>
              {warning}
            </div>
          ))}
          {file.kind === "pdf" ? (
            <details className="document-text">
              <summary>{tr("Read extracted text and page numbers")}</summary>
              <pre>{file.text}</pre>
            </details>
          ) : (
            <>
              <div className="data-controls">
                <form onSubmit={filter}>
                  <input
                    aria-label={tr("Search dataset")}
                    value={term}
                    onChange={(e) => setTerm(e.target.value)}
                    placeholder={tr("Find an exact term across columns…")}
                    maxLength={300}
                  />
                  <select
                    aria-label={tr("Distribution column")}
                    value={column}
                    onChange={(e) => setColumn(e.target.value)}
                  >
                    <option value="">
                      {tr("Choose a distribution column")}
                    </option>
                    {file.profile.columns.map((c) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                  <button className="secondary">
                    {tr("Search / analyse")}
                  </button>
                </form>
              </div>
              {rows && (
                <p className="data-caption">
                  {rows.total_matches.toLocaleString()}
                  {tr(" matching rows. Showing up to the first 100.")}
                </p>
              )}
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      {file.profile.columns.map((c) => (
                        <th key={c}>{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(rows?.rows || file.preview).map((r, i) => (
                      <tr key={i}>
                        {file.profile.columns.map((c) => (
                          <td key={c}>{r[c] === null ? "—" : String(r[c])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <details className="document-text">
                <summary>
                  {tr("Schema, missing values & summary statistics")}
                </summary>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>{tr("Column")}</th>
                        <th>{tr("Type")}</th>
                        <th>{tr("Missing values")}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {file.profile.columns.map((c) => (
                        <tr key={c}>
                          <td>{c}</td>
                          <td>{file.profile.dtypes[c]}</td>
                          <td>{file.profile.null_counts[c]}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <pre>{JSON.stringify(file.profile.summary_stats, null, 2)}</pre>
              </details>
              {rows?.distribution &&
                Object.keys(rows.distribution).length > 0 && (
                  <div className="distribution">
                    <h3>
                      {column}
                      {tr(" · largest categories")}
                    </h3>
                    {Object.entries(rows.distribution).map(([key, value]) => (
                      <div key={key}>
                        <span>{key}</span>
                        <meter
                          min="0"
                          max={file.profile.row_count}
                          value={value}
                        />
                        <strong>{value}</strong>
                      </div>
                    ))}
                  </div>
                )}
            </>
          )}
          <div className="chat-heading">
            <h3>
              <MessageSquare size={17} />
              {tr("Ask about this file")}
            </h3>
            <DownloadMenu
              label="Download response"
              disabled={
                running ||
                !messages.some(
                  (message) => message.role === "assistant" && message.content,
                )
              }
              items={[
                {
                  id: "docx",
                  label: "Word document (.docx)",
                  description: "Latest response · editable",
                  onSelect: () => exportFile("response_docx"),
                },
                {
                  id: "pdf",
                  label: "PDF document (.pdf)",
                  description: "Latest response · ready to share",
                  onSelect: () => exportFile("response_pdf"),
                },
                {
                  id: "conversation-docx",
                  label: "Word conversation (.docx)",
                  description: "Recent questions and responses",
                  onSelect: () => exportFile("transcript"),
                },
                {
                  id: "conversation-pdf",
                  label: "PDF conversation (.pdf)",
                  description: "Recent questions and responses",
                  onSelect: () => exportFile("transcript_pdf"),
                },
              ]}
            />
          </div>
          <p className="data-caption">
            {tr(
              "Uploads expire after 30 minutes and are isolated to this browser session. Expired files are removed from memory automatically. Download anything you want to keep.",
            )}
          </p>
          <div className="file-chat" aria-live="polite">
            {messages.map((m, i) => (
              <div key={i} className={"chat-message " + m.role}>
                <span>
                  {m.role === "user"
                    ? tr("Your question")
                    : tr("From your file")}
                </span>
                <div className="markdown">
                  <Markdown>
                    {m.content ||
                      (running
                        ? "Preparing the response…"
                        : "No response received.")}
                  </Markdown>
                </div>
              </div>
            ))}
          </div>
          <form className="chat-form" onSubmit={ask}>
            <select
              aria-label={tr("Drafting mode")}
              value={mode}
              onChange={(e) => setMode(e.target.value)}
            >
              <option value="answer">{tr("Question & answer")}</option>
              <option value="note">{tr("Formal office noting sheet")}</option>
            </select>
            <textarea
              aria-label={tr("File question")}
              required
              minLength={3}
              maxLength={6000}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={tr(
                "Ask about facts in this file, or describe the office note you need…",
              )}
            />
            <div>
              <span>
                {language} ·{" "}
                {enabled
                  ? tr("AI configured")
                  : tr("Add a token in Settings for AI responses")}
              </span>
              <button className="primary" disabled={running || !enabled}>
                {running ? (
                  <LoaderCircle className="spin" size={15} />
                ) : (
                  <ArrowRight size={15} />
                )}{" "}
                {running ? tr("Generating…") : tr("Submit")}
              </button>
            </div>
          </form>
        </>
      )}
    </section>
  );
}

const OPERATIONS = [
  ["merge", "Merge PDFs"],
  ["extract", "Extract selected pages"],
  ["split", "Split into individual PDFs"],
  ["rotate", "Rotate pages"],
  ["compress", "Compress PDF"],
  ["images", "Images to PDF"],
  ["render", "PDF to images"],
  ["ocr", "OCR scanned PDF"],
];
export function PdfTools() {
  const { t: tr } = useI18n();
  const [operation, setOperation] = useState("merge"),
    [files, setFiles] = useState([]),
    [pages, setPages] = useState(""),
    [rotation, setRotation] = useState("90"),
    [dpi, setDpi] = useState("150"),
    [language, setLanguage] = useState("eng+hin"),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [done, setDone] = useState("");
  async function run(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    setDone("");
    try {
      if (
        files.some((f) => f.size > 50 * 1024 * 1024) ||
        files.reduce((n, f) => n + f.size, 0) > 99 * 1024 * 1024
      )
        throw Error(
          "Each file may be up to 50 MB; the combined upload must be under 100 MB.",
        );
      const form = new FormData();
      for (const f of files) form.append("files", f);
      form.append("operation", operation);
      form.append("pages", pages);
      form.append("rotation", rotation);
      form.append("dpi", dpi);
      form.append("language", language);
      await saveResponse(
        await request("/api/pdf-tools", { method: "POST", body: form }),
        "processed-document",
      );
      setDone("Your processed document has been downloaded.");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="workspace-panel">
      <div className="workspace-intro">
        <Files size={23} />
        <div>
          <h2>{tr("PDF tools")}</h2>
          <p>
            {tr(
              "Prepare and organise your PDFs, or make scanned pages searchable.",
            )}
          </p>
        </div>
      </div>
      <div className="tool-grid">
        {OPERATIONS.map(([id, label], i) => (
          <button
            className={operation === id ? "active" : ""}
            key={id}
            onClick={() => {
              setOperation(id);
              setFiles([]);
              setError("");
              setDone("");
            }}
          >
            <span>{String(i + 1).padStart(2, "0")}</span>
            <strong>{tr(label)}</strong>
          </button>
        ))}
      </div>
      {error && (
        <div className="error-box" role="alert">
          {error}
        </div>
      )}
      {done && (
        <div className="success-box" role="status">
          <Check size={17} />
          {tr(done)}
        </div>
      )}
      <form onSubmit={run}>
        <label className="upload-zone">
          <Upload size={25} />
          <strong>
            {files.length
              ? tr("{value0} file(s) selected", { value0: files.length })
              : operation === "images"
                ? tr("Choose image files")
                : tr("Choose PDF files")}
          </strong>
          <span>
            {tr("50 MB per file · 100 MB per operation · 20 files maximum")}
          </span>
          <input
            key={operation}
            aria-label={tr("PDF utility files")}
            type="file"
            required
            accept={
              operation === "images"
                ? ".png,.jpg,.jpeg,.webp,.tiff,.bmp"
                : ".pdf"
            }
            multiple={["merge", "images"].includes(operation)}
            onChange={(e) => setFiles(Array.from(e.target.files))}
          />
        </label>
        {files.length > 0 && (
          <ol className="selected-files">
            {files.map((f, i) => (
              <li key={i}>
                {f.name}
                <span>
                  {(f.size / 1024).toFixed(0)}
                  {tr(" KB")}
                </span>
                {["merge", "images"].includes(operation) && (
                  <div>
                    <button
                      type="button"
                      disabled={!i}
                      onClick={() =>
                        setFiles((v) => {
                          const a = [...v];
                          [a[i - 1], a[i]] = [a[i], a[i - 1]];
                          return a;
                        })
                      }
                    >
                      {tr("Move up")}
                    </button>
                    <button
                      type="button"
                      disabled={i === files.length - 1}
                      onClick={() =>
                        setFiles((v) => {
                          const a = [...v];
                          [a[i + 1], a[i]] = [a[i], a[i + 1]];
                          return a;
                        })
                      }
                    >
                      {tr("Move down")}
                    </button>
                  </div>
                )}
              </li>
            ))}
          </ol>
        )}
        <div className="tool-options">
          {["extract", "rotate"].includes(operation) && (
            <label>
              {tr("Page numbers / ranges")}
              <input
                aria-label={tr("Page ranges")}
                value={pages}
                onChange={(e) => setPages(e.target.value)}
                placeholder={
                  operation === "extract"
                    ? tr("Required, e.g. 1,3-5")
                    : tr("All pages, or e.g. 1,3-5")
                }
                required={operation === "extract"}
              />
            </label>
          )}
          {operation === "rotate" && (
            <label>
              {tr("Rotation")}
              <select
                aria-label={tr("Rotation")}
                value={rotation}
                onChange={(e) => setRotation(e.target.value)}
              >
                <option value="90">{tr("90° clockwise")}</option>
                <option value="180">180°</option>
                <option value="270">{tr("270° clockwise")}</option>
              </select>
            </label>
          )}
          {["render", "ocr"].includes(operation) && (
            <label>
              {tr("Resolution")}
              <select
                aria-label={tr("Resolution")}
                value={dpi}
                onChange={(e) => setDpi(e.target.value)}
              >
                {[100, 150, 200].map((d) => (
                  <option value={d} key={d}>
                    {d}
                    {tr(" DPI")}
                  </option>
                ))}
              </select>
            </label>
          )}
          {operation === "ocr" && (
            <label>
              {tr("OCR language")}
              <select
                aria-label={tr("OCR language")}
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
              >
                <option value="eng+hin">{tr("English + Hindi")}</option>
                <option value="eng">{tr("English")}</option>
                <option value="hin">{tr("Hindi")}</option>
              </select>
            </label>
          )}
        </div>
        <div className="tool-submit">
          <p>
            {operation === "ocr"
              ? tr(
                  "OCR supports up to 30 pages and downloads a searchable PDF, Word document and extracted text in a ZIP.",
                )
              : operation === "render"
                ? tr(
                    "Rendering supports up to 50 pages and downloads page images in a ZIP.",
                  )
                : operation === "compress"
                  ? tr(
                      "Compression is lossless; the size reduction depends on the input document.",
                    )
                  : tr(
                      "PDF operations support up to 250 pages per file and 300 pages combined.",
                    )}
          </p>
          <button className="primary" disabled={!files.length || busy}>
            {busy ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Download size={16} />
            )}{" "}
            {busy ? tr("Processing…") : tr("Process & download")}
          </button>
        </div>
      </form>
    </section>
  );
}

export function SessionSettings({ hasToken, onToken }) {
  const { t: tr } = useI18n();
  const [value, setValue] = useState(""),
    [saved, setSaved] = useState(false);
  return (
    <section className="workspace-panel settings-panel">
      <div className="workspace-intro">
        <Settings2 size={22} />
        <div>
          <h2>{tr("Settings")}</h2>
          <p>{tr("Use a personal Hugging Face token if you need one.")}</p>
        </div>
      </div>
      <p>
        {tr(
          "The hosted app uses its configured Hugging Face token. You can provide your own token when the shared service is unavailable or rate limited. The model and provider remain configured on the server.",
        )}
      </p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          setToken(value);
          onToken(Boolean(value.trim()));
          setValue("");
          setSaved(true);
        }}
      >
        <label>
          {tr("Hugging Face token")}
          <input
            aria-label={tr("Hugging Face token")}
            type="password"
            autoComplete="off"
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              setSaved(false);
            }}
            placeholder={tr("hf_…")}
            maxLength={300}
          />
        </label>
        <div>
          <button className="primary">{tr("Use for this visit")}</button>
          <button
            type="button"
            className="secondary"
            onClick={() => {
              setToken("");
              onToken(false);
              setValue("");
              setSaved(false);
            }}
          >
            {tr("Clear token")}
          </button>
        </div>
      </form>
      <p className="data-caption">
        {tr(
          "The token stays in JavaScript memory and is sent only in request headers to this app over HTTPS. It is not saved to local storage or included in exported files. Reloading the page clears it.",
        )}
      </p>
      {saved && (
        <div className="success-box">
          <Check size={16} />
          {hasToken
            ? tr("Your session token is configured.")
            : tr("Your session token was cleared.")}
        </div>
      )}
    </section>
  );
}
