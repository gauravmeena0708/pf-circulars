import React, { useState, useRef } from "react";
import Markdown from "./Markdown";
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
import { json, request, post, stream, saveResponse, setToken } from "./client";

export function Assistant({ language, onLanguage, enabled }) {
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
          <h2>Document & data assistant</h2>
          <select
            aria-label="File answer language"
            value={language}
            onChange={(e) => onLanguage(e.target.value)}
          >
            <option>English</option>
            <option>Hindi</option>
          </select>
          <p>
            Read a PDF or examine a CSV, then ask a question or draft an office
            note.
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
            ? "Extracting and checking the file…"
            : file
              ? "Replace the uploaded file"
              : "Choose a PDF or CSV"}
        </strong>
        <span>PDF: up to 50 MB / 100 pages · CSV: up to 25 MB</span>
        <input
          aria-label="Upload document or CSV"
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
                  ? `${file.page_count} pages · ${file.scanned_page_count} pages checked for OCR`
                  : `${file.profile.row_count.toLocaleString()} rows · ${file.profile.column_count} columns`}
              </small>
            </div>
            <div>
              <button
                className="secondary"
                onClick={() => exportFile("original")}
              >
                <Download size={14} />
                Original
              </button>
              <button className="secondary" onClick={() => exportFile("docx")}>
                <Download size={14} />
                Word
              </button>
              <button className="secondary" onClick={() => exportFile("text")}>
                <Download size={14} />
                Text
              </button>
              <button
                className="icon-btn"
                aria-label="Remove upload"
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
              <summary>Read extracted text and physical page markers</summary>
              <pre>{file.text}</pre>
            </details>
          ) : (
            <>
              <div className="data-controls">
                <form onSubmit={filter}>
                  <input
                    aria-label="Search dataset"
                    value={term}
                    onChange={(e) => setTerm(e.target.value)}
                    placeholder="Find an exact term across columns…"
                    maxLength={300}
                  />
                  <select
                    aria-label="Distribution column"
                    value={column}
                    onChange={(e) => setColumn(e.target.value)}
                  >
                    <option value="">Choose a distribution column</option>
                    {file.profile.columns.map((c) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                  <button className="secondary">Search / analyse</button>
                </form>
              </div>
              {rows && (
                <p className="data-caption">
                  {rows.total_matches.toLocaleString()} matching rows. Showing
                  up to the first 100.
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
                <summary>Schema, missing values & summary statistics</summary>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Missing values</th>
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
                    <h3>{column} · largest categories</h3>
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
              Ask about this file
            </h3>
            <button
              className="text-button"
              onClick={() => exportFile("transcript")}
            >
              <Download size={14} />
              Word transcript
            </button>
          </div>
          <p className="data-caption">
            Uploads expire after 30 minutes and are isolated to this browser
            session. Expired files are removed from memory automatically.
            Download anything you want to keep.
          </p>
          <div className="file-chat" aria-live="polite">
            {messages.map((m, i) => (
              <div key={i} className={"chat-message " + m.role}>
                <span>
                  {m.role === "user"
                    ? "YOUR QUESTION"
                    : "FILE-GROUNDED RESPONSE"}
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
              aria-label="Drafting mode"
              value={mode}
              onChange={(e) => setMode(e.target.value)}
            >
              <option value="answer">Question & answer</option>
              <option value="note">Formal office noting sheet</option>
            </select>
            <textarea
              aria-label="File question"
              required
              minLength={3}
              maxLength={6000}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about facts in this file, or describe the office note you need…"
            />
            <div>
              <span>
                {language} ·{" "}
                {enabled
                  ? "AI configured"
                  : "Add a token in Session settings for AI responses"}
              </span>
              <button className="primary" disabled={running || !enabled}>
                {running ? (
                  <LoaderCircle className="spin" size={15} />
                ) : (
                  <ArrowRight size={15} />
                )}{" "}
                {running ? "Generating…" : "Submit"}
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
          <h2>Office PDF utilities</h2>
          <p>
            Prepare, organise and extract documents. These operations do not use
            the language model.
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
            <strong>{label}</strong>
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
          {done}
        </div>
      )}
      <form onSubmit={run}>
        <label className="upload-zone">
          <Upload size={25} />
          <strong>
            {files.length
              ? `${files.length} file(s) selected`
              : operation === "images"
                ? "Choose image files"
                : "Choose PDF files"}
          </strong>
          <span>50 MB per file · 100 MB per operation · 20 files maximum</span>
          <input
            key={operation}
            aria-label="PDF utility files"
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
                <span>{(f.size / 1024).toFixed(0)} KB</span>
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
                      Move up
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
                      Move down
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
              PAGE NUMBERS / RANGES
              <input
                aria-label="Page ranges"
                value={pages}
                onChange={(e) => setPages(e.target.value)}
                placeholder={
                  operation === "extract"
                    ? "Required, e.g. 1,3-5"
                    : "All pages, or e.g. 1,3-5"
                }
                required={operation === "extract"}
              />
            </label>
          )}
          {operation === "rotate" && (
            <label>
              ROTATION
              <select
                aria-label="Rotation"
                value={rotation}
                onChange={(e) => setRotation(e.target.value)}
              >
                <option value="90">90° clockwise</option>
                <option value="180">180°</option>
                <option value="270">270° clockwise</option>
              </select>
            </label>
          )}
          {["render", "ocr"].includes(operation) && (
            <label>
              RESOLUTION
              <select
                aria-label="Resolution"
                value={dpi}
                onChange={(e) => setDpi(e.target.value)}
              >
                {[100, 150, 200].map((d) => (
                  <option value={d} key={d}>
                    {d} DPI
                  </option>
                ))}
              </select>
            </label>
          )}
          {operation === "ocr" && (
            <label>
              OCR LANGUAGE
              <select
                aria-label="OCR language"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
              >
                <option value="eng+hin">English + Hindi</option>
                <option value="eng">English</option>
                <option value="hin">Hindi</option>
              </select>
            </label>
          )}
        </div>
        <div className="tool-submit">
          <p>
            {operation === "ocr"
              ? "OCR supports up to 30 pages and downloads a searchable PDF, Word document and extracted text in a ZIP."
              : operation === "render"
                ? "Rendering supports up to 50 pages and downloads page images in a ZIP."
                : operation === "compress"
                  ? "Compression is lossless; the size reduction depends on the input document."
                  : "PDF operations support up to 250 pages per file and 300 pages combined."}
          </p>
          <button className="primary" disabled={!files.length || busy}>
            {busy ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Download size={16} />
            )}{" "}
            {busy ? "Processing…" : "Process & download"}
          </button>
        </div>
      </form>
    </section>
  );
}

export function SessionSettings({ hasToken, onToken }) {
  const [value, setValue] = useState(""),
    [saved, setSaved] = useState(false);
  return (
    <section className="workspace-panel settings-panel">
      <div className="workspace-intro">
        <Settings2 size={22} />
        <div>
          <h2>Session settings</h2>
          <p>Optional inference credentials for this browser visit.</p>
        </div>
      </div>
      <p>
        The hosted app uses its configured Hugging Face token. You can provide
        your own token when the shared service is unavailable or rate limited.
        The model and provider remain configured on the server.
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
          HUGGING FACE TOKEN
          <input
            aria-label="Hugging Face token"
            type="password"
            autoComplete="off"
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              setSaved(false);
            }}
            placeholder="hf_…"
            maxLength={300}
          />
        </label>
        <div>
          <button className="primary">Use for this visit</button>
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
            Clear token
          </button>
        </div>
      </form>
      <p className="data-caption">
        The token stays in JavaScript memory and is sent only in request headers
        to this app over HTTPS. It is not saved to local storage or included in
        exported files. Reloading the page clears it.
      </p>
      {saved && (
        <div className="success-box">
          <Check size={16} />
          {hasToken
            ? "Your session token is configured."
            : "Your session token was cleared."}
        </div>
      )}
    </section>
  );
}
