import {
  ArrowUpRight,
  BookOpen,
  Building2,
  FileText,
  Files,
  GitBranch,
  History,
  LibraryBig,
  Menu,
  Search,
  Settings2,
  Sparkles,
} from "lucide-react";
import { useI18n } from "./UiLanguage.jsx";
import { dateText } from "./lib";

// Navigation, the collection summary and this visit's recent questions.
export function Sidebar({
  page,
  menu,
  setPage,
  setMenu,
  policies,
  recent,
  catalog,
  newQuestion,
  setQuery,
  search,
}) {
  const { t: tr } = useI18n();
  return (
    <aside id="workspace-nav" className={"sidebar " + (menu ? "open" : "")}>
      <a
        className="brand"
        href="#"
        onClick={(e) => {
          e.preventDefault();
          newQuestion();
        }}
      >
        <span className="brand-symbol">
          <BookOpen size={23} />
        </span>
        <span>
          {tr("EPFO")}
          <span className="brand-sub">{tr("Knowledge assistant")}</span>
        </span>
      </a>
      <nav>
        <button
          className={page === "research" ? "active" : ""}
          onClick={newQuestion}
        >
          <Search size={18} />
          {tr("New question")}
          <span className="nav-shortcut">{tr("⌘ K")}</span>
        </button>
        <button
          className={page === "history" ? "active" : ""}
          onClick={() => {
            setPage("history");
            setMenu(false);
          }}
        >
          <GitBranch size={18} />
          {tr("Policy changes")}
          <span className="nav-count">{policies.length || "—"}</span>
        </button>
        <button
          className={page === "library" ? "active" : ""}
          onClick={() => {
            setPage("library");
            setMenu(false);
          }}
        >
          <LibraryBig size={18} />
          {tr("Browse policies")}
        </button>
        <button
          className={page === "divisions" ? "active" : ""}
          onClick={() => {
            setPage("divisions");
            setMenu(false);
          }}
        >
          <Building2 size={18} />
          {tr("Browse by division")}
        </button>
        <button
          className={page === "assistant" ? "active" : ""}
          onClick={() => {
            setPage("assistant");
            setMenu(false);
          }}
        >
          <FileText size={18} />
          {tr("File assistant")}
        </button>
        <button
          className={page === "tools" ? "active" : ""}
          onClick={() => {
            setPage("tools");
            setMenu(false);
          }}
        >
          <Files size={18} />
          {tr("PDF tools")}
        </button>
        <button
          className={page === "settings" ? "active" : ""}
          onClick={() => {
            setPage("settings");
            setMenu(false);
          }}
        >
          <Settings2 size={18} />
          {tr("Settings")}
        </button>
      </nav>
      <div className="sidebar-divider" />
      <div className="sidebar-label">{tr("Recent questions")}</div>
      <div className="recent">
        {recent.length ? (
          recent.map((q) => (
            <button
              key={q}
              onClick={(e) => {
                setQuery(q);
                search(e, q);
              }}
            >
              <History size={14} />
              <span>{q}</span>
            </button>
          ))
        ) : (
          <p>
            {tr("Your recent questions will")}
            <br />
            {tr("appear here during this visit.")}
          </p>
        )}
      </div>
      <div className="sidebar-bottom">
        <div className="collection">
          <span className="collection-icon">
            <LibraryBig size={18} />
          </span>
          <div>
            {tr("Research collection")}
            <small>
              {catalog?.passage_count?.toLocaleString() || tr("Loading…")}{" "}
              {tr("searchable passages")}
              {catalog?.corpus_as_of && (
                <>
                  {tr(" · circulars to ")}
                  {dateText(catalog.corpus_as_of)}
                </>
              )}
            </small>
          </div>
        </div>
        <div className="independent">
          <span />
          {tr("Independent research tool")}
          <small>{tr("Not an official EPFO service")}</small>
        </div>
      </div>
    </aside>
  );
}

export function Topbar({ page, menu, setMenu }) {
  const { t: tr, locale, setLocale } = useI18n();
  return (
    <header className="topbar">
      <button
        className="mobile-menu icon-btn"
        aria-label={tr("Open navigation")}
        aria-expanded={menu}
        aria-controls="workspace-nav"
        onClick={() => setMenu(!menu)}
      >
        <Menu size={20} />
      </button>
      <div className="breadcrumb">
        <strong>
          {tr(
            {
              research: "Ask about EPFO rules",
              history: "Policy changes",
              library: "Browse policies",
              divisions: "Browse by division",
              assistant: "File assistant",
              tools: "PDF tools",
              settings: "Settings",
            }[page],
          )}
        </strong>
      </div>
      <div className="top-actions">
        <div
          className="interface-language"
          role="group"
          aria-label={tr("Interface language")}
        >
          <button
            type="button"
            lang="en"
            aria-pressed={locale === "en"}
            onClick={() => setLocale("en")}
          >
            English
          </button>
          <button
            type="button"
            lang="hi"
            aria-pressed={locale === "hi"}
            onClick={() => setLocale("hi")}
          >
            हिन्दी
          </button>
        </div>
        <span className="preview-tag">
          {tr("RESEARCH & OFFICE WORKSPACE")}
        </span>
        <a
          href="https://www.epfo.gov.in/circulars"
          target="_blank"
          rel="noreferrer"
        >
          {tr("EPFO circulars")}
          <ArrowUpRight size={14} />
        </a>
      </div>
    </header>
  );
}

export function PageHeading({ page }) {
  const { t: tr } = useI18n();
  return (
    <div className="page-heading">
      <div>
        {page === "research" && (
          <span className="welcome-icon" aria-hidden="true">
            <Sparkles size={25} strokeWidth={1.5} />
          </span>
        )}
        <h1>
          {tr(
            {
              research: "How can I help with EPFO?",
              history: "Follow the policy through time.",
              library: "Browse policies by topic.",
              divisions: "Find policies by EPFO division.",
              assistant: "Work from the document.",
              tools: "Prepare documents with care.",
              settings: "Your workspace preferences.",
            }[page],
          )}
        </h1>
        <p>
          {tr(
            {
              research:
                "Ask about PF, pensions, employer duties or staff matters.",
              history:
                "Review issued instructions, amendments and the links that still need verification.",
              library:
                "Explore policy changes, circulars and statutory manuals by topic.",
              divisions:
                "Choose a division to explore its policy topics and source circulars.",
              assistant:
                "Review a PDF, explore a CSV, and draft file-grounded notes in English or Hindi.",
              tools:
                "Merge, extract, rotate, compress and OCR your office documents.",
              settings:
                "Manage your optional personal token for this visit.",
            }[page],
          )}
        </p>
      </div>
    </div>
  );
}
