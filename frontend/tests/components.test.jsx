import { describe, expect, test, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import AnswerFeedback from "../src/AnswerFeedback.jsx";
import HistoryPage from "../src/HistoryPage.jsx";
import { PageHeading, Sidebar, Topbar } from "../src/Layout.jsx";
import LibraryPage from "../src/LibraryPage.jsx";
import ResearchComposer from "../src/ResearchComposer.jsx";
import ResearchResults from "../src/ResearchResults.jsx";
import Timeline from "../src/Timeline.jsx";
import { I18nProvider } from "../src/UiLanguage.jsx";

const show = (element) => render(<I18nProvider>{element}</I18nProvider>);
const noop = () => {};

const POLICIES = [
  {
    id: "bank_kyc",
    label: "Bank KYC / cheque upload",
    scope: "Bank account seeding and cheque upload",
    topics: [],
    first_issued: "2024-05-28",
    last_issued: "2025-04-03",
    checked_count: 2,
    document_count: 2,
  },
  {
    id: "delegation",
    label: "Delegation of powers",
    scope: "Administrative and financial powers",
    topics: [{ id: "pid", label: "Physical infrastructure" }],
    first_issued: "2012-03-05",
    last_issued: "2020-01-17",
    checked_count: 9,
    document_count: 11,
  },
];
const CATALOG = { today: "2026-11-11", answer_enabled: true, policies: POLICIES, passage_count: 82919 };

const DOCUMENTS = [
  { id: "bank-2024", issued_on: "2024-05-28", title: "Conditional relaxation", circular_no: "WSU/1", status: "source_checked" },
  {
    id: "bank-2025",
    issued_on: "2025-04-03",
    title: "Employer approval removed",
    circular_no: "WSU/2",
    status: "source_unverified",
    effective_from: "2025-04-03",
    pdf_url: "/api/policies/bank_kyc/documents/bank-2025/pdf",
  },
];
const RELATIONSHIPS = [
  { id: "r1", source: "bank-2025", target: "bank-2024", type: "partially_modifies", verification: "source_checked", scope: "Upload rule" },
];

const RESULT = {
  packet_id: "a".repeat(32),
  question: "Is employer approval still required?",
  as_of: "2025-04-03",
  selection_mode: "auto",
  currency: null,
  policy_context: {
    policy_id: "bank_kyc",
    policy_label: "Bank KYC / cheque upload",
    documents: DOCUMENTS,
    relationships: RELATIONSHIPS,
    selected_topics: [],
    topics: [],
    limitations: ["The 2024 circular's target is unverified."],
    fallback_reason: "Curated coverage is incomplete.",
  },
  sources: [
    { id: 1, text: "No employer approval.", document_id: "bank-2025", metadata: { title: "Employer approval removed", date: "2025-04-03", page_number: 2, page_provenance: "physical_pdf_page" } },
    { id: 2, text: "Older text.", document_id: null, metadata: { title: "An older circular", date: "12/01/2019", page_number: "1-3" } },
  ],
};

function results(overrides = {}) {
  const props = {
    result: RESULT,
    answer: "Employer approval is no longer required [1].",
    generating: false,
    canGenerate: true,
    generate: vi.fn(),
    source: RESULT.sources[0],
    sourceOpen: false,
    setSourceOpen: vi.fn(),
    selectSource: vi.fn(),
    resultTab: "answer",
    setResultTab: vi.fn(),
    exportResearch: vi.fn(),
    exportReport: vi.fn(),
    setError: vi.fn(),
    ...overrides,
  };
  // Same renderer mapping as App: a #source-n link becomes a button.
  props.answerComponents = {
    a: ({ href, children }) =>
      href?.startsWith("#source-") ? (
        <button className="citation" onClick={() => props.selectSource(href.slice(8))}>
          {children}
        </button>
      ) : (
        <a href={href}>{children}</a>
      ),
  };
  show(<ResearchResults {...props} />);
  return props;
}

describe("research results", () => {
  test("show the detected policy, the evidence date and a clickable citation", () => {
    const props = results();
    expect(screen.getByRole("heading", { name: RESULT.question })).toBeTruthy();
    expect(screen.getByText("Detected policy:", { exact: false }).textContent).toContain("03 Apr 2025");
    fireEvent.click(screen.getByRole("button", { name: "1" }));
    expect(props.selectSource).toHaveBeenCalledWith("1");
    expect(screen.getByText("Curated coverage is incomplete.")).toBeTruthy();
  });

  test("the sources tab states a page only when it is a verified PDF page", () => {
    results({ resultTab: "sources" });
    const list = document.querySelector(".source-list");
    expect(within(list).getByText(/PDF page 2/)).toBeTruthy();
    expect(within(list).getByText(/Page number unverified/)).toBeTruthy();
    expect(list.textContent).not.toContain("1-3");
  });

  test("the history tab opens the source that belongs to a document", () => {
    const props = results({ resultTab: "history" });
    fireEvent.click(screen.getAllByRole("button", { name: /Review source/ })[0]);
    expect(props.selectSource).toHaveBeenCalledWith(1);
  });

  test("a search of all documents says the date limit was not applied", () => {
    results({ result: { ...RESULT, policy_context: null }, answer: "" });
    expect(screen.getByText(/No policy topic matched/).textContent).toContain("The date limit was not applied.");
    expect(screen.getByRole("button", { name: /Get answer with references/ }).disabled).toBe(false);
  });

  test("without inference the answer button is disabled and the reason is shown", () => {
    results({ answer: "", canGenerate: false });
    expect(screen.getByRole("button", { name: /Get answer with references/ }).disabled).toBe(true);
    expect(screen.getByText(/AI generation is not configured/)).toBeTruthy();
  });

  test("no feedback is offered before there is an answer", () => {
    results({ answer: "" });
    expect(screen.queryByText("Was this answer helpful?")).toBeNull();
  });
});

describe("answer feedback", () => {
  const sent = () => vi.mocked(fetch).mock.calls.map(([path, options]) => [path, JSON.parse(options.body)]);

  test("a yes is sent at once and thanked", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response('{"status":"recorded"}')));
    show(<AnswerFeedback packetId="p1" />);
    fireEvent.click(screen.getByRole("button", { name: "Yes, this answer helped" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("Thank you"));
    expect(sent()).toEqual([["/api/feedback", { packet_id: "p1", helpful: true, comment: "" }]]);
  });

  test("a no asks what was wrong and sends the note with the rating", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response('{"status":"recorded"}')));
    show(<AnswerFeedback packetId="p1" />);
    fireEvent.click(screen.getByRole("button", { name: "No, this answer did not help" }));
    expect(fetch).not.toHaveBeenCalled();
    fireEvent.change(screen.getByPlaceholderText("What was wrong? (optional)"), { target: { value: " Wrong circular " } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(screen.getByRole("status")).toBeTruthy());
    expect(sent()).toEqual([["/api/feedback", { packet_id: "p1", helpful: false, comment: "Wrong circular" }]]);
  });

  test("a failed send is reported and can be retried", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response('{"detail":"gone"}', { status: 410 })));
    show(<AnswerFeedback packetId="p1" />);
    fireEvent.click(screen.getByRole("button", { name: "Yes, this answer helped" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("could not be sent"));
    expect(screen.getByRole("button", { name: "Yes, this answer helped" })).toBeTruthy();
  });

  test("a rating belongs to one answer", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response('{"status":"recorded"}')));
    const view = show(<AnswerFeedback packetId="p1" />);
    fireEvent.click(screen.getByRole("button", { name: "Yes, this answer helped" }));
    await waitFor(() => expect(screen.getByRole("status")).toBeTruthy());
    view.rerender(
      <I18nProvider>
        <AnswerFeedback packetId="p2" />
      </I18nProvider>,
    );
    expect(screen.getByText("Was this answer helpful?")).toBeTruthy();
  });
});

describe("question composer", () => {
  function composer(overrides = {}) {
    const props = {
      input: { current: null },
      researchOptions: { current: null },
      catalog: CATALOG,
      policies: POLICIES,
      selected: undefined,
      hasToken: false,
      busy: false,
      generating: false,
      result: null,
      query: "",
      setQuery: vi.fn(),
      language: "English",
      setLanguage: vi.fn(),
      detail: "brief",
      setDetail: vi.fn(),
      policy: "auto",
      setPolicy: vi.fn(),
      queryTopic: "",
      setQueryTopic: vi.fn(),
      day: CATALOG.today,
      setDay: vi.fn(),
      search: vi.fn((e) => e?.preventDefault?.()),
      setPage: vi.fn(),
      ...overrides,
    };
    show(<ResearchComposer {...props} />);
    return props;
  }

  test("a starter task asks its question with automatic policy detection", () => {
    const props = composer();
    fireEvent.click(screen.getByRole("button", { name: "Transfer PF" }));
    expect(props.setPolicy).toHaveBeenCalledWith("auto");
    expect(props.search.mock.calls[0].slice(1)).toEqual(["Can a transfer claim be rejected for overlapping service?", "auto"]);
  });

  test("custom filters are flagged, and subtopics appear only for a policy that has them", () => {
    composer({ policy: "delegation", selected: POLICIES[1] });
    expect(screen.getByText("Custom filters")).toBeTruthy();
    expect(screen.getByLabelText("Policy subtopic")).toBeTruthy();
  });

  test("with default filters there is no flag and no subtopic control", () => {
    composer();
    expect(screen.queryByText("Custom filters")).toBeNull();
    expect(screen.queryByLabelText("Policy subtopic")).toBeNull();
    expect(screen.getByLabelText("Policy collection").querySelectorAll("option").length).toBe(POLICIES.length + 2);
  });

  test("starter tasks give way to results, and searching disables the buttons", () => {
    composer({ result: RESULT, busy: true });
    expect(screen.queryByRole("button", { name: "Transfer PF" })).toBeNull();
    expect(screen.getByRole("button", { name: /Finding sources/ }).disabled).toBe(true);
    expect(screen.getByRole("button", { name: /Search documents only/ }).disabled).toBe(true);
  });

  test("says so when answers are unavailable", () => {
    composer({ catalog: { ...CATALOG, answer_enabled: false } });
    expect(screen.getByRole("status").textContent).toContain("Answers are unavailable right now");
  });
});

describe("layout", () => {
  test("the sidebar navigates, lists recent questions and shows the collection size", () => {
    const props = {
      page: "research",
      menu: true,
      setPage: vi.fn(),
      setMenu: vi.fn(),
      policies: POLICIES,
      recent: ["What is CPPS?"],
      catalog: CATALOG,
      newQuestion: vi.fn(),
      setQuery: vi.fn(),
      search: vi.fn(),
    };
    show(<Sidebar {...props} />);
    fireEvent.click(screen.getByRole("button", { name: /Policy changes/ }));
    expect(props.setPage).toHaveBeenCalledWith("history");
    expect(props.setMenu).toHaveBeenCalledWith(false);
    fireEvent.click(screen.getByRole("button", { name: "What is CPPS?" }));
    expect(props.setQuery).toHaveBeenCalledWith("What is CPPS?");
    expect(props.search.mock.calls[0][1]).toBe("What is CPPS?");
    expect(screen.getByText(/82,919/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /New question/ }));
    expect(props.newQuestion).toHaveBeenCalled();
  });

  test("the interface language switch translates the page heading", () => {
    show(
      <>
        <Topbar page="library" menu={false} setMenu={noop} />
        <PageHeading page="library" />
      </>,
    );
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Browse policies by topic.");
    fireEvent.click(screen.getByRole("button", { name: "हिन्दी" }));
    expect(screen.getByRole("heading", { level: 1 }).textContent).not.toBe("Browse policies by topic.");
    expect(document.documentElement.lang).toBe("hi");
  });
});

describe("policy pages", () => {
  test("the timeline lists the newest document first and marks unverified originals", () => {
    const onDocument = vi.fn();
    show(<Timeline documents={DOCUMENTS} relationships={RELATIONSHIPS} onDocument={onDocument} />);
    const items = document.querySelectorAll(".timeline-item");
    expect(items[0].textContent).toContain("Employer approval removed");
    expect(items[0].textContent).toContain("Most recent in inventory");
    expect(items[0].className).toContain("unverified");
    expect(items[0].textContent).toContain("partially modifies");
    expect(items[1].textContent).toContain("Commencement unverified");
    // Only the document with a PDF can be opened.
    expect(screen.getAllByRole("button", { name: /Review source/ }).length).toBe(1);
    fireEvent.click(screen.getByRole("button", { name: /Review source/ }));
    expect(onDocument).toHaveBeenCalledWith(DOCUMENTS[1]);
  });

  test("the history page shows limitations and hands the policy to the composer", () => {
    const props = {
      catalog: CATALOG,
      policies: POLICIES,
      history: { label: "Delegation of powers", scope: "Powers", documents: DOCUMENTS, relationships: RELATIONSHIPS, limitations: ["One original is missing."] },
      selectedHistory: "delegation",
      setSelectedHistory: vi.fn(),
      historyTopic: "pid",
      setHistoryTopic: vi.fn(),
      day: CATALOG.today,
      setDay: vi.fn(),
      setPolicy: vi.fn(),
      setQueryTopic: vi.fn(),
      setPage: vi.fn(),
      input: { current: null },
    };
    show(<HistoryPage {...props} />);
    expect(screen.getByText("One original is missing.")).toBeTruthy();
    expect(screen.getByLabelText("History subtopic")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Ask about this policy/ }));
    expect(props.setPolicy).toHaveBeenCalledWith("delegation");
    expect(props.setQueryTopic).toHaveBeenCalledWith("pid");
    expect(props.setPage).toHaveBeenCalledWith("research");
  });

  test("the history page shows a loading state until the history arrives", () => {
    show(
      <HistoryPage
        catalog={CATALOG}
        policies={POLICIES}
        history={null}
        selectedHistory="bank_kyc"
        setSelectedHistory={noop}
        historyTopic=""
        setHistoryTopic={noop}
        day={CATALOG.today}
        setDay={noop}
        setPolicy={noop}
        setQueryTopic={noop}
        setPage={noop}
        input={{ current: null }}
      />,
    );
    expect(screen.getByText(/Loading policy history/)).toBeTruthy();
    expect(screen.queryByLabelText("History subtopic")).toBeNull();
  });

  test("the library orders policies and opens a policy's history", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ divisions: [] }))));
    const props = {
      policies: POLICIES,
      librarySort: "latest",
      setLibrarySort: vi.fn(),
      setSelectedHistory: vi.fn(),
      setHistoryTopic: vi.fn(),
      setPage: vi.fn(),
      setDivisionFocus: vi.fn(),
    };
    show(<LibraryPage {...props} />);
    const cards = document.querySelectorAll(".library-card");
    expect(cards[0].textContent).toContain("Bank KYC");
    expect(cards[0].textContent).toContain("2/2");
    fireEvent.click(cards[1]);
    expect(props.setSelectedHistory).toHaveBeenCalledWith("delegation");
    expect(props.setPage).toHaveBeenCalledWith("history");
    fireEvent.click(screen.getByRole("button", { name: "A to Z" }));
    expect(props.setLibrarySort).toHaveBeenCalledWith("az");
    await waitFor(() => expect(fetch).toHaveBeenCalled());
  });
});
