"""The circular listing page is merged into circular-data.json without duplicating or dropping records."""

import json

from scripts import fetch


def box(title, division, date, url):
    return (
        '<div class="download-box" data-division="x" data-year="2026" data-month="6">'
        f'<span class="serial-num">1.</span><div class="left-area"><h3 class="download-title">{title}</h3>'
        f'<div class="meta-row"><span class="meta-item"><i class="fa fa-file-lines"></i>{division}</span>'
        f'<span class="meta-sep">.</span><span class="meta-item">{date}</span></div></div>'
        f'<a href="{url}" class="view-pdf-btn" target="_blank">View PDF</a></div>'
    )


CDN = "https://pmvbry-cdn.epfindia.gov.in/wp-content/uploads/2026/06/"
OLD = "https://www.epfindia.gov.in/site_docs/PDFs/Circulars/Y2025-2026/"

EXISTING = [
    {
        "serial_no": "5",
        "title": "Final Seniority list in the cadre of ACC as on 09.10.2025 - Regarding",
        "circular_no": "No. HRM-I/S-4(1)95/S.L./806",
        "date": "24/10/2025",
        "hindi_pdf_link": None,
        "english_pdf_link": OLD + "SeniorityList_ACC.pdf",
    },
    {
        "serial_no": "4",
        "title": "Probationary Examination for SSAs - Regarding File",
        "circular_no": "File No.33/7/2023-24/SSAs(DR)Prob.Exam-2021/168",
        "date": "13/06/2025",
        "hindi_pdf_link": None,
        "english_pdf_link": OLD + "SSAs_ProbationaryExamNotification_2025.pdf",
    },
    {
        "serial_no": "3",
        "title": "Regulation of claims in reference to Para 22(3) of EDLI Scheme, 1976",
        "circular_no": "",
        "date": "17/12/2025",
        "hindi_pdf_link": None,
        "english_pdf_link": OLD + "Circular_EDLIPara22(3)_EDLI1976.pdf",
    },
    {
        "serial_no": "2",
        "title": "Publicity Campaigns of EPFO HO",
        "circular_no": "",
        "date": "13/06/2017",
        "hindi_pdf_link": None,
        "english_pdf_link": OLD + "PublicityCampaigns_2017.pdf",
    },
]

PAGE = (
    "<html><body>"
    + "".join(
        [
            box(
                "Revised procedure for settlement of death claims - Regarding No. WSU/2026/Death/41 dated 02/06/2026 Annexure-I",
                "WSU",
                "05/06/2026",
                CDN + "212.pdf",
            ),
            box(
                "Final Seniority list in the cadre of ACC as on 09.10.2025 - Regarding No. HRM-I/S-4(1)95/S.L./806 dated 24/10/2025",
                "HRM",
                "",
                CDN + "157.pdf",
            ),
            box(
                "Probationary Examination for SSAs - Regarding File No.33/7/2023-24/SSAs(DR)Prob.Exam-2021/168 dated 13/06/2025",
                "PDUNASS",
                "",
                CDN + "56.pdf",
            ),
            box(
                "clarification on treatment of weekends closed holidays",
                "EDLI",
                "07/09/2015",
                CDN + "Circular_EDLIPara223_EDLI1976-14.pdf",
            ),
            box("Publicity Campaigns of EPFO HO (Hindi)", "C&amp;PR", "15/06/2017", CDN + "hindi-copy-of-campaign.pdf"),
            box("Campaign_Social_Media_Handles_23112022", "C&amp;PR", "22/11/2022", "http://false"),
            box("Quarterly review of grievance disposal", "CSD", "31/02/2026", CDN + "QuarterlyReview_Grievance.pdf"),
        ]
    )
    + "</body></html>"
)


def test_listing_entries_are_parsed_in_page_order():
    listed = fetch.parse_listing(PAGE)
    assert len(listed) == 7
    assert listed[0] == {
        "title": "Revised procedure for settlement of death claims - Regarding No. WSU/2026/Death/41 dated 02/06/2026 Annexure-I",
        "division": "WSU",
        "date": "05/06/2026",
        "url": CDN + "212.pdf",
    }
    assert listed[4]["division"] == "C&PR"
    # An impossible date is not kept as a date.
    assert listed[6]["date"] == ""


def test_only_unseen_circulars_with_a_pdf_are_added():
    records, added = fetch.merge(EXISTING, fetch.parse_listing(PAGE))
    assert [record["english_pdf_link"] for record in added] == [CDN + "212.pdf", CDN + "QuarterlyReview_Grievance.pdf"]
    # Existing records keep their content and order, after the new ones.
    assert records[len(added) :] == EXISTING
    assert records[: len(added)] == added


def test_new_record_takes_number_and_letter_date_from_the_title():
    _, added = fetch.merge(EXISTING, fetch.parse_listing(PAGE))
    assert added[0] == {
        "serial_no": "",
        "title": "Revised procedure for settlement of death claims - Regarding No. WSU/2026/Death/41 dated 02/06/2026 Annexure-I",
        "circular_no": "WSU/2026/Death/41",
        "date": "02/06/2026",
        "hindi_pdf_link": None,
        "english_pdf_link": CDN + "212.pdf",
        "division": "WSU",
    }
    assert added[1]["date"] == "" and added[1]["circular_no"] == ""


def test_merge_is_repeatable():
    records, added = fetch.merge(EXISTING, fetch.parse_listing(PAGE))
    again, added_again = fetch.merge(records, fetch.parse_listing(PAGE))
    assert added_again == [] and again == records


def test_short_page_is_refused_and_data_file_left_alone(tmp_path):
    page, data = tmp_path / "page.html", tmp_path / "circular-data.json"
    page.write_text(PAGE, encoding="utf-8")
    data.write_text(json.dumps(EXISTING), encoding="utf-8")
    try:
        fetch.fetch_circular_metadata(html_path=str(page), data_file=str(data))
    except SystemExit as error:
        assert "Too few circulars" in str(error)
    else:
        raise AssertionError("A short listing must not be accepted")
    assert json.loads(data.read_text(encoding="utf-8")) == EXISTING


def test_new_circulars_are_written_and_dry_run_writes_nothing(tmp_path, monkeypatch):
    page, data = tmp_path / "page.html", tmp_path / "circular-data.json"
    page.write_text(PAGE, encoding="utf-8")
    data.write_text(json.dumps(EXISTING), encoding="utf-8")
    monkeypatch.setattr(fetch, "MIN_EXPECTED_ENTRIES", 1)
    assert len(fetch.fetch_circular_metadata(html_path=str(page), data_file=str(data), dry_run=True)) == 2
    assert json.loads(data.read_text(encoding="utf-8")) == EXISTING
    fetch.fetch_circular_metadata(html_path=str(page), data_file=str(data))
    saved = json.loads(data.read_text(encoding="utf-8"))
    assert len(saved) == len(EXISTING) + 2 and saved[2:] == EXISTING
