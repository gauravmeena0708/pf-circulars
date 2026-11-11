import io
import unicodedata
import zipfile
from xml.etree import ElementTree

import fitz

from epfo.docx_export import create_research_report_docx
from epfo.pdf_export import create_chat_transcript_pdf, create_research_report_pdf, create_text_document_pdf


def pdf_text(document):
    # PDF typography may encode Latin ligatures such as fi as one Unicode glyph.
    return unicodedata.normalize('NFKC', '\n'.join(page.get_text() for page in document))


def test_research_pdf_paginates_and_preserves_evidence_and_clickable_citations():
    url = 'https://www.epfo.gov.in/source.pdf'
    answer = (
        'A procedure with a material condition [1].\n\n' * 160
        + '| Item | Requirement |\n| --- | --- |\n| Evidence | Original page |\n\n'
        + 'Evidence cutoff: 2025-04-03. Applicability is not certified.'
    )
    data = create_research_report_pdf(
        'Which procedure?',
        answer,
        [
            {
                'title': 'Official instruction',
                'source': 'Circular ABC | Physical page: 2',
                'date': '2025-04-03',
                'text': 'The checked source excerpt.',
                'url': url,
            }
        ],
    )
    assert data.startswith(b'%PDF')
    with fitz.open(stream=data, filetype='pdf') as document:
        assert len(document) >= 3
        text = pdf_text(document)
        for expected in (
            'Which procedure?',
            '2025-04-03',
            'Applicability is not certified',
            'Physical page: 2',
            'Original page',
            'The checked source excerpt.',
        ):
            assert expected in text
        links = [link for page in document for link in page.get_links()]
        assert any(link.get('uri') == url for link in links)
        assert any(link['kind'] == fitz.LINK_GOTO and link['page'] == len(document) - 1 for link in links)
        assert f'Page {len(document)} of {len(document)}' in document[-1].get_text()


def test_hindi_is_rendered_with_an_embedded_devanagari_font():
    data = create_text_document_pdf(
        'भारत — कार्यालय टिप्पणी', 'सदस्य विवरण में सुधार के लिए आवेदन करें। राशि ₹45,000 है।', {'Question': 'सुधार कैसे करें?'}
    )
    with fitz.open(stream=data, filetype='pdf') as document:
        assert 'भारत' in document[0].get_text()
        assert any('Devanagari' in font[3] for page in document for font in page.get_fonts())
        assert '45,000' in document[0].get_text()


def test_untrusted_html_images_and_local_links_cannot_become_active_pdf_content():
    data = create_research_report_pdf(
        '<script>title</script>',
        '<img src="file:///etc/passwd">\n\n[local](file:///etc/passwd)\n\n![image](https://example.com/remote.png)',
        [{'title': '<b>Untrusted title</b>', 'url': 'file:///etc/passwd', 'text': 'Literal source text.'}],
    )
    with fitz.open(stream=data, filetype='pdf') as document:
        assert not document.embfile_count()
        assert not any(page.get_images() for page in document)
        assert not any(
            link.get('uri', '').startswith(('file:', 'javascript:', 'data:'))
            for page in document
            for link in page.get_links()
        )
        assert '<b>Untrusted title</b>' in document[0].get_text()


def test_conversation_pdf_contains_questions_responses_and_source_filename():
    data = create_chat_transcript_pdf(
        'Office notes',
        [
            {'role': 'user', 'content': 'Which action is proposed?'},
            {'role': 'assistant', 'content': 'Review the original file before approval.'},
        ],
        {'Source file': 'office-noting.pdf'},
    )
    with fitz.open(stream=data, filetype='pdf') as document:
        text = pdf_text(document)
        assert 'office-noting.pdf' in text and 'Which action is proposed?' in text
        assert 'Review the original file before approval.' in text


def test_word_source_urls_are_actual_clickable_hyperlinks():
    url = 'https://www.epfo.gov.in/source.pdf?part=1&lang=en'
    data = create_research_report_docx('Question', 'Answer [1].', [{'title': 'Source', 'url': url}])
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        relationships = ElementTree.fromstring(archive.read('word/_rels/document.xml.rels'))
        assert any(item.get('Target') == url and item.get('Type', '').endswith('/hyperlink') for item in relationships)
        assert b'w:hyperlink' in archive.read('word/document.xml')
