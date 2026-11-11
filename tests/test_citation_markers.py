"""Citation styles that models emit are rewritten to the [n] form the interface and exports link."""

import pytest

from epfo.citation_provenance import normalise_citation_stream, normalise_citations
from epfo.policy_registry import select_policy


@pytest.mark.parametrize(
    'written, expected',
    [
        ('deployed on 10 April 2024【9】.', 'deployed on 10 April 2024[9].'),
        ('dated 05 February 2026【2†L1-L4】.', 'dated 05 February 2026[2].'),
        ('as stated 【 3 】', 'as stated [3]'),
        ('every quarter [10, p. 5] [11, p. 8].', 'every quarter [10] (p. 5) [11] (p. 8).'),
        ('without transfer [1] para 2, [2] para 8.', 'without transfer [1] para 2, [2] para 8.'),
        ('both apply [1, 2] and [3;4;5].', 'both apply [1][2] and [3][4][5].'),
        ('see [2: para 4]', 'see [2] (para 4)'),
        ('unchanged [1][6] and [12].', 'unchanged [1][6] and [12].'),
        (
            'a [link](https://example.com) and a list [a, b] stay',
            'a [link](https://example.com) and a list [a, b] stay',
        ),
        ('| Limit | See [1](#source-1) |', '| Limit | See [1](#source-1) |'),
        ('', ''),
    ],
)
def test_citation_styles_are_rewritten(written, expected):
    assert normalise_citations(written) == expected


@pytest.mark.parametrize(
    'answer',
    [
        'The limit is ₹1,00,000【9†source】 and was deployed on 10 April 2024 [1, 2]. Review is quarterly [10, p. 5].',
        'पेंशन दावा किसी भी शाखा के लिए संसाधित किया जा सकता है【1】।',
        'No citation here, only an opening [ that never closes',
    ],
)
@pytest.mark.parametrize('size', [1, 2, 3, 7, 1000])
def test_stream_gives_the_same_text_however_it_is_split(answer, size):
    parts = [answer[i : i + size] for i in range(0, len(answer), size)]
    assert ''.join(normalise_citation_stream(parts)) == normalise_citations(answer)


def test_stream_does_not_hold_ordinary_text_back():
    parts = list(normalise_citation_stream(['The limit ', 'is ₹1,00,000', '【9', '】.']))
    assert parts[:2] == ['The limit ', 'is ₹1,00,000'] and ''.join(parts).endswith('[9].')


@pytest.mark.parametrize(
    'question, policy',
    [
        (
            'What is the auto claim settlement limit for illness advances under para 68J, and when was it deployed?',
            'advances',
        ),
        (
            'How is pension worked out when an EPS member has concurrent employment in two establishments?',
            'eps_corrections',
        ),
        ('Who is delegated the power to permit an EPFO employee to go abroad on a personal visit?', 'foreign_visit'),
        (
            'For pension on higher wages, over what period is average monthly pay computed when pension commenced before 1 September 2014, and after it?',
            'higher_pension',
        ),
        (
            "Who is the competent authority for transfers in the Commissioners' cadre after the May 2022 amendment?",
            'staff_transfer',
        ),
        ('Can a member remove a member ID that was wrongly linked to their UAN?', 'mid_delinking'),
        # Wordings that must keep their earlier route.
        ('delegation of power to authorise claims out of inoperative accounts Account Officer', 'claim_authority'),
        ('two years weightage of service pension', 'pension_member_rules'),
        ('Does destination office need three level processing approval of Form 13?', 'transfers'),
    ],
)
def test_plain_wordings_reach_the_right_policy(question, policy):
    assert select_policy(question).id == policy
