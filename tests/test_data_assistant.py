import unittest
import pandas as pd
import numpy as np

from data_assistant import (
    DataExtractionError,
    load_csv_dataframe,
    generate_dataset_profile,
    search_dataframe,
    prepare_dataframe_llm_context,
)


class DataAssistantTests(unittest.TestCase):
    def setUp(self):
        self.sample_csv_text = """Issue_ID,Issue_Topic,Category,Functional_Owner,Status,Count_Affected
101,Form 13 Transfer Request stuck at employer,Transfers,Field Office Delhi,Open,15
102,DSC registration error on Employer Portal,DSC,Tech Backend,In Progress,8
103,Form 13 Member Passbook not updating after transfer,Transfers,Tech Backend,Open,24
104,Joint Declaration name mismatch,Member Profile,Field Office Mumbai,Resolved,5
105,Form 19 Claim rejection due to KYC,Claims,Settlement Team,Open,12
106,Form 13 Annexure K generation failed,Transfers,Tech Backend,Resolved,3
"""
        self.sample_csv_bytes = self.sample_csv_text.encode("utf-8")

    def test_load_valid_csv(self):
        df = load_csv_dataframe(self.sample_csv_bytes, "cites_sample.csv")
        self.assertEqual(len(df), 6)
        self.assertIn("Issue_Topic", df.columns)
        self.assertIn("Functional_Owner", df.columns)

    def test_arbitrary_schema_and_double_csv_suffix_are_supported(self):
        location_csv = (
            b"place,latitude,longitude,population\n"
            b"North Point,28.61,77.21,1200\n"
            b"South Point,19.08,72.88,900\n"
        )
        df = load_csv_dataframe(
            location_csv,
            "~/Downloads/location.csv - location.csv.csv",
        )

        self.assertEqual(list(df.columns), ["place", "latitude", "longitude", "population"])
        self.assertEqual(len(df), 2)

    def test_domain_words_are_not_discarded_from_deterministic_search(self):
        df = load_csv_dataframe(b"category\nissues\nother\n", "generic.csv")
        context = prepare_dataframe_llm_context(df, "How many issues are there?")

        self.assertIn("Exactly **1** matching row(s) found", context)

    def test_load_empty_csv_raises_error(self):
        with self.assertRaises(DataExtractionError):
            load_csv_dataframe(b"", "empty.csv")
        with self.assertRaises(DataExtractionError):
            load_csv_dataframe(b"   \n  ", "whitespace.csv")

    def test_load_non_csv_and_malformed_rows_raise_error(self):
        with self.assertRaises(DataExtractionError):
            load_csv_dataframe(b"not actually csv", "plain.txt")
        with self.assertRaises(DataExtractionError):
            load_csv_dataframe(b"a,b\n1,2\n3,4,5\n6,7\n", "broken.csv")

    def test_duplicate_trimmed_columns_are_made_unique(self):
        df = load_csv_dataframe(b"Name, Name\nAlice,Bob\n", "duplicates.csv")
        self.assertEqual(list(df.columns), ["Name", "Name_2"])
        profile = generate_dataset_profile(df)
        self.assertEqual(profile.column_count, 2)

    def test_load_semicolon_delimited_csv(self):
        semicolon_csv = b"ColA;ColB;ColC\n1;2;3\n4;5;6\n"
        df = load_csv_dataframe(semicolon_csv, "semi.csv")
        self.assertEqual(len(df), 2)
        self.assertEqual(len(df.columns), 3)
        self.assertIn("ColA", df.columns)

    def test_dataset_profile_generation(self):
        df = load_csv_dataframe(self.sample_csv_bytes)
        profile = generate_dataset_profile(df)
        self.assertEqual(profile.row_count, 6)
        self.assertEqual(profile.column_count, 6)
        self.assertIn("Count_Affected", profile.numeric_columns)
        self.assertIn("Issue_Topic", profile.categorical_columns)
        self.assertIn("Count_Affected", profile.summary_stats)
        self.assertEqual(profile.summary_stats["Count_Affected"]["min"], 3.0)
        self.assertEqual(profile.summary_stats["Count_Affected"]["max"], 24.0)

    def test_search_dataframe_form_13_exact_count(self):
        df = load_csv_dataframe(self.sample_csv_bytes)
        matched_df, total_matches, breakdown = search_dataframe(df, "Form 13")
        
        # In our sample CSV: rows 101, 103, 106 have "Form 13" in Issue_Topic
        self.assertEqual(total_matches, 3)
        self.assertEqual(len(matched_df), 3)
        self.assertEqual(breakdown.get("Issue_Topic"), 3)

    def test_search_dataframe_case_insensitive(self):
        df = load_csv_dataframe(self.sample_csv_bytes)
        matched_df, total_matches, _ = search_dataframe(df, "form 13", case_sensitive=False)
        self.assertEqual(total_matches, 3)

    def test_search_dataframe_owner_filter(self):
        df = load_csv_dataframe(self.sample_csv_bytes)
        matched_df, total_matches, _ = search_dataframe(df, "Tech Backend")
        self.assertEqual(total_matches, 3)

    def test_prepare_dataframe_llm_context_includes_deterministic_count(self):
        df = load_csv_dataframe(self.sample_csv_bytes)
        user_query = "How many Form 13 issues are there in the dataset?"
        context = prepare_dataframe_llm_context(df, user_query)
        
        self.assertIn("Dataset Structure", context)
        self.assertIn("Form 13", context)
        self.assertIn("Exactly **3** matching row(s) found", context)

    def test_prepare_context_includes_combined_filter_count(self):
        df = load_csv_dataframe(self.sample_csv_bytes)
        context = prepare_dataframe_llm_context(
            df,
            "How many Form 13 issues are Open?",
        )

        self.assertIn("Combined filter", context)
        self.assertIn("Exactly **2** row(s) match all identified terms", context)

    def test_prepare_context_respects_character_limit(self):
        df = pd.DataFrame(
            {f"Column_{index}": ["x" * 200 for _ in range(20)] for index in range(40)}
        )
        context = prepare_dataframe_llm_context(
            df,
            "summarize this dataset",
            max_context_chars=2_500,
        )

        self.assertLessEqual(len(context), 2_500)
        self.assertIn("dataset context truncated", context)


if __name__ == "__main__":
    unittest.main()
