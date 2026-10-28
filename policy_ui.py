"""Streamlit controls and evidence review for the JD-only policy route."""

from datetime import date
import json

import streamlit as st


def policy_controls():
    policy_column, date_column = st.columns([2, 1])
    with policy_column:
        mode = st.selectbox(
            "Evidence route", ["Automatic", "JD member profile", "Standard search"],
            help="Automatic uses JD history for member-profile questions and standard search for other policies. Select JD member profile for short follow-ups.",
        )
    with date_column:
        as_of = st.date_input(
            "JD evidence as of", value=date.today(),
            help="JD only: include instructions issued by this date. Set this control for historical questions; it does not certify complete legal applicability.",
        )
    return mode, as_of


def render_policy_review(context):
    if not context:
        st.caption("Standard hybrid search · policy applicability has not been verified")
        return
    st.caption(f"JD policy history · instructions issued by {context['as_of']}")
    st.warning("JD evidence is incomplete. Source verification does not establish that every provision remains applicable.")
    if context.get("fallback_reason"):
        st.info(context["fallback_reason"])
    labels = {
        "source_unverified": "Original source not verified",
        "issued_effective_date_unverified": "Issued; commencement unverified",
        "not_yet_effective": "Not yet effective",
        "effective_date_established": "Commencement established; continued validity not certified",
    }
    with st.expander("JD history, changes and evidence gaps"):
        if context["documents"]:
            st.dataframe([
                {"Circular": d["circular_no"], "Issued": d["issued_on"],
                 "Effective from": d["effective_from"], "Evidence status": labels.get(d["status"], d["status"])}
                for d in context["documents"]
            ], use_container_width=True)
        for relation in context["relationships"]:
            status = "Source checked; commencement unresolved"
            if relation["status"] == "relationship_unverified":
                status = "Relationship target unverified"
            elif relation["status"] == "effective_relationship":
                status = "Relationship and commencement established"
            elif relation["status"] == "not_yet_effective":
                status = "Not yet effective"
            st.write(f"{relation['source']} → {relation['target']}: {relation['type'].replace('_', ' ')}")
            st.caption(status + ". " + relation["scope"])
        for limitation in context["limitations"]:
            st.write("• " + limitation)
        st.download_button(
            "Download JD evidence review", data=json.dumps(context, indent=2, ensure_ascii=False),
            file_name="jd-evidence-review.json", mime="application/json",
        )
