"""Streamlit controls and evidence review for all registered policy histories."""

from datetime import date
from policy_registry import registered_policies
import json

import streamlit as st


def policy_controls():
    policy_column, date_column = st.columns([2, 1])
    with policy_column:
        mode = st.selectbox(
            "Evidence route", ["Automatic", *(p.label for p in registered_policies()), "Standard search"],
            help="Automatic selects a registered policy history when recognised. Other topics use standard search. Select a policy explicitly for short follow-ups.",
        )
    with date_column:
        as_of = st.date_input(
            "Evidence as of", value=date.today(), min_value=date(1952, 1, 1), max_value=date.today(),
            help="For curated policies: include instructions issued by this date. Set this control for historical questions; it does not certify complete legal applicability.",
        )
    return mode, as_of


def render_policy_review(context):
    if not context:
        st.caption("Standard hybrid search · the history date filter is not applied; applicability is unverified")
        return
    label = context.get("policy_label", "JD")
    st.caption(f"{label} history · instructions issued by {context['as_of']}")
    st.warning(f"{label} evidence is incomplete. Source verification does not establish that every provision remains applicable.")
    if context.get("fallback_reason"):
        st.info(context["fallback_reason"])
    labels = {
        "source_unverified": "Original source not verified",
        "issued_effective_date_unverified": "Issued; commencement unverified",
        "not_yet_effective": "Not yet effective",
        "effective_date_established": "Commencement established; continued validity not certified",
    }
    with st.expander(f"{label} history, changes and evidence gaps"):
        if context["documents"]:
            checked = [d for d in context["documents"] if d["status"] != "source_unverified"]
            if checked:
                newest = max(checked, key=lambda d: d["issued_on"])
                st.caption(f"Most recent checked document in this inventory: {newest.get('title') or newest['id']} ({newest['issued_on']}). Later dated changes can follow a numbered SOP; this inventory does not certify the latest applicable policy.")
            st.dataframe([
                {"Version / document": d.get("version_label") or d.get("title") or d["id"], "Authority": d.get("authority", "EPFO document"), "Circular": d["circular_no"], "Issued": d["issued_on"],
                 "Effective from": d["effective_from"], "Evidence status": labels.get(d["status"], d["status"])}
                for d in sorted(context["documents"], key=lambda d: d["issued_on"], reverse=True)
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
            "Download policy evidence review", data=json.dumps(context, indent=2, ensure_ascii=False),
            file_name=f"{context.get('policy_id', 'jd')}-evidence-review.json", mime="application/json",
        )
