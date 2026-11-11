import { useState } from "react";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { useI18n } from "./UiLanguage.jsx";
import { api, post } from "./lib";

const EMPTY = { choice: null, sent: false, failed: false, comment: "" };

// Lets the reader say whether an answer helped. A "yes" is sent at once; a "no"
// offers an optional note first. The rating belongs to one answer (packetId).
export default function AnswerFeedback({ packetId, disabled }) {
  const { t: tr } = useI18n();
  const [saved, setSaved] = useState({ packetId: null, ...EMPTY });
  const state = saved.packetId === packetId ? saved : EMPTY;
  const update = (changes) => setSaved({ ...state, packetId, ...changes });

  async function send(helpful, comment = "") {
    update({ choice: helpful, sent: true, failed: false, comment });
    try {
      await api(
        "/api/feedback",
        post({ packet_id: packetId, helpful, comment }),
      );
    } catch {
      setSaved((current) =>
        current.packetId === packetId
          ? { ...current, sent: false, failed: true }
          : current,
      );
    }
  }

  if (state.sent)
    return (
      <span className="answer-feedback" role="status">
        {tr("Thank you for the feedback.")}
      </span>
    );
  return (
    <div className="answer-feedback">
      <span
        id="answer-feedback-label"
        title={tr("Your question and rating are recorded to improve this tool.")}
      >
        {tr("Was this answer helpful?")}
      </span>
      <button
        type="button"
        className="icon-btn"
        aria-label={tr("Yes, this answer helped")}
        disabled={disabled}
        onClick={() => send(true)}
      >
        <ThumbsUp size={15} />
      </button>
      <button
        type="button"
        className="icon-btn"
        aria-label={tr("No, this answer did not help")}
        aria-pressed={state.choice === false}
        disabled={disabled}
        onClick={() => update({ choice: false })}
      >
        <ThumbsDown size={15} />
      </button>
      {state.choice === false && (
        <form
          className="answer-feedback-note"
          onSubmit={(e) => {
            e.preventDefault();
            send(false, state.comment.trim());
          }}
        >
          <input
            aria-label={tr("What was wrong? (optional)")}
            placeholder={tr("What was wrong? (optional)")}
            maxLength={500}
            value={state.comment}
            onChange={(e) => update({ comment: e.target.value })}
          />
          <button className="text-button">{tr("Send")}</button>
        </form>
      )}
      {state.failed && (
        <small role="alert">{tr("Feedback could not be sent.")}</small>
      )}
      <small className="answer-feedback-notice">
        {tr("Your question and rating are recorded to improve this tool.")}
      </small>
    </div>
  );
}
