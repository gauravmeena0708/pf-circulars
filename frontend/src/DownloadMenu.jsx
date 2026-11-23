import React, {
  useEffect,
  useLayoutEffect,
  useId,
  useRef,
  useState,
} from "react";
import { ChevronDown, Download, LoaderCircle } from "lucide-react";
import { useI18n } from "./UiLanguage.jsx";

export default function DownloadMenu({
  items,
  disabled = false,
  label = "Download",
  ariaLabel = label,
}) {
  const { t: tr } = useI18n();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [position, setPosition] = useState({ visibility: "hidden" });
  const trigger = useRef(null),
    panel = useRef(null);
  const id = useId();
  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const rect = trigger.current.getBoundingClientRect();
      const width = Math.min(260, window.innerWidth - 24);
      const height = Math.min(
        panel.current.scrollHeight,
        window.innerHeight - 24,
      );
      const left = Math.max(
        12,
        Math.min(rect.right - width, window.innerWidth - width - 12),
      );
      const below = rect.bottom + 8;
      const top =
        below + height <= window.innerHeight - 12
          ? below
          : Math.max(12, rect.top - height - 8);
      setPosition({ left, top, width, visibility: "visible" });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open]);
  useEffect(() => {
    if (!open) return;
    const outside = (event) => {
      if (
        !trigger.current?.contains(event.target) &&
        !panel.current?.contains(event.target)
      )
        setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [open]);
  useLayoutEffect(() => {
    if (open && position.visibility === "visible") {
      panel.current
        ?.querySelector('[role="menuitem"]')
        ?.focus({ preventScroll: true });
    }
  }, [open, position.visibility]);
  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);
  const close = () => {
    setOpen(false);
    trigger.current?.focus({ preventScroll: true });
  };
  return (
    <div className="download-control">
      <button
        ref={trigger}
        type="button"
        className="secondary"
        aria-label={tr(ariaLabel)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        aria-busy={busy}
        disabled={disabled || busy}
        onClick={() => setOpen((value) => !value)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            setOpen(true);
          }
        }}
      >
        {busy ? (
          <LoaderCircle size={16} className="spin" />
        ) : (
          <Download size={16} />
        )}
        {tr(busy ? "Preparing download…" : label)}
        <ChevronDown size={14} />
      </button>
      {open && (
        <div
          id={id}
          ref={panel}
          role="menu"
          aria-label={tr(ariaLabel)}
          className="download-menu"
          style={position}
          onKeyDown={(event) => {
            const buttons = [
              ...event.currentTarget.querySelectorAll('[role="menuitem"]'),
            ];
            const current = buttons.indexOf(document.activeElement);
            if (event.key === "Escape") {
              event.preventDefault();
              event.stopPropagation();
              close();
            } else if (
              ["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)
            ) {
              event.preventDefault();
              const index =
                event.key === "Home"
                  ? 0
                  : event.key === "End"
                    ? buttons.length - 1
                    : (current +
                        (event.key === "ArrowDown" ? 1 : -1) +
                        buttons.length) %
                      buttons.length;
              buttons[index]?.focus();
            } else if (event.key === "Tab") {
              setTimeout(() => setOpen(false), 0);
            }
          }}
        >
          {items.map((item) => (
            <button
              key={item.id}
              type="button"
              role="menuitem"
              tabIndex={-1}
              onClick={async () => {
                close();
                setBusy(true);
                try {
                  await item.onSelect();
                } finally {
                  setBusy(false);
                  if (document.activeElement === document.body) {
                    requestAnimationFrame(() =>
                      trigger.current?.focus({ preventScroll: true }),
                    );
                  }
                }
              }}
            >
              <span>{tr(item.label)}</span>
              {item.description && <small>{tr(item.description)}</small>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
