// meetscribe: MeetScribe lockup: the two-bubble mark next to the "MeetScribe" name.
// Real text instead of SVG text stretched to a fixed width, so it never overflows or gets clipped,
// whatever font the webview picks. `width` is the size the lockup is laid out for; it scales the
// text, and the lockup never grows past its container.
import HandyHand from "./HandyHand";

const WORD_1 = "Meet";
const WORD_2 = "Scribe";

const HandyTextLogo = ({
  width = 120,
  className = "",
  mark = true,
}: {
  width?: number;
  height?: number;
  className?: string;
  /** Show the bubble mark (off next to the mascot, which already is one). */
  mark?: boolean;
}) => {
  // "MeetScribe" at weight 700 is about 5.4 em wide, plus the mark and the gap
  const fontSize = Math.round(width / 7.1);
  return (
    <span
      role="img"
      aria-label="MeetScribe"
      className={`inline-flex max-w-full items-center gap-[0.3em] whitespace-nowrap font-bold leading-none tracking-tight ${className}`}
      style={{ fontSize }}
    >
      {mark && (
        <span aria-hidden="true" className="inline-flex shrink-0">
          <HandyHand width="1.2em" height="1.2em" />
        </span>
      )}
      {/* The brand name is not translated. */}
      <span aria-hidden="true">
        <span className="text-text">{WORD_1}</span>
        <span style={{ color: "#6366F1" }}>{WORD_2}</span>
      </span>
    </span>
  );
};

export default HandyTextLogo;
