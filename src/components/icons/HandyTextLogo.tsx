// meetscribe: plain "MeetScribe" wordmark (replaces Handy's hand-lettered logo).
// Keeps the old 930×328 box so existing width/height props lay out the same.
const WORD_1 = "Meet";
const WORD_2 = "Scribe";

const HandyTextLogo = ({
  width,
  height,
  className,
}: {
  width?: number;
  height?: number;
  className?: string;
}) => {
  return (
    <svg
      width={width}
      height={height}
      className={className}
      viewBox="0 0 930 328"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="MeetScribe"
    >
      <text
        x="465"
        y="222"
        textAnchor="middle"
        textLength="840"
        lengthAdjust="spacingAndGlyphs"
        fontFamily="ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"
        fontSize="190"
        fontWeight="700"
      >
        {/* The brand name is not translated. */}
        <tspan className="fill-text">{WORD_1}</tspan>
        <tspan fill="#6366F1">{WORD_2}</tspan>
      </text>
    </svg>
  );
};

export default HandyTextLogo;
