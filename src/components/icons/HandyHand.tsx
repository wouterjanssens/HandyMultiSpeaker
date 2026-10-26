// meetscribe: MeetScribe's two-speech-bubble mark (replaces Handy's hand).
// Same geometry as the idle tray icon (scripts/meetscribe/make_icons.py).
const HandyHand = ({
  width,
  height,
}: {
  width?: number | string;
  height?: number | string;
  // The sidebar passes a className for its icons; the mark ignores it.
  className?: string;
}) => (
  <svg
    width={width || 126}
    height={height || 135}
    viewBox="0 0 64 64"
    className="fill-text stroke-text"
    xmlns="http://www.w3.org/2000/svg"
  >
    <mask id="meetscribe-mark-gap">
      <rect width="64" height="64" fill="#fff" stroke="none" />
      <rect
        x="3"
        y="19"
        width="44"
        height="35"
        rx="11"
        fill="#000"
        stroke="none"
      />
      <path d="M8 44 L28 44 L6 63 Z" fill="#000" stroke="none" />
    </mask>
    <g mask="url(#meetscribe-mark-gap)">
      <rect
        x="24.5"
        y="8.5"
        width="33"
        height="24"
        rx="8"
        fill="none"
        strokeWidth="5"
      />
      <path d="M43 33 L53 33 L55 42 Z" stroke="none" />
    </g>
    <rect
      x="8.5"
      y="24.5"
      width="33"
      height="24"
      rx="8"
      fill="none"
      strokeWidth="5"
    />
    <path d="M12 49 L23 49 L10 59 Z" stroke="none" />
  </svg>
);

export default HandyHand;
