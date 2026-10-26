/**
 * meetscribe: MeetScribe's mascot, a friendly speech bubble (replaces Handy's
 * glove). The outline and face follow the theme (`mascot-line`). Keeps the
 * old 151×165 box so `Mascot`'s sizing and wave pivot still fit.
 */
const HandyMascot = ({ className = "" }: { className?: string }) => {
  return (
    <svg
      viewBox="0 0 151 165"
      className={className}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M52 14 H99 A40 40 0 0 1 139 54 V80 A40 40 0 0 1 99 120 H78 L30 156 L50 120 A38 38 0 0 1 12 80 V54 A40 40 0 0 1 52 14 Z"
        fill="#fff"
        className="stroke-mascot-line"
        strokeWidth="7"
        strokeLinejoin="round"
      />
      <circle cx="55" cy="62" r="8" className="fill-mascot-line" />
      <circle cx="96" cy="62" r="8" className="fill-mascot-line" />
      <circle cx="40" cy="82" r="7" fill="#A5B4FC" />
      <circle cx="111" cy="82" r="7" fill="#A5B4FC" />
      <path
        d="M59 84 Q75.5 100 92 84"
        className="stroke-mascot-line"
        strokeWidth="6"
        strokeLinecap="round"
      />
    </svg>
  );
};

export default HandyMascot;
