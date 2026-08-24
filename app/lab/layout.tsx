import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "7×7 generation lab · Pathagon",
  description: "Track Pathagon's 7×7 diverse-game generation corpus, Scout player, and learner evaluation gates.",
};

export default function LearningLabLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return children;
}
