import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "7×7 model leaderboard · Pathagon",
  description: "Track Pathagon's 7×7 model progression, provisional standings, and cross-play promotion gates.",
};

export default function LearningLabLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return children;
}
