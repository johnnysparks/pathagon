import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Leaderboard · Pathagon",
  description: "7×7 model rankings, live cross-play results, and promotion candidates.",
};

export default function LearningLabLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return children;
}
