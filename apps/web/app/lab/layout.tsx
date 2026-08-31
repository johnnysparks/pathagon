import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Leaderboard · Pathagon",
  description: "7×7 Rust/WASM model rankings and research evidence; research-only archives remain unranked.",
};

export default function LearningLabLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return children;
}
