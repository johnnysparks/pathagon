import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Learning lab · Pathagon",
  description: "Browse Pathagon GNN checkpoints, replay archives, and evaluation signals.",
};

export default function LearningLabLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return children;
}
