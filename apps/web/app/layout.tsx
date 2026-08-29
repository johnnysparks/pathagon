import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: process.env.NEXT_PUBLIC_APP_URL
    ? new URL(process.env.NEXT_PUBLIC_APP_URL)
    : undefined,
  title: "Pathagon",
  description: "The Fuchs family path-building strategy game, preserved for the web.",
  openGraph: {
    title: "Pathagon",
    description: "Build the path. Break theirs.",
    images: [{ url: "/og.png", width: 1536, height: 1024, alt: "Pathagon wooden strategy game" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Pathagon",
    description: "Build the path. Break theirs.",
    images: ["/og.png"],
  },
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
